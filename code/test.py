import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

from dataset import VectorMapDataset, collate_fn
from models.transformer import MapTransformer
from models.loss import SetCriterion
from models.matcher import HungarianMatcher


val_data_path = "data/val/"
save_path = "C:/data/result/"
os.makedirs(save_path, exist_ok=True)

    num_queries = 50
    max_trips = 10
    max_lines = 50
    points_per_line = 10


vec_root = os.path.join(val_data_path, "vec")
sample_names = sorted(os.listdir(vec_root))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MapTransformer(num_classes=3, max_trips=max_trips, max_lines=max_lines, points_per_line=points_per_line, num_queries=num_queries).to(device)
model.load_state_dict(torch.load("checkpoints/model_epoch_500.pth", map_location=device))
model.eval()


val_dataset = VectorMapDataset(val_data_path, max_trips=max_trips, max_lines=max_lines, points_per_line=points_per_line)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


class_id_to_name = {
    0: "lane_marking",
    1: "stop_location",
    2: "cross_walk"
}

def save_normalized_samples(dataset, sample_names, save_dir, class_id_to_name):
    print("Saving normalized (denormalized) ground-truth samples...")

    for idx, sample_name in enumerate(sample_names):
        item = dataset[idx]
        bounds = item["bounds"]
        label_tensor = item["label_tensor"]

        lines = []
        for polyline in label_tensor:
            coords = polyline[:, :2].numpy()
            class_id = int(polyline[0, 2].item())

            if class_id < 0:
                continue
            if np.all(coords == 0):
                continue

            coords[:, 0] = coords[:, 0] * (bounds["x_max"] - bounds["x_min"]) + bounds["x_min"]
            coords[:, 1] = coords[:, 1] * (bounds["y_max"] - bounds["y_min"]) + bounds["y_min"]

            label_name = class_id_to_name.get(class_id, f"unknown_{class_id}")
            coord_str = ", ".join([f"{x:.6f} {y:.6f}" for x, y in coords])
            line = f"\"LINESTRING({coord_str})\"\t\"{label_name}\""
            lines.append(line)

        new_file = os.path.join(save_dir, f"{sample_name}_new.txt")
        with open(new_file, "w", encoding="utf-8") as f:
            f.write("\"geom\"\t\"type\"\n")
            f.write("\n".join(lines))

    print(f"All normalized samples saved to: {save_dir}")

save_normalized_samples(val_dataset, sample_names, save_path, class_id_to_name)


matcher = HungarianMatcher(cost_class=1, cost_polyline=1, cost_direction=1)
criterion = SetCriterion(num_classes=3, matcher=matcher, weight_dict={'loss_ce': 1, 'loss_polyline': 3, 'loss_direction': 1})


total_loss = 0.0
total_batches = 0
label_counter = defaultdict(int)

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        input_tensor = batch["input_tensor"].to(device)
        input_mask = batch["input_mask"].to(device)
        bounds_list = batch["bounds"]
        batch_size = input_tensor.size(0)

        outputs_class, outputs_coords = model(input_tensor, input_mask)
        probs = torch.softmax(outputs_class, dim=-1)
        scores, labels = probs.max(dim=-1)
        pred_polylines = outputs_coords.view(batch_size, num_queries, points_per_line, 2)

        batch_polylines = pred_polylines.cpu().numpy()
        batch_labels = labels.cpu().numpy()

        for b in range(batch_size):
            bounds = bounds_list[b]
            lines = []

            for i in range(batch_labels.shape[1]):
                class_id = batch_labels[b, i]
                label_name = class_id_to_name.get(class_id, f"unknown_{class_id}")
                label_counter[label_name] += 1

                coords = batch_polylines[b, i]

                x_min, x_max = bounds["x_min"], bounds["x_max"]
                y_min, y_max = bounds["y_min"], bounds["y_max"]
                coords[:, 0] = coords[:, 0] * (x_max - x_min) + x_min
                coords[:, 1] = coords[:, 1] * (y_max - y_min) + y_min

                coord_str = ", ".join([f"{x:.6f} {y:.6f}" for x, y in coords])
                line = f"\"LINESTRING({coord_str})\"\t\"{label_name}\""
                lines.append(line)

            sample_name = sample_names[batch_idx * batch_size + b]
            save_file = os.path.join(save_path, f"{sample_name}.txt")
            with open(save_file, "w", encoding="utf-8") as f:
                f.write("\"geom\"\t\"type\"\n")
                f.write("\n".join(lines))

        label_tensor = batch['label_tensor'].to(device)
        label_mask = batch['label_mask'].to(device)
        label_classes = label_tensor[..., 2].long()
        label_coords = label_tensor[..., :2]

        targets = [{
            'labels': label_classes[b].t()[0][~label_mask[b]],
            'polylines': label_coords[b].view(-1, points_per_line, 2)[~label_mask[b]]
        } for b in range(batch_size)]

        outputs_dict = {
            'pred_logits': outputs_class,
            'pred_polylines': outputs_coords.view(batch_size, num_queries, points_per_line, 2)
        }

        loss_dict = criterion(outputs_dict, targets)
        loss_value = sum(loss_dict.values()).item()
        total_loss += loss_value
        total_batches += 1

        print(f"[Batch {batch_idx + 1}] Loss: "
              f"CE={loss_dict['loss_ce']:.4f}, "
              f"Polyline={loss_dict['loss_polyline']:.4f}, "
              f"Direction={loss_dict['loss_direction']:.4f}")


avg_loss = total_loss / total_batches
print(f"\nAll predictions saved to: {save_path}")
print(f"Average Validation Loss: {avg_loss:.4f}")

print("\nPrediction Label Distribution:")
for label, count in label_counter.items():
    print(f"  {label:<15}: {count}")
