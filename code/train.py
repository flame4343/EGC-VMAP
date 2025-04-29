import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from dataset import VectorMapDataset, collate_fn
from models.transformer import MapTransformer
from models.loss import SetCriterion
from models.matcher import HungarianMatcher


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    run_dir = "runs"
    version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir = os.path.join(run_dir, f"logs-{version}")
    writer = SummaryWriter(log_dir=log_dir)

    total_epochs = 200
    save_interval = 10
    train_batch_size = 8
    val_batch_size = 8

    num_queries = 50
    max_trips = 10
    max_lines = 50
    points_per_line = 10


    train_dataset = VectorMapDataset("data/train/", max_trips, max_lines, points_per_line)
    val_dataset = VectorMapDataset("data/val/", max_trips, max_lines, points_per_line)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    print('Data loading complete.')


    model = MapTransformer(max_trips=max_trips, max_lines=max_lines, points_per_line=points_per_line,
                           num_queries=num_queries, num_classes=3).to(device)
    matcher = HungarianMatcher(cost_class=1, cost_polyline=1)
    criterion = SetCriterion(num_classes=3, matcher=matcher, weight_dict={'loss_ce': 1, 'loss_polyline': 3, 'loss_direction': 1})
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=5e-5)

    for epoch in range(total_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            input_tensor = batch['input_tensor'].to(device)
            label_tensor = batch['label_tensor'].to(device)
            input_mask = batch['input_mask'].to(device)
            label_mask = batch['label_mask'].to(device)

            outputs_class, outputs_coords = model(input_tensor, input_mask)
            outputs = {
                'pred_logits': outputs_class,
                'pred_polylines': outputs_coords.view(train_batch_size, num_queries, points_per_line, 2)
            }

            label_classes = label_tensor[..., 2].long()
            label_coords = label_tensor[..., :2]

            targets = [{
                'labels': label_classes[b].t()[0][~label_mask[b]],
                'polylines': label_coords[b].view(-1, points_per_line, 2)[~label_mask[b]]
            } for b in range(label_classes.size(0))]

            loss_dict = criterion(outputs, targets)
            print(f"Train Loss: CE={loss_dict['loss_ce']:.4f}, Polyline={loss_dict['loss_polyline']:.4f}, Direction={loss_dict['loss_direction']:.4f}")
            loss = sum(loss_dict.values())
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch + 1)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_tensor = batch['input_tensor'].to(device)
                label_tensor = batch['label_tensor'].to(device)
                input_mask = batch['input_mask'].to(device)
                label_mask = batch['label_mask'].to(device)

                outputs_class, outputs_coords = model(input_tensor, input_mask)
                outputs = {
                    'pred_logits': outputs_class,
                    'pred_polylines': outputs_coords.view(val_batch_size, num_queries, points_per_line, 2)
                }

                label_classes = label_tensor[..., 2].long()
                label_coords = label_tensor[..., :2]

                targets = [{
                    'labels': label_classes[b].t()[0][~label_mask[b]],
                    'polylines': label_coords[b].view(-1, points_per_line, 2)[~label_mask[b]]
                } for b in range(label_classes.size(0))]

                loss_dict = criterion(outputs, targets)
                print(f"Val Loss: CE={loss_dict['loss_ce']:.4f}, Polyline={loss_dict['loss_polyline']:.4f}, Direction={loss_dict['loss_direction']:.4f}")
                loss = sum(loss_dict.values())
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")
        writer.add_scalar('Loss/validation', val_loss, epoch + 1)

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved: {save_path}")

    writer.close()
