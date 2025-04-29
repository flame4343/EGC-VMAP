import os
import re
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

class VectorMapDataset(Dataset):
    """
    A dataset class for vectorized map data (polylines and labels).
    """
    def __init__(self, data_folder, max_trips=5, max_lines=20, points_per_line=50):
        """
        Args:
            data_folder (str): Root path containing 'vec' and 'label' subfolders.
            max_trips (int): Maximum trips per scene.
            max_lines (int): Maximum lines per trip.
            points_per_line (int): Number of points per line.
        """
        self.vec_folder = os.path.join(data_folder, "vec")
        self.label_folder = os.path.join(data_folder, "label")
        self.max_trips = max_trips
        self.max_lines = max_lines
        self.points_per_line = points_per_line
        self.maps = self._load_data_from_folders()

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        map_data = self.maps[idx]
        vec_orders = map_data["vec_orders"]
        label_order_lines = map_data["label_order_lines"]
        bounds = map_data["bounds"]

        input_tensor = torch.zeros((self.max_trips, self.max_lines, self.points_per_line, 3))
        label_tensor = torch.zeros((len(label_order_lines), self.points_per_line, 3))
        input_mask = torch.ones((self.max_trips, self.max_lines), dtype=torch.bool)

        for trip_idx, order in enumerate(vec_orders.values()):
            if trip_idx >= self.max_trips:
                break
            for line_idx, line in enumerate(order["lines"]):
                if line_idx >= self.max_lines:
                    break
                points = line["points"]
                normalized_points = self._normalize_coordinates(points, bounds["x_min"], bounds["x_max"],
                                                                bounds["y_min"], bounds["y_max"])
                resampled_points = self._resample_polyline(normalized_points, self.points_per_line)
                input_tensor[trip_idx, line_idx, :, :2] = torch.tensor([[p[0], p[1]] for p in resampled_points])
                input_tensor[trip_idx, line_idx, :, 2] = line["class"]
                input_mask[trip_idx, line_idx] = False

        for line_idx, line in enumerate(label_order_lines):
            points = line["points"]
            normalized_points = self._normalize_coordinates(points, bounds["x_min"], bounds["x_max"],
                                                            bounds["y_min"], bounds["y_max"])
            resampled_points = self._resample_polyline(normalized_points, self.points_per_line)
            label_tensor[line_idx, :, :2] = torch.tensor([[p[0], p[1]] for p in resampled_points])
            label_tensor[line_idx, :, 2] = line["class"]

        return {
            "input_tensor": input_tensor,
            "label_tensor": label_tensor,
            "input_mask": input_mask,
            "bounds": bounds
        }

    def _load_data_from_folders(self):
        """
        Load all maps from vec and label folders.
        """
        maps = []
        vec_files = os.listdir(self.vec_folder)
        for vec_file in vec_files:
            vec_path_all = [os.path.join(self.vec_folder, vec_file, f"{i}.txt") for i in range(self.max_trips)]
            label_path = os.path.join(self.label_folder, f"{vec_file}.txt")
            map_data = self._parse_files(vec_path_all, label_path)
            maps.append(map_data)
        return maps

    def _parse_files(self, vec_path_all, label_path):
        """
        Parse .txt files and build structured map data.
        """
        type_to_class = {
            "lane_marking": 0,
            "stop_location": 1,
            "cross_walk": 2
        }

        vec_orders = defaultdict(lambda: {"order_id": None, "lines": []})
        all_points = []

        for vec_file in vec_path_all:
            order_id = int(os.path.basename(vec_file).split(".")[0])
            df = pd.read_csv(vec_file, sep='\t')

            for _, row in df.iterrows():
                geom = row["geom"].strip()
                geom_type = row["type"].strip()
                class_id = type_to_class.get(geom_type, -1)
                coord_strs = re.findall(r"\(\(?(.+?)\)?\)", geom)

                for coord_str in coord_strs:
                    point_list = [(float(x), float(y)) for x, y in (coord.split() for coord in coord_str.split(", "))]
                    vec_orders[order_id]["order_id"] = order_id
                    vec_orders[order_id]["lines"].append({"points": point_list, "class": class_id})
                    all_points.extend(point_list)

        df = pd.read_csv(label_path, sep='\t')
        label_order_lines = []

        for _, row in df.iterrows():
            geom = row["geom"].strip()
            geom_type = row["type"].strip()
            class_id = type_to_class.get(geom_type, -1)
            coord_strs = re.findall(r"\(\(?(.+?)\)?\)", geom)

            for coord_str in coord_strs:
                point_list = [(float(x), float(y)) for x, y in (coord.split() for coord in coord_str.split(", "))]
                label_order_lines.append({"points": point_list, "class": class_id})
                all_points.extend(point_list)

        bounds = {
            'x_min': min(x for x, _ in all_points),
            'y_min': min(y for _, y in all_points),
            'x_max': max(x for x, _ in all_points),
            'y_max': max(y for _, y in all_points)
        }
        return {
            "vec_orders": vec_orders,
            "label_order_lines": label_order_lines,
            "bounds": bounds
        }

    def _normalize_coordinates(self, points, x_min, x_max, y_min, y_max):
        """
        Normalize (x, y) coordinates to [0, 1] range.
        """
        normalized_points = []
        for point in points:
            x = (point[0] - x_min) / (x_max - x_min)
            y = (point[1] - y_min) / (y_max - y_min)
            normalized_points.append((x, y))
        return normalized_points

    def _compute_length(self, point1, point2):
        """
        Compute Euclidean distance between two points.
        """
        return np.hypot(point2[0] - point1[0], point2[1] - point1[1])

    def _calculate_total_length(self, polylines):
        """
        Calculate total length of a polyline.
        """
        return sum(self._compute_length(polylines[i], polylines[i+1]) for i in range(len(polylines) - 1))

    def _resample_polyline(self, polylines, num_points):
        """
        Resample a polyline to a fixed number of points.
        """
        if len(polylines) < 2:
            raise ValueError("Polyline must contain at least two points.")

        is_closed = polylines[0] == polylines[-1]
        total_length = self._calculate_total_length(polylines)
        if total_length == 0:
            return [polylines[0]] * num_points

        segment_length = total_length / num_points if is_closed else total_length / (num_points - 1)
        sampled_points = [polylines[0]]
        distance_covered = 0.0
        next_sample_at = segment_length
        i = 0

        while len(sampled_points) < num_points:
            if i >= len(polylines) - 1:
                break

            a = polylines[i]
            b = polylines[i + 1]
            ab_length = self._compute_length(a, b)

            if ab_length == 0:
                i += 1
                continue

            while distance_covered + ab_length >= next_sample_at:
                t = (next_sample_at - distance_covered) / ab_length
                x = a[0] + (b[0] - a[0]) * t
                y = a[1] + (b[1] - a[1]) * t
                sampled_points.append((x, y))
                next_sample_at += segment_length

            distance_covered += ab_length
            i += 1

        while len(sampled_points) < num_points:
            sampled_points.append(polylines[-1])

        return sampled_points[:num_points]

def collate_fn(batch):
    """
    Custom collate function to handle batches with variable-length labels.
    """
    input_tensors = [item["input_tensor"] for item in batch]
    label_tensors = [item["label_tensor"] for item in batch]
    input_masks = [item["input_mask"] for item in batch]
    bounds = [item["bounds"] for item in batch]

    label_lengths = [len(t) for t in label_tensors]
    max_label_length = max(label_lengths)

    padded_label_tensors = []
    label_masks = []

    for label_tensor, length in zip(label_tensors, label_lengths):
        padded_label_tensor = torch.zeros((max_label_length, label_tensor.shape[1], label_tensor.shape[2]))
        padded_label_tensor[:length] = label_tensor
        padded_label_tensors.append(padded_label_tensor)

        label_mask = torch.zeros(max_label_length, dtype=torch.bool)
        label_mask[:length] = False
        label_masks.append(label_mask)

    input_tensors = torch.stack(input_tensors, dim=0)
    input_masks = torch.stack(input_masks, dim=0)
    label_masks = torch.stack(label_masks, dim=0)
    padded_label_tensors = torch.stack(padded_label_tensors, dim=0)

    return {
        "input_tensor": input_tensors,
        "label_tensor": padded_label_tensors,
        "input_mask": input_masks,
        "label_mask": label_masks,
        "bounds": bounds
    }
