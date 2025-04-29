import math
import torch

def normalize_coordinates(points, x_min, x_max, y_min, y_max):
    """
    Normalize coordinates to the [0,1] range.
    Args:
        points (list of tuple): List of (x, y) points.
        x_min, x_max, y_min, y_max (float): Bounding box values.
    Returns:
        List of normalized (x, y) points.
    """
    normalized_points = []
    for point in points:
        x = (point[0] - x_min) / (x_max - x_min)
        y = (point[1] - y_min) / (y_max - y_min)
        normalized_points.append((x, y))
    return normalized_points

def compute_length(point1, point2):
    """
    Compute Euclidean distance between two points.
    Args:
        point1, point2 (tuple): (x, y) coordinates.
    Returns:
        Distance (float).
    """
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def calculate_total_length(polylines):
    """
    Compute the total length of a polyline.
    Args:
        polylines (list of tuple): List of (x, y) points.
    Returns:
        Total length (float).
    """
    total_length = 0.0
    for i in range(len(polylines) - 1):
        total_length += compute_length(polylines[i], polylines[i + 1])
    return total_length

def resample_polyline(polylines, num_points):
    """
    Resample a polyline to a fixed number of points.
    Args:
        polylines (list of tuple): List of (x, y) points.
        num_points (int): Target number of points after resampling.
    Returns:
        List of resampled (x, y) points.
    """
    if len(polylines) < 2:
        raise ValueError("Polyline must contain at least two points.")

    is_closed = polylines[0] == polylines[-1]

    total_length = calculate_total_length(polylines)
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
        ab_length = compute_length(a, b)

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
