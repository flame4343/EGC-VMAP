import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between predicted polylines and ground-truth polylines.
    It uses a cost combining classification, polyline similarity, and direction alignment.
    """
    def __init__(self, cost_class=1, cost_polyline=1, cost_direction=1):
        """
        Args:
            cost_class (float): Weight of the classification loss in matching cost.
            cost_polyline (float): Weight of the polyline distance loss.
            cost_direction (float): Weight of the direction loss.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_polyline = cost_polyline
        self.cost_direction = cost_direction

    def polyline_distance(self, polyline1, polyline2):
        """
        Compute the average minimum distance between two polylines.
        Args:
            polyline1 (Tensor): Shape [N, 2]
            polyline2 (Tensor): Shape [M, 2]
        Returns:
            Average bidirectional distance.
        """
        if polyline1.dim() == 1:
            polyline1 = polyline1.unsqueeze(0)
        if polyline2.dim() == 1:
            polyline2 = polyline2.unsqueeze(0)

        dist_matrix = torch.cdist(polyline1, polyline2, p=1)
        min_dist1 = dist_matrix.min(dim=1).values.mean()
        min_dist2 = dist_matrix.min(dim=0).values.mean()

        return (min_dist1 + min_dist2) / 2

    def direction_loss(self, pred_polyline, gt_polyline):
        """
        Compute cosine distance between the start-end vectors of two polylines.
        Args:
            pred_polyline (Tensor): Predicted polyline [N, 2]
            gt_polyline (Tensor): Ground-truth polyline [N, 2]
        Returns:
            Direction loss (1 - cosine similarity).
        """
        pred_direction = pred_polyline[-1] - pred_polyline[0]
        gt_direction = gt_polyline[-1] - gt_polyline[0]

        pred_direction = pred_direction / (torch.norm(pred_direction) + 1e-6)
        gt_direction = gt_direction / (torch.norm(gt_direction) + 1e-6)

        cosine_similarity = torch.dot(pred_direction, gt_direction)
        return 1 - cosine_similarity

    def forward(self, outputs, targets):
        """
        Perform the matching between outputs and targets.
        Args:
            outputs (dict): Dictionary containing 'pred_logits' and 'pred_polylines'.
            targets (list of dict): Each dict contains 'labels' and 'polylines'.
        Returns:
            List of matched indices for each batch element.
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
            out_polylines = outputs["pred_polylines"].flatten(0, 1)

            tgt_ids = torch.cat([t["labels"] for t in targets])
            tgt_polylines = torch.cat([t["polylines"] for t in targets], dim=0)

            # Classification cost
            cost_class = -out_prob[:, tgt_ids]

            # Polyline distance cost
            cost_polyline = torch.zeros_like(cost_class)
            for i, pred_poly in enumerate(out_polylines):
                for j, tgt_poly in enumerate(tgt_polylines):
                    cost_polyline[i, j] = self.polyline_distance(pred_poly, tgt_poly)

            # Direction cost
            cost_direction = torch.zeros_like(cost_class)
            for i, pred_poly in enumerate(out_polylines):
                for j, tgt_poly in enumerate(tgt_polylines):
                    cost_direction[i, j] = self.direction_loss(pred_poly, tgt_poly)

            # Combined cost
            C = (self.cost_class * cost_class +
                 self.cost_polyline * cost_polyline +
                 self.cost_direction * cost_direction)
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(t["labels"]) for t in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
