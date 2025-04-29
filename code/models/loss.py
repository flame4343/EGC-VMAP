import torch
import torch.nn as nn
import torch.nn.functional as F

class SetCriterion(nn.Module):
    """
    Compute losses for the Transformer output.
    Includes classification loss, polyline matching loss, and direction loss.
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        """
        Args:
            num_classes (int): Number of object classes.
            matcher (nn.Module): Matching algorithm (e.g., Hungarian Matcher).
            weight_dict (dict): Weight for each loss.
            eos_coef (float): End-of-sequence token coefficient (not used here).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

    def loss_labels(self, outputs, targets, indices, num_polylines):
        """
        Classification loss (Cross Entropy).
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        return {'loss_ce': loss_ce}

    def loss_polylines(self, outputs, targets, indices, num_polylines):
        """
        Polyline regression loss (average L1 distance between predicted and target polylines).
        """
        idx = self._get_src_permutation_idx(indices)
        src_polylines = outputs['pred_polylines'][idx]
        target_polylines = torch.cat([t['polylines'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        polyline_loss = 0.0
        for polyline1, polyline2 in zip(src_polylines, target_polylines):
            dist_matrix = torch.cdist(polyline1.unsqueeze(0), polyline2.unsqueeze(0), p=1)
            min_dist1 = dist_matrix.min(dim=1).values.mean()
            min_dist2 = dist_matrix.min(dim=2).values.mean()
            polyline_loss += (min_dist1 + min_dist2) / 2
        polyline_loss /= num_polylines

        return {'loss_polyline': polyline_loss}

    def loss_direction(self, outputs, targets, indices, num_polylines):
        """
        Direction loss (cosine distance between polyline start-end vectors).
        """
        idx = self._get_src_permutation_idx(indices)
        src_polylines = outputs['pred_polylines'][idx]
        target_polylines = torch.cat([t['polylines'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        direction_loss = 0.0
        for src, tgt in zip(src_polylines, target_polylines):
            src_diff = src[-1] - src[0]
            tgt_diff = tgt[-1] - tgt[0]

            src_diff = src_diff / (torch.norm(src_diff) + 1e-6)
            tgt_diff = tgt_diff / (torch.norm(tgt_diff) + 1e-6)

            cosine_similarity = torch.dot(src_diff, tgt_diff)
            direction_loss += 1 - cosine_similarity
        direction_loss /= num_polylines

        return {'loss_direction': direction_loss}

    def _get_src_permutation_idx(self, indices):
        """
        Permutation indices for gathering matching predictions.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """
        Compute all losses and combine them.
        """
        indices = self.matcher(outputs, targets)
        num_polylines = sum(len(t["labels"]) for t in targets)
        num_polylines = torch.clamp(torch.as_tensor(num_polylines, dtype=torch.float, device=next(iter(outputs.values())).device), min=1).item()

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_polylines))
        losses.update(self.loss_polylines(outputs, targets, indices, num_polylines))
        losses.update(self.loss_direction(outputs, targets, indices, num_polylines))
        return losses
