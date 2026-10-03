"""Full PointPillars-style detector: pillar encoding -> BEV scatter -> 2D
backbone -> per-cell detection head, plus target assignment and loss.

This is a from-scratch, simplified reimplementation (single-scale, per-cell
anchor-free-ish head) rather than the full MMDetection3D PointPillars — it
keeps training runnable on CPU/GPU without sparse-conv dependencies while
preserving the same core idea: pillarize, project to BEV, detect with a 2D
CNN. It supports fine-tuning on the base KITTI classes and adapting to a
new class by simply growing `class_names` / `anchor_sizes` and continuing
training (see train_detector.py).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kitti_detection.models.backbone2d import Backbone2D, DetectionHead
from kitti_detection.models.pillar_encoder import PillarFeatureNet, PointPillarsScatter, voxelize_to_pillars


class PointPillarsDetector(nn.Module):
    def __init__(self, class_names, x_range=(-40, 40), y_range=(-40, 40), voxel_size=0.32,
                 pillar_feat_channels=64):
        super().__init__()
        self.class_names = class_names
        self.x_range = x_range
        self.y_range = y_range
        self.voxel_size = voxel_size
        self.grid_w = int((x_range[1] - x_range[0]) / voxel_size)
        self.grid_h = int((y_range[1] - y_range[0]) / voxel_size)

        self.pillar_feature_net = PillarFeatureNet(num_point_features=9, feat_channels=pillar_feat_channels)
        self.scatter = PointPillarsScatter(self.grid_h, self.grid_w)
        self.backbone = Backbone2D(in_channels=pillar_feat_channels)
        self.head = DetectionHead(in_channels=384, num_classes=len(class_names))
        self.out_stride = 2  # backbone output is at half the pillar-grid resolution

    def pillarize_batch(self, point_clouds):
        """point_clouds: list of [N_i, 4] numpy arrays (one raw scan per batch item).
        Returns tensors ready for the pillar feature net + scatter."""
        all_points, all_mask, all_coords = [], [], []
        for b, pts in enumerate(point_clouds):
            pillar_points, pillar_mask, coords = voxelize_to_pillars(
                pts, self.x_range, self.y_range, self.voxel_size)
            batch_col = np.full((coords.shape[0], 1), b, dtype="int64")
            all_points.append(pillar_points)
            all_mask.append(pillar_mask)
            all_coords.append(np.concatenate([batch_col, coords], axis=1))
        device = next(self.parameters()).device
        pillar_points = torch.from_numpy(np.concatenate(all_points, axis=0)).to(device)
        pillar_mask = torch.from_numpy(np.concatenate(all_mask, axis=0)).to(device)
        coords = torch.from_numpy(np.concatenate(all_coords, axis=0)).to(device)
        return pillar_points, pillar_mask, coords

    def forward(self, point_clouds):
        """point_clouds: list of [N_i, 4] numpy point clouds, len == batch size.
        Returns cls_logits [B, num_classes, H', W'], box_preds [B, num_classes, 7, H', W'].
        """
        batch_size = len(point_clouds)
        pillar_points, pillar_mask, coords = self.pillarize_batch(point_clouds)
        pillar_features = self.pillar_feature_net(pillar_points, pillar_mask)
        bev = self.scatter(pillar_features, coords, batch_size)
        feat = self.backbone(bev)
        cls_logits, box_preds = self.head(feat)
        return cls_logits, box_preds

    def grid_cell_to_xy(self, row, col, feature_h, feature_w):
        """Maps a (row, col) cell in the backbone's output grid back to a
        world (x, y) coordinate at that cell's center."""
        stride_x = (self.x_range[1] - self.x_range[0]) / feature_w
        stride_y = (self.y_range[1] - self.y_range[0]) / feature_h
        x = self.x_range[0] + (col + 0.5) * stride_x
        y = self.y_range[0] + (row + 0.5) * stride_y
        return x, y


def build_targets(detector, gt_boxes_batch, feature_h, feature_w, pos_radius=1):
    """Assigns each GT box to its nearest output-grid cell (and `pos_radius`
    neighbors) as a positive sample. gt_boxes_batch: list (len=batch) of list
    of box dicts {"type", "centroid":[x,y,z], "dimensions":{l,w,h}, "yaw"}.

    Returns cls_targets [B,C,H,W] (1/0), box_targets [B,C,7,H,W], pos_mask [B,C,H,W].
    """
    B = len(gt_boxes_batch)
    C = len(detector.class_names)
    cls_targets = torch.zeros(B, C, feature_h, feature_w)
    box_targets = torch.zeros(B, C, 7, feature_h, feature_w)
    pos_mask = torch.zeros(B, C, feature_h, feature_w, dtype=torch.bool)

    stride_x = (detector.x_range[1] - detector.x_range[0]) / feature_w
    stride_y = (detector.y_range[1] - detector.y_range[0]) / feature_h

    for b, boxes in enumerate(gt_boxes_batch):
        for box in boxes:
            if box["type"] not in detector.class_names:
                continue
            c = detector.class_names.index(box["type"])
            cx, cy, cz = box["centroid"]
            col = int((cx - detector.x_range[0]) / stride_x)
            row = int((cy - detector.y_range[0]) / stride_y)
            if not (0 <= row < feature_h and 0 <= col < feature_w):
                continue
            for dr in range(-pos_radius, pos_radius + 1):
                for dc in range(-pos_radius, pos_radius + 1):
                    r, c2 = row + dr, col + dc
                    if not (0 <= r < feature_h and 0 <= c2 < feature_w):
                        continue
                    cls_targets[b, c, r, c2] = 1.0
                    pos_mask[b, c, r, c2] = True
                    center_x, center_y = detector.grid_cell_to_xy(r, c2, feature_h, feature_w)
                    dims = box["dimensions"]
                    box_targets[b, c, :, r, c2] = torch.tensor([
                        cx - center_x, cy - center_y, cz,
                        np.log(max(dims["length"], 1e-3)),
                        np.log(max(dims["width"], 1e-3)),
                        np.log(max(dims["height"], 1e-3)),
                        box["yaw"],
                    ])
    return cls_targets, box_targets, pos_mask


def compute_loss(cls_logits, box_preds, cls_targets, box_targets, pos_mask, pos_weight=15.0):
    """Classification loss normalized by the number of positive cells (not
    the full grid) — with thousands of empty BEV cells per handful of
    objects, a plain mean-reduced BCE lets the model minimize loss by
    predicting background everywhere. Weighting positives up and
    normalizing by their count keeps the rare foreground cells from being
    drowned out. Box regression (smooth-L1) is computed only at positive
    cells, as usual."""
    per_cell_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, reduction="none")
    weights = torch.where(cls_targets > 0, pos_weight, 1.0)
    num_pos = cls_targets.sum().clamp(min=1.0)
    cls_loss = (per_cell_loss * weights).sum() / (num_pos * pos_weight + cls_targets.numel())

    pos_mask_expanded = pos_mask.unsqueeze(2).expand_as(box_preds)
    if pos_mask_expanded.any():
        box_loss = F.smooth_l1_loss(box_preds[pos_mask_expanded], box_targets[pos_mask_expanded])
    else:
        box_loss = torch.tensor(0.0, device=cls_logits.device)

    total = cls_loss + 2.0 * box_loss
    return total, {"cls_loss": cls_loss.item(), "box_loss": box_loss.item()}


def _nms_3d(detections, iou_threshold=0.3):
    """Greedy non-max suppression using the same 3D IoU the eval pipeline
    uses, per class — collapses the cluster of nearby positive cells a
    single object triggers (from the `pos_radius` neighborhood in
    build_targets) down to one box."""
    from kitti_detection.eval.box_iou import iou_3d

    kept = []
    by_class = {}
    for det in detections:
        by_class.setdefault(det["type"], []).append(det)

    for boxes in by_class.values():
        boxes = sorted(boxes, key=lambda b: -b["score"])
        while boxes:
            best = boxes.pop(0)
            kept.append(best)
            boxes = [b for b in boxes if iou_3d(best, b) < iou_threshold]
    return kept


def decode_predictions(detector, cls_logits, box_preds, score_threshold=0.5, max_detections=50,
                        nms_iou_threshold=0.3):
    """Decodes raw head outputs for one sample (no batch dim) into a list of
    box dicts: thresholds on sigmoid(cls score), then applies per-class NMS."""
    C, H, W = cls_logits.shape
    scores = torch.sigmoid(cls_logits)
    detections = []
    topk = torch.topk(scores.view(-1), min(max_detections, scores.numel()))
    for score, flat_idx in zip(topk.values, topk.indices):
        if score.item() < score_threshold:
            continue
        c = (flat_idx // (H * W)).item()
        r = ((flat_idx % (H * W)) // W).item()
        col = (flat_idx % W).item()
        dx, dy, dz, log_l, log_w, log_h, yaw = box_preds[c, :, r, col].tolist()
        center_x, center_y = detector.grid_cell_to_xy(r, col, H, W)
        detections.append({
            "type": detector.class_names[c],
            "score": score.item(),
            "centroid": [center_x + dx, center_y + dy, dz],
            "dimensions": {"length": float(np.exp(log_l)), "width": float(np.exp(log_w)),
                            "height": float(np.exp(log_h))},
            "yaw": yaw,
        })
    return _nms_3d(detections, iou_threshold=nms_iou_threshold)
