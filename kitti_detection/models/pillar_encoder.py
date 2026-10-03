"""PointPillars-style pillar feature encoding: voxelizes the point cloud into
vertical columns ("pillars") on a bird's-eye-view (BEV) grid, then runs a
per-pillar PointNet (shared MLP + max-pool) to get one feature vector per
pillar. Scattering those vectors back to BEV grid locations gives a
pseudo-image a standard 2D CNN backbone can consume — this is what lets
PointPillars reuse 2D detection heads instead of 3D sparse convolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PillarFeatureNet(nn.Module):
    def __init__(self, num_point_features=9, feat_channels=64):
        """num_point_features: per-point input dims (x,y,z,reflectance + the
        5 PointPillars augmentation channels: offsets to pillar mean and center)."""
        super().__init__()
        self.linear = nn.Linear(num_point_features, feat_channels, bias=False)
        self.bn = nn.BatchNorm1d(feat_channels)

    def forward(self, pillar_points, pillar_mask):
        """pillar_points: [P, N, C] (P pillars, N points per pillar, C features).
        pillar_mask: [P, N] bool, True where the point is real (not padding).
        Returns [P, feat_channels] — one feature vector per pillar."""
        P, N, C = pillar_points.shape
        x = self.linear(pillar_points.view(P * N, C))
        x = self.bn(x).view(P, N, -1)
        x = F.relu(x)
        x = x.masked_fill(~pillar_mask.unsqueeze(-1), float("-inf"))
        pooled = torch.max(x, dim=1)[0]
        pooled = torch.nan_to_num(pooled, neginf=0.0)  # pillars with zero real points
        return pooled


class PointPillarsScatter(nn.Module):
    """Scatters per-pillar features back to their BEV grid coordinates,
    producing a pseudo-image [B, C, H, W] for a 2D CNN backbone."""

    def __init__(self, grid_h, grid_w):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

    def forward(self, pillar_features, coords, batch_size):
        """pillar_features: [P, C]. coords: [P, 3] (batch_idx, row, col)."""
        C = pillar_features.shape[1]
        canvas = torch.zeros(batch_size, C, self.grid_h, self.grid_w,
                              device=pillar_features.device, dtype=pillar_features.dtype)
        batch_idx = coords[:, 0].long()
        row = coords[:, 1].long().clamp(0, self.grid_h - 1)
        col = coords[:, 2].long().clamp(0, self.grid_w - 1)
        canvas[batch_idx, :, row, col] = pillar_features
        return canvas


def voxelize_to_pillars(points, x_range, y_range, voxel_size, max_points_per_pillar=32, max_pillars=12000):
    """CPU-side voxelization of one point cloud [N,4] (x,y,z,reflectance) into
    pillars on the (x,y) BEV grid. Returns:
      pillar_points: [P, max_points_per_pillar, 9] (padded)
      pillar_mask:   [P, max_points_per_pillar] bool
      coords:        [P, 2] (row, col) grid location of each pillar
    Augments each point with: offset to its pillar's point-mean (3 dims) and
    offset to its pillar's geometric center (2 dims), per the PointPillars paper.
    """
    x, y = points[:, 0], points[:, 1]
    in_range = (x >= x_range[0]) & (x < x_range[1]) & (y >= y_range[0]) & (y < y_range[1])
    points = points[in_range]
    x, y = points[:, 0], points[:, 1]

    col = ((x - x_range[0]) / voxel_size).astype("int64")
    row = ((y - y_range[0]) / voxel_size).astype("int64")

    import numpy as np
    pillar_keys = row * 100000 + col  # unique key per (row, col)
    unique_keys, inverse = np.unique(pillar_keys, return_inverse=True)
    num_pillars = min(len(unique_keys), max_pillars)

    pillar_points = np.zeros((num_pillars, max_points_per_pillar, 9), dtype="float32")
    pillar_mask = np.zeros((num_pillars, max_points_per_pillar), dtype=bool)
    coords = np.zeros((num_pillars, 2), dtype="int64")

    order = np.argsort(inverse)
    sorted_inverse = inverse[order]
    boundaries = np.searchsorted(sorted_inverse, np.arange(len(unique_keys)))
    boundaries = list(boundaries) + [len(sorted_inverse)]

    for p in range(num_pillars):
        idx_in_pillar = order[boundaries[p]:boundaries[p + 1]]
        n = min(len(idx_in_pillar), max_points_per_pillar)
        idx_in_pillar = idx_in_pillar[:n]
        pts = points[idx_in_pillar]  # [n, 4]
        mean_xyz = pts[:, :3].mean(axis=0)
        key = unique_keys[p]
        r, c = divmod(key, 100000)
        center_x = x_range[0] + (c + 0.5) * voxel_size
        center_y = y_range[0] + (r + 0.5) * voxel_size

        feat = np.zeros((n, 9), dtype="float32")
        feat[:, :4] = pts
        feat[:, 4:7] = pts[:, :3] - mean_xyz
        feat[:, 7] = pts[:, 0] - center_x
        feat[:, 8] = pts[:, 1] - center_y

        pillar_points[p, :n] = feat
        pillar_mask[p, :n] = True
        coords[p] = [r, c]

    return pillar_points, pillar_mask, coords
