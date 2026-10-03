"""Core PointNet++ operations: farthest point sampling, ball query, grouping,
and the Set Abstraction layer. Implemented in pure PyTorch (no custom CUDA
kernels) so the model runs on both CPU and GPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """Pairwise squared Euclidean distance. src: [B,N,C], dst: [B,M,C] -> [B,N,M]."""
    return torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2, dim=-1)


def index_points(points, idx):
    """Gather points at given indices. points: [B,N,C], idx: [B,S] or [B,S,K] -> [B,S,C] or [B,S,K,C]."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """Iteratively picks the point farthest from the already-chosen set, giving
    a spatially-spread-out subsample. xyz: [B,N,3] -> indices [B,npoint]."""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def ball_query(radius, nsample, xyz, new_xyz):
    """For each query point in new_xyz, find up to nsample neighbors within
    `radius` of it in xyz. Falls back to repeating the nearest point when a
    query ball contains fewer than nsample points. Returns index tensor
    [B,S,nsample]."""
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstraction(nn.Module):
    """One PointNet++ Set Abstraction level:
    1. Sample `npoint` centroids via farthest point sampling.
    2. Group each centroid's local neighborhood via ball query.
    3. Run a shared MLP over each local group and max-pool -> local feature.
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """xyz: [B,N,3] coordinates. points: [B,N,D] features or None.
        Returns new_xyz [B,S,3], new_points [B,S,mlp[-1]]."""
        if self.group_all:
            new_xyz, new_points = self._sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self._sample_and_group(xyz, points)

        # new_points: [B, S, nsample, C] -> [B, C, nsample, S] for Conv2d
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, dim=2)[0]  # max-pool over neighbors
        new_points = new_points.permute(0, 2, 1)  # [B, S, C]
        return new_xyz, new_points

    def _sample_and_group(self, xyz, points):
        B, N, C = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm
        return new_xyz, new_points

    def _sample_and_group_all(self, xyz, points):
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C, device=xyz.device)
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points
