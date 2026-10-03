"""PointNet++ classification network (single-scale grouping variant).

Three Set Abstraction levels progressively downsample the point cloud
(1024 -> 512 -> 128 -> 1 "global" point) while growing the feature
dimension, then a small MLP head produces per-class logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointnet2_utils import PointNetSetAbstraction


class PointNet2Classifier(nn.Module):
    def __init__(self, num_classes=40, normal_channel=False):
        super().__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel + 3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        """xyz: [B, N, 3] (or [B, N, 6] if normal_channel) -> logits [B, num_classes]."""
        if self.normal_channel:
            points = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            points = None

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(l3_points.shape[0], -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        logits = self.fc3(x)
        return logits


def get_loss():
    return nn.CrossEntropyLoss()
