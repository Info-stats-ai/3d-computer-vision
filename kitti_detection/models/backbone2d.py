"""PointPillars' 2D backbone + single-stage detection head, applied to the
BEV pseudo-image produced by PointPillarsScatter.

The backbone downsamples at 3 scales (like the original paper's 2x/2x/2x
blocks), then upsamples and concatenates into one feature map; the head is
a plain anchor-free-ish per-cell predictor: one (class, box, direction)
prediction per BEV grid cell per class, which keeps the implementation
simple while covering the same input/output contract a real anchor head
would.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, stride=1, num_layers=1):
    layers = [nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
              nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    for _ in range(num_layers - 1):
        layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class Backbone2D(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.block1 = conv_block(in_channels, 64, stride=2, num_layers=3)
        self.block2 = conv_block(64, 128, stride=2, num_layers=4)
        self.block3 = conv_block(128, 256, stride=2, num_layers=5)

        self.up1 = nn.ConvTranspose2d(64, 128, 1)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 4, stride=4)

    def forward(self, x):
        # Pad to a multiple of 8 so the three stride-2 downsamples divide
        # evenly and the upsampled feature maps line up exactly for concat.
        H, W = x.shape[-2:]
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        c1 = self.block1(x)   # stride 2
        c2 = self.block2(c1)  # stride 4
        c3 = self.block3(c2)  # stride 8

        u1 = self.up1(c1)
        u2 = self.up2(c2)
        u3 = self.up3(c3)
        out = torch.cat([u1, u2, u3], dim=1)  # [B, 384, H/2, W/2]

        target_h, target_w = (H + pad_h) // 2, (W + pad_w) // 2
        return out[:, :, :target_h, :target_w]


class DetectionHead(nn.Module):
    """Per-cell predictor: for each class, a binary objectness score, a 7-dim
    box regression (dx, dy, dz, log(l), log(w), log(h), yaw-sin/cos handled
    as 2 channels), giving 7 box channels total per class."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = nn.Conv2d(in_channels, num_classes, 1)
        self.box_head = nn.Conv2d(in_channels, num_classes * 7, 1)

    def forward(self, x):
        cls_logits = self.cls_head(x)  # [B, num_classes, H, W]
        box_preds = self.box_head(x)   # [B, num_classes*7, H, W]
        B, _, H, W = box_preds.shape
        box_preds = box_preds.view(B, self.num_classes, 7, H, W)
        return cls_logits, box_preds
