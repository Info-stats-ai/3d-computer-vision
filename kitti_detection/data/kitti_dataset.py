"""KITTI detection dataset wrapper, plus a synthetic point-cloud+box generator
used to smoke-test the full training/eval/viz pipeline without requiring the
(manually downloaded, registration-gated) real KITTI files.
"""
import glob
import os

import numpy as np
from torch.utils.data import Dataset

from kitti_detection.data.kitti_io import load_labels, load_velodyne_bin


class KittiDetectionDataset(Dataset):
    """Loads real KITTI velodyne .bin + label_2 .txt pairs.

    Note: KITTI labels are in *camera* coordinates while velodyne points are
    in *lidar* coordinates; a full pipeline would transform one into the
    other with the calib Tr_velo_to_cam matrix (see kitti_io.Calibration).
    This dataset keeps both in their native frames and leaves the transform
    to the caller, since PointPillars' own convention varies by codebase.
    """

    def __init__(self, root, split="training"):
        self.velodyne_dir = os.path.join(root, split, "velodyne")
        self.label_dir = os.path.join(root, split, "label_2")
        self.frame_ids = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(self.velodyne_dir, "*.bin"))
        )

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        points = load_velodyne_bin(os.path.join(self.velodyne_dir, f"{frame_id}.bin"))
        label_path = os.path.join(self.label_dir, f"{frame_id}.txt")
        objects = load_labels(label_path) if os.path.exists(label_path) else []
        boxes = [{"type": o.type, "centroid": list(o.loc),
                  "dimensions": {"length": o.l, "width": o.w, "height": o.h},
                  "yaw": o.yaw} for o in objects]
        return frame_id, points, boxes


class SyntheticKittiDataset(Dataset):
    """Generates synthetic LiDAR-like scenes with a known number of boxed
    "objects" (clusters of points inside an oriented box) plus background
    noise, with ground-truth boxes recorded exactly — used to verify the
    PointPillars pipeline (pillarization -> training -> decode -> eval ->
    viz) end-to-end without the real dataset.
    """

    def __init__(self, num_samples=200, class_names=("Car", "Pedestrian", "Cyclist"),
                 x_range=(-40, 40), y_range=(-40, 40), seed=0):
        self.num_samples = num_samples
        self.class_names = list(class_names)
        self.x_range = x_range
        self.y_range = y_range
        self.rng = np.random.default_rng(seed)
        self.class_dims = {
            "Car": (3.9, 1.6, 1.56), "Pedestrian": (0.8, 0.6, 1.73),
            "Cyclist": (1.76, 0.6, 1.73), "Pole": (0.3, 0.3, 3.0), "TrafficCone": (0.4, 0.4, 0.7),
        }

    def __len__(self):
        return self.num_samples

    def _sample_box_points(self, centroid, dims, yaw, num_points=60):
        l, w, h = dims
        local = self.rng.uniform([-l / 2, -w / 2, -h], [l / 2, w / 2, 0], size=(num_points, 3))
        cos, sin = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos, -sin], [sin, cos]])
        local[:, :2] = local[:, :2] @ rot.T
        local += np.array([centroid[0], centroid[1], centroid[2]])
        reflectance = self.rng.uniform(0, 1, size=(num_points, 1)).astype("float32")
        return np.concatenate([local.astype("float32"), reflectance], axis=1)

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        num_objects = rng.integers(2, 6)
        boxes = []
        point_chunks = []
        for _ in range(num_objects):
            cls = rng.choice(self.class_names)
            l, w, h = self.class_dims[cls]
            cx = rng.uniform(self.x_range[0] + 5, self.x_range[1] - 5)
            cy = rng.uniform(self.y_range[0] + 5, self.y_range[1] - 5)
            cz = -1.5
            yaw = rng.uniform(-np.pi, np.pi)
            boxes.append({"type": cls, "centroid": [cx, cy, cz],
                           "dimensions": {"length": l, "width": w, "height": h}, "yaw": yaw})
            self.rng = rng
            point_chunks.append(self._sample_box_points([cx, cy, cz], (l, w, h), yaw))

        background = rng.uniform(
            [self.x_range[0], self.y_range[0], -2, 0],
            [self.x_range[1], self.y_range[1], 1, 1],
            size=(800, 4)).astype("float32")
        points = np.concatenate(point_chunks + [background], axis=0)
        return f"synthetic_{idx}", points, boxes
