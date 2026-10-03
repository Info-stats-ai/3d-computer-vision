"""I/O helpers for raw KITTI LiDAR scans and label files."""
import numpy as np


def load_velodyne_bin(bin_path):
    """Load a raw KITTI Velodyne scan. Returns [N, 4]: x, y, z, reflectance."""
    scan = np.fromfile(bin_path, dtype=np.float32)
    return scan.reshape(-1, 4)


class KittiObject:
    """One parsed line of a KITTI label_2 file (one annotated 3D object)."""

    def __init__(self, line):
        parts = line.strip().split(" ")
        self.type = parts[0]
        self.truncated = float(parts[1])
        self.occluded = int(parts[2])
        self.alpha = float(parts[3])
        self.bbox_2d = tuple(float(x) for x in parts[4:8])
        # dimensions in camera coords: height, width, length
        self.h, self.w, self.l = (float(x) for x in parts[8:11])
        # location in camera coords: x, y, z (y is "down" in camera frame)
        self.loc = tuple(float(x) for x in parts[11:14])
        self.yaw = float(parts[14])  # rotation_y

    def __repr__(self):
        return f"KittiObject(type={self.type}, loc={self.loc}, dims=({self.l},{self.w},{self.h}), yaw={self.yaw})"


def load_labels(label_path):
    """Load a KITTI label_2 .txt file. Returns list[KittiObject], excluding 'DontCare'."""
    objects = []
    with open(label_path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = KittiObject(line)
            if obj.type != "DontCare":
                objects.append(obj)
    return objects


class Calibration:
    """Parses a KITTI calib .txt file; exposes Tr_velo_to_cam for coordinate conversion."""

    def __init__(self, calib_path):
        self.matrices = {}
        with open(calib_path) as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                nums = np.array([float(x) for x in value.strip().split(" ")])
                self.matrices[key.strip()] = nums

    def velo_to_cam_matrix(self):
        """Returns the 3x4 velodyne->camera transform, extended to a 4x4 homogeneous matrix."""
        tr = self.matrices["Tr_velo_to_cam"].reshape(3, 4)
        mat = np.eye(4)
        mat[:3, :4] = tr
        return mat
