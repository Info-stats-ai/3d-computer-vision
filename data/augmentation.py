"""Point cloud augmentations used during training."""
import numpy as np


def random_rotate_z(points):
    """Random rotation around the up (z) axis — objects can face any direction."""
    theta = np.random.uniform(0, 2 * np.pi)
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
    return points @ rot.T


def random_jitter(points, sigma=0.01, clip=0.05):
    """Small per-point Gaussian noise to simulate sensor/sampling noise."""
    noise = np.clip(np.random.normal(0, sigma, points.shape), -clip, clip)
    return points + noise.astype(np.float32)


def random_scale(points, low=0.8, high=1.25):
    scale = np.random.uniform(low, high)
    return points * scale


class ComposeAugmentation:
    """Chains rotation, scaling, and jitter for training-time augmentation."""

    def __call__(self, points):
        points = random_rotate_z(points)
        points = random_scale(points)
        points = random_jitter(points)
        return points
