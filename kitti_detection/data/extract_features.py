"""Step 3: Extract 3D bounding-box features and line (road edge) features.

Usage:
    python -m kitti_detection.data.extract_features --label_path <label_2/000000.txt>
"""
import argparse
import json

import numpy as np

from kitti_detection.data.kitti_io import load_labels


def box_features(obj):
    """Geometric features of one detected/labeled 3D box: centroid, dims, yaw.
    KITTI label location/dims are in camera coordinates (x right, y down, z forward)."""
    x, y, z = obj.loc
    return {
        "type": obj.type,
        "centroid": [x, y, z],
        "dimensions": {"length": obj.l, "width": obj.w, "height": obj.h},
        "yaw": obj.yaw,
    }


def fit_line_ransac(points_xy, num_iterations=200, inlier_threshold=0.15):
    """Fit a 2D line (point + direction) to a set of (x, y) points via RANSAC.
    Used for road-edge / lane-marking line features from ground-plane boundary points."""
    best_inliers = None
    best_params = None
    n = len(points_xy)
    if n < 2:
        return None
    rng = np.random.default_rng(0)
    for _ in range(num_iterations):
        i, j = rng.choice(n, 2, replace=False)
        p1, p2 = points_xy[i], points_xy[j]
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction = direction / norm
        normal = np.array([-direction[1], direction[0]])
        dist = np.abs((points_xy - p1) @ normal)
        inliers = dist < inlier_threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_params = (p1, direction)
    if best_params is None:
        return None
    point, direction = best_params
    return {"point": point.tolist(), "direction": direction.tolist(),
            "num_inliers": int(best_inliers.sum())}


def extract_line_features(ground_points_xyz, num_lines=2):
    """Iteratively fits up to `num_lines` lines to ground-plane points (projected
    to the x-y plane), removing inliers after each fit — approximating road
    edges / lane markings as the dominant linear structures in the ground plane."""
    points_xy = ground_points_xyz[:, :2].copy()
    lines = []
    for _ in range(num_lines):
        if len(points_xy) < 10:
            break
        result = fit_line_ransac(points_xy)
        if result is None or result["num_inliers"] < 10:
            break
        lines.append(result)
        p1 = np.array(result["point"])
        direction = np.array(result["direction"])
        normal = np.array([-direction[1], direction[0]])
        dist = np.abs((points_xy - p1) @ normal)
        points_xy = points_xy[dist >= 0.15]
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label_path", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    objects = load_labels(args.label_path)
    features = [box_features(o) for o in objects]
    print(json.dumps(features, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(features, f, indent=2)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
