"""Step 5: Render predicted vs. ground-truth 3D boxes on the LiDAR point cloud.

Usage:
    python -m kitti_detection.viz.render --bin_path <scan.bin> --preds preds.json \
        --gt gt.json --frame_id 000000 [--headless]
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d

from kitti_detection.data.kitti_io import load_velodyne_bin
from kitti_detection.eval.box_iou import box_corners_bev

BOX_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0],  # top rectangle
             [4, 5], [5, 6], [6, 7], [7, 4],  # bottom rectangle
             [0, 4], [1, 5], [2, 6], [3, 7]]  # verticals


def box_to_lineset(box, color):
    """Builds an Open3D LineSet wireframe for one 3D box dict
    ({'centroid', 'dimensions', 'yaw'}), in camera coordinates (x,y,z)."""
    centroid = box["centroid"]
    dims = box["dimensions"]
    top_corners = box_corners_bev(centroid, dims["length"], dims["width"], box["yaw"])
    y_top = centroid[1] - dims["height"]
    y_bot = centroid[1]

    points = []
    for cx, cz in top_corners:
        points.append([cx, y_top, cz])
    for cx, cz in top_corners:
        points.append([cx, y_bot, cz])

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.array(points))
    lineset.lines = o3d.utility.Vector2iVector(np.array(BOX_EDGES))
    lineset.colors = o3d.utility.Vector3dVector([color for _ in BOX_EDGES])
    return lineset


def render(bin_path, pred_boxes, gt_boxes, headless=False, output_path="viz_output/render.png"):
    scan = load_velodyne_bin(bin_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan[:, :3])
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    geometries = [pcd]
    geometries += [box_to_lineset(b, [0.1, 0.8, 0.1]) for b in gt_boxes]   # green = ground truth
    geometries += [box_to_lineset(b, [0.9, 0.1, 0.1]) for b in pred_boxes]  # red = prediction

    if headless:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for g in geometries:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        print(f"Saved -> {output_path}")
    else:
        o3d.visualization.draw_geometries(geometries, window_name="green=GT, red=prediction")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin_path", type=str, required=True)
    p.add_argument("--preds", type=str, required=True)
    p.add_argument("--gt", type=str, required=True)
    p.add_argument("--frame_id", type=str, required=True)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output", type=str, default="viz_output/render.png")
    args = p.parse_args()

    with open(args.preds) as f:
        preds = json.load(f)
    with open(args.gt) as f:
        gt = json.load(f)

    pred_boxes = next((r["boxes"] for r in preds if r["frame_id"] == args.frame_id), [])
    gt_boxes = next((r["boxes"] for r in gt if r["frame_id"] == args.frame_id), [])

    render(args.bin_path, pred_boxes, gt_boxes, headless=args.headless, output_path=args.output)


if __name__ == "__main__":
    main()
