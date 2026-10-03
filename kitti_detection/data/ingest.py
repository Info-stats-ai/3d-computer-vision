"""Step 1: Load a raw KITTI .bin LiDAR scan, preprocess, and optionally visualize.

Usage:
    python -m kitti_detection.data.ingest --bin_path <scan.bin> --visualize
"""
import argparse

import numpy as np
import open3d as o3d

from kitti_detection.data.kitti_io import load_velodyne_bin
from kitti_detection.data.preprocess import crop_range, remove_ground_plane, voxel_downsample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bin_path", type=str, required=True)
    p.add_argument("--voxel_size", type=float, default=0.1)
    p.add_argument("--ground_threshold", type=float, default=0.2)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--headless", action="store_true",
                    help="Save a screenshot instead of opening an interactive window.")
    p.add_argument("--output", type=str, default="viz_output/ingest.png")
    return p.parse_args()


def run_pipeline(bin_path, voxel_size=0.1, ground_threshold=0.2):
    """Returns dict with raw, cropped, downsampled, ground, non_ground point arrays."""
    raw = load_velodyne_bin(bin_path)
    xyz = raw[:, :3]
    print(f"Raw scan: {xyz.shape[0]} points")

    cropped = crop_range(xyz)
    print(f"After ROI crop: {cropped.shape[0]} points")

    downsampled = voxel_downsample(cropped, voxel_size=voxel_size)
    print(f"After voxel downsample (voxel={voxel_size}m): {downsampled.shape[0]} points")

    non_ground, ground, plane_model = remove_ground_plane(downsampled, distance_threshold=ground_threshold)
    print(f"After ground removal: {non_ground.shape[0]} non-ground, {ground.shape[0]} ground points "
          f"(plane: {plane_model})")

    return {
        "raw": xyz,
        "cropped": cropped,
        "downsampled": downsampled,
        "non_ground": non_ground,
        "ground": ground,
        "plane_model": plane_model,
    }


def visualize_result(result, headless=False, output_path="viz_output/ingest.png"):
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(result["ground"])
    ground_pcd.paint_uniform_color([0.6, 0.6, 0.6])  # gray = ground/road

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(result["non_ground"])
    obj_pcd.paint_uniform_color([0.1, 0.6, 0.9])  # blue = everything else (objects, buildings, etc.)

    geometries = [ground_pcd, obj_pcd]
    if headless:
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for g in geometries:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        print(f"Saved visualization -> {output_path}")
    else:
        o3d.visualization.draw_geometries(geometries, window_name="KITTI ingest: gray=ground, blue=objects")


def main():
    args = parse_args()
    result = run_pipeline(args.bin_path, voxel_size=args.voxel_size, ground_threshold=args.ground_threshold)
    if args.visualize:
        visualize_result(result, headless=args.headless, output_path=args.output)


if __name__ == "__main__":
    main()
