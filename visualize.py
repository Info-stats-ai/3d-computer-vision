"""Visualize PointNet++ predictions on held-out ModelNet40 point clouds with Open3D.

Each sample is rendered as a point cloud, colored green if the model's
prediction matches ground truth and red otherwise, with predicted/true
class names printed to the console (and saved as a screenshot when run
headless, since Open3D's interactive viewer needs a display).
"""
import argparse
import os

import numpy as np
import open3d as o3d
import torch

from data.modelnet40 import ModelNet40
from models.pointnet2_cls import PointNet2Classifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="viz_output")
    p.add_argument("--headless", action="store_true",
                    help="Save screenshots instead of opening an interactive window.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = ModelNet40(split="test", num_points=args.num_points, augment=None)
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt.get("num_classes", len(test_set.class_names))
    model = PointNet2Classifier(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    indices = np.random.choice(len(test_set), min(args.num_samples, len(test_set)), replace=False)

    for n, idx in enumerate(indices):
        points, label = test_set[idx]
        with torch.no_grad():
            logits = model(torch.from_numpy(points).unsqueeze(0).to(device))
            pred = int(logits.argmax(dim=-1).item())

        correct = pred == label
        color = [0.1, 0.8, 0.1] if correct else [0.9, 0.1, 0.1]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)

        pred_name = test_set.class_names[pred]
        true_name = test_set.class_names[label]
        status = "CORRECT" if correct else "WRONG"
        print(f"[{n}] pred={pred_name} true={true_name} -> {status}")

        if args.headless:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            out_path = os.path.join(args.output_dir, f"sample_{n}_{status}_{pred_name}.png")
            vis.capture_screen_image(out_path)
            vis.destroy_window()
            print(f"  saved -> {out_path}")
        else:
            o3d.visualization.draw_geometries(
                [pcd], window_name=f"pred={pred_name} true={true_name} ({status})")


if __name__ == "__main__":
    main()
