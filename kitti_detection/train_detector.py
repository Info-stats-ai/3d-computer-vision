"""Step 2: Train / fine-tune the PointPillars-style detector on KITTI
(car/pedestrian/cyclist), and adapt it to a new object class by extending
`class_names` and continuing training with the base classes' weights as
initialization.

Usage (real KITTI):
    python -m kitti_detection.train_detector --data_root data/kitti --epochs 30

Usage (synthetic smoke test, no download required):
    python -m kitti_detection.train_detector --synthetic --epochs 3

Adapt to a new class from a base checkpoint:
    python -m kitti_detection.train_detector --synthetic --epochs 3 \
        --classes Car Pedestrian Cyclist Pole --resume checkpoints/pointpillars_base.pth
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from kitti_detection.data.kitti_dataset import KittiDetectionDataset, SyntheticKittiDataset
from kitti_detection.models.pointpillars import PointPillarsDetector, build_targets, compute_loss


def collate_fn(batch):
    frame_ids, points, boxes = zip(*batch)
    return list(frame_ids), list(points), list(boxes)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None, help="Path to KITTI root (training/velodyne, training/label_2).")
    p.add_argument("--synthetic", action="store_true", help="Train on synthetic data instead of real KITTI.")
    p.add_argument("--classes", nargs="+", default=["Car", "Pedestrian", "Cyclist"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--voxel_size", type=float, default=0.32)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint_name", type=str, default="pointpillars.pth")
    p.add_argument("--resume", type=str, default=None,
                    help="Checkpoint to initialize from (e.g. a base-class checkpoint when adapting to a new class).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        dataset = SyntheticKittiDataset(num_samples=64, class_names=args.classes)
    else:
        if not args.data_root:
            raise ValueError("--data_root is required unless --synthetic is set.")
        dataset = KittiDetectionDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    detector = PointPillarsDetector(class_names=args.classes, voxel_size=args.voxel_size).to(device)
    optimizer = torch.optim.Adam(detector.parameters(), lr=args.lr)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        # Adapting to a new class: load matching params, leave new class's
        # head output channels randomly initialized (shapes differ -> skip).
        missing, unexpected = detector.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Resumed from {args.resume} (missing={len(missing)}, unexpected={len(unexpected)})")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    detector.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for frame_ids, points_batch, boxes_batch in loader:
            cls_logits, box_preds = detector(points_batch)
            _, _, feature_h, feature_w = (cls_logits.shape[0], cls_logits.shape[1],
                                           cls_logits.shape[2], cls_logits.shape[3])
            cls_targets, box_targets, pos_mask = build_targets(detector, boxes_batch, feature_h, feature_w)
            cls_targets, box_targets, pos_mask = cls_targets.to(device), box_targets.to(device), pos_mask.to(device)

            loss, parts = compute_loss(cls_logits, box_preds, cls_targets, box_targets, pos_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}")

    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    torch.save({"model_state": detector.state_dict(), "class_names": args.classes,
                "voxel_size": args.voxel_size}, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
