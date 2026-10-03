"""Run a trained PointPillars checkpoint over a dataset and export predictions
+ ground truth in the JSON format expected by eval/evaluate.py and viz/render.py,
closing the loop from Step 2 -> Step 4 -> Step 5.

Usage:
    python -m kitti_detection.infer --checkpoint checkpoints/pointpillars.pth --synthetic \
        --preds_out preds.json --gt_out gt.json
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from kitti_detection.data.kitti_dataset import KittiDetectionDataset, SyntheticKittiDataset
from kitti_detection.models.pointpillars import PointPillarsDetector, decode_predictions
from kitti_detection.train_detector import collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--score_threshold", type=float, default=0.5)
    p.add_argument("--preds_out", type=str, default="preds.json")
    p.add_argument("--gt_out", type=str, default="gt.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]
    detector = PointPillarsDetector(class_names=class_names, voxel_size=ckpt.get("voxel_size", 0.32)).to(device)
    detector.load_state_dict(ckpt["model_state"])
    detector.eval()

    if args.synthetic:
        dataset = SyntheticKittiDataset(num_samples=args.num_samples, class_names=class_names)
    else:
        if not args.data_root:
            raise ValueError("--data_root is required unless --synthetic is set.")
        dataset = KittiDetectionDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    preds_records, gt_records = [], []
    with torch.no_grad():
        for i, (frame_ids, points_batch, boxes_batch) in enumerate(loader):
            if i >= args.num_samples:
                break
            cls_logits, box_preds = detector(points_batch)
            detections = decode_predictions(
                detector, cls_logits[0], box_preds[0], score_threshold=args.score_threshold)
            preds_records.append({"frame_id": frame_ids[0], "boxes": detections})
            gt_records.append({"frame_id": frame_ids[0], "boxes": boxes_batch[0]})
            print(f"[{frame_ids[0]}] {len(detections)} detections, {len(boxes_batch[0])} ground-truth boxes")

    with open(args.preds_out, "w") as f:
        json.dump(preds_records, f, indent=2)
    with open(args.gt_out, "w") as f:
        json.dump(gt_records, f, indent=2)
    print(f"Saved -> {args.preds_out}, {args.gt_out}")


if __name__ == "__main__":
    main()
