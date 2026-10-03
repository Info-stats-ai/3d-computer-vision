"""Step 4: 3D detection evaluation — IoU-based precision/recall and a
regression test against a stored baseline (suitable as a CI gate).

Expected input JSON format: a list of per-frame records, each
    {"frame_id": str, "boxes": [{"type": str, "centroid": [x,y,z],
      "dimensions": {"length","width","height"}, "yaw": float, "score": float}]}
`score` is required for predictions, ignored (optional) for ground truth.

Usage:
    python -m kitti_detection.eval.evaluate --preds preds.json --gt gt.json \
        [--baseline baseline_metrics.json] [--iou_threshold 0.5] [--tolerance 0.02]
"""
import argparse
import json
import sys

from kitti_detection.eval.box_iou import iou_3d


def match_frame(pred_boxes, gt_boxes, iou_threshold, object_type=None):
    """Greedy matching by descending prediction score. Returns (tp, fp, fn)."""
    preds = [b for b in pred_boxes if object_type is None or b["type"] == object_type]
    gts = [b for b in gt_boxes if object_type is None or b["type"] == object_type]
    preds = sorted(preds, key=lambda b: -b.get("score", 1.0))

    matched_gt = set()
    tp = 0
    for pred in preds:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue
            iou = iou_3d(pred, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def evaluate(preds, gt, iou_threshold=0.5, object_types=None):
    """preds, gt: list of per-frame records (same frame order/frame_id).
    Returns dict of {object_type: {precision, recall, tp, fp, fn}} plus 'overall'."""
    gt_by_frame = {r["frame_id"]: r["boxes"] for r in gt}
    types = object_types or sorted({b["type"] for r in gt for b in r["boxes"]})

    metrics = {}
    for obj_type in types:
        tp = fp = fn = 0
        for pred_record in preds:
            gt_boxes = gt_by_frame.get(pred_record["frame_id"], [])
            t, f, n = match_frame(pred_record["boxes"], gt_boxes, iou_threshold, obj_type)
            tp += t
            fp += f
            fn += n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[obj_type] = {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}

    all_p = [m["precision"] for m in metrics.values()]
    all_r = [m["recall"] for m in metrics.values()]
    metrics["overall"] = {
        "precision": sum(all_p) / len(all_p) if all_p else 0.0,
        "recall": sum(all_r) / len(all_r) if all_r else 0.0,
    }
    return metrics


def check_regression(metrics, baseline, tolerance=0.02):
    """Returns list of regression messages (empty if none). A metric regresses
    if it drops by more than `tolerance` below the baseline's value."""
    regressions = []
    for obj_type, base_vals in baseline.items():
        if obj_type not in metrics:
            continue
        for key in ("precision", "recall"):
            if key not in base_vals:
                continue
            new_val = metrics[obj_type].get(key, 0.0)
            base_val = base_vals[key]
            if new_val < base_val - tolerance:
                regressions.append(
                    f"REGRESSION [{obj_type}.{key}]: {new_val:.4f} < baseline {base_val:.4f} "
                    f"(tolerance {tolerance})")
    return regressions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=str, required=True)
    p.add_argument("--gt", type=str, required=True)
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--save_metrics", type=str, default=None,
                    help="Write this run's metrics to a file (e.g. to become the next baseline).")
    args = p.parse_args()

    with open(args.preds) as f:
        preds = json.load(f)
    with open(args.gt) as f:
        gt = json.load(f)

    metrics = evaluate(preds, gt, iou_threshold=args.iou_threshold)
    print(json.dumps(metrics, indent=2))

    if args.save_metrics:
        with open(args.save_metrics, "w") as f:
            json.dump(metrics, f, indent=2)

    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        regressions = check_regression(metrics, baseline, tolerance=args.tolerance)
        if regressions:
            print("\n".join(regressions), file=sys.stderr)
            sys.exit(1)
        print("No regressions vs. baseline.")


if __name__ == "__main__":
    main()
