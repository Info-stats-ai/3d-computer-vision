#!/usr/bin/env bash
# End-to-end smoke test of the full 5-step pipeline on synthetic data
# (no KITTI download required). Verifies: train -> infer -> eval -> render
# all wire together correctly. Run from the repo root.
set -euo pipefail

CKPT_DIR=${1:-checkpoints}
OUT_DIR=${2:-viz_output}
mkdir -p "$CKPT_DIR" "$OUT_DIR"

echo "== Step 2: training PointPillars on synthetic data =="
python -m kitti_detection.train_detector --synthetic --epochs 30 --batch_size 4 \
    --checkpoint_dir "$CKPT_DIR" --checkpoint_name pointpillars_smoketest.pth

echo "== Step 2/4: running inference + exporting preds/gt =="
python -m kitti_detection.infer --checkpoint "$CKPT_DIR/pointpillars_smoketest.pth" --synthetic \
    --num_samples 8 --score_threshold 0.5 --preds_out "$OUT_DIR/preds.json" --gt_out "$OUT_DIR/gt.json"

echo "== Step 4: evaluating (3D IoU precision/recall) =="
python -m kitti_detection.eval.evaluate --preds "$OUT_DIR/preds.json" --gt "$OUT_DIR/gt.json" \
    --iou_threshold 0.3 --save_metrics "$OUT_DIR/metrics.json"

echo "== Step 5: rendering one sample (headless) =="
python -c "
from kitti_detection.data.kitti_dataset import SyntheticKittiDataset
import numpy as np
ds = SyntheticKittiDataset(num_samples=1, class_names=['Car','Pedestrian','Cyclist'])
_, pts, _ = ds[0]
pts.astype(np.float32).tofile('$OUT_DIR/smoketest_frame.bin')
"
python -m kitti_detection.viz.render --bin_path "$OUT_DIR/smoketest_frame.bin" \
    --preds "$OUT_DIR/preds.json" --gt "$OUT_DIR/gt.json" --frame_id synthetic_0 \
    --headless --output "$OUT_DIR/render.png"

echo "Smoke test complete. See $OUT_DIR/metrics.json and $OUT_DIR/render.png"
