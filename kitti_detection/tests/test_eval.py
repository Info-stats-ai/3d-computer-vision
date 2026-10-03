"""Lightweight sanity tests for box IoU and the eval pipeline (no KITTI data needed)."""
from kitti_detection.eval.box_iou import iou_3d
from kitti_detection.eval.evaluate import evaluate, check_regression


def test_identical_boxes_iou_is_one():
    box = {"centroid": [0, 0, 0], "dimensions": {"length": 4, "width": 2, "height": 1.5}, "yaw": 0.0}
    assert abs(iou_3d(box, box) - 1.0) < 1e-6


def test_disjoint_boxes_iou_is_zero():
    box_a = {"centroid": [0, 0, 0], "dimensions": {"length": 1, "width": 1, "height": 1}, "yaw": 0.0}
    box_b = {"centroid": [100, 0, 100], "dimensions": {"length": 1, "width": 1, "height": 1}, "yaw": 0.0}
    assert iou_3d(box_a, box_b) == 0.0


def test_evaluate_perfect_match():
    box = {"type": "Car", "centroid": [0, 0, 0], "dimensions": {"length": 4, "width": 2, "height": 1.5},
           "yaw": 0.0, "score": 0.9}
    preds = [{"frame_id": "0", "boxes": [box]}]
    gt = [{"frame_id": "0", "boxes": [box]}]
    metrics = evaluate(preds, gt, iou_threshold=0.5)
    assert metrics["Car"]["precision"] == 1.0
    assert metrics["Car"]["recall"] == 1.0


def test_evaluate_missed_detection():
    gt_box = {"type": "Car", "centroid": [0, 0, 0], "dimensions": {"length": 4, "width": 2, "height": 1.5},
              "yaw": 0.0}
    preds = [{"frame_id": "0", "boxes": []}]
    gt = [{"frame_id": "0", "boxes": [gt_box]}]
    metrics = evaluate(preds, gt, iou_threshold=0.5)
    assert metrics["Car"]["recall"] == 0.0
    assert metrics["Car"]["fn"] == 1


def test_check_regression_flags_drop():
    metrics = {"Car": {"precision": 0.5, "recall": 0.5}}
    baseline = {"Car": {"precision": 0.8, "recall": 0.8}}
    regressions = check_regression(metrics, baseline, tolerance=0.02)
    assert len(regressions) == 2


def test_check_regression_within_tolerance():
    metrics = {"Car": {"precision": 0.79, "recall": 0.8}}
    baseline = {"Car": {"precision": 0.8, "recall": 0.8}}
    regressions = check_regression(metrics, baseline, tolerance=0.02)
    assert len(regressions) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
