"""Standard 3D classification/segmentation metrics: OA, per-class accuracy, mIoU."""
import numpy as np


def overall_accuracy(preds, labels):
    """Fraction of correctly classified samples, over all samples (OA)."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    return float((preds == labels).mean())


def per_class_accuracy(preds, labels, num_classes):
    """Accuracy computed per class then averaged (mean class accuracy),
    which — unlike OA — isn't dominated by frequent classes."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    accs = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        accs.append(float((preds[mask] == labels[mask]).mean()))
    return float(np.mean(accs)) if accs else 0.0


def mean_iou(preds, labels, num_classes):
    """mIoU over per-point predictions, for segmentation tasks: for each class,
    IoU = |pred ∩ gt| / |pred ∪ gt|, averaged over classes present in the sample.
    preds, labels: arrays of per-point integer labels of the same shape.
    """
    preds, labels = np.asarray(preds), np.asarray(labels)
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        union = np.logical_or(pred_c, label_c).sum()
        if union == 0:
            continue  # class not present in pred or gt for this sample
        intersection = np.logical_and(pred_c, label_c).sum()
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


class ConfusionAccumulator:
    """Accumulates confusion-matrix stats across batches for OA / per-class accuracy
    without keeping every prediction in memory."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.correct_per_class = np.zeros(num_classes, dtype=np.int64)
        self.total_per_class = np.zeros(num_classes, dtype=np.int64)
        self.correct = 0
        self.total = 0

    def update(self, preds, labels):
        preds, labels = np.asarray(preds), np.asarray(labels)
        self.correct += int((preds == labels).sum())
        self.total += len(labels)
        for c in range(self.num_classes):
            mask = labels == c
            self.total_per_class[c] += mask.sum()
            self.correct_per_class[c] += (preds[mask] == labels[mask]).sum()

    def overall_accuracy(self):
        return self.correct / max(self.total, 1)

    def mean_class_accuracy(self):
        valid = self.total_per_class > 0
        if not valid.any():
            return 0.0
        return float(np.mean(self.correct_per_class[valid] / self.total_per_class[valid]))
