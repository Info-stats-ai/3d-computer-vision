"""Class list config used when fine-tuning / adapting a PointPillars detection
head (via MMDetection3D) to a new object class, per Step 2 of this pipeline.
"""

# Base KITTI classes the pretrained PointPillars checkpoint was trained on.
BASE_CLASSES = ["Car", "Pedestrian", "Cyclist"]

# New class(es) to adapt the detector to. Extend the detection head's output
# layer (num_classes) and anchor generator to include these, then fine-tune
# with the base classes' pretrained backbone weights frozen or lightly
# fine-tuned, and the new class trained from a small labeled subset.
NEW_CLASSES = ["Pole", "TrafficCone"]

ALL_CLASSES = BASE_CLASSES + NEW_CLASSES

# Typical anchor sizes (length, width, height in meters) per class — used by
# PointPillars' anchor generator. New-class anchors are rough priors to be
# refined from the new class's labeled box-size distribution.
ANCHOR_SIZES = {
    "Car": [3.9, 1.6, 1.56],
    "Pedestrian": [0.8, 0.6, 1.73],
    "Cyclist": [1.76, 0.6, 1.73],
    "Pole": [0.3, 0.3, 3.0],
    "TrafficCone": [0.4, 0.4, 0.7],
}
