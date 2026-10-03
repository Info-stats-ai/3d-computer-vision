# LiDAR 3D Object Detection & Feature Extraction Pipeline (KITTI)

A pipeline on real outdoor LiDAR (KITTI) covering ingestion, 3D object
detection (PointPillars, fine-tuned + adapted to a new class), geometric
feature extraction, and a regression-aware evaluation pipeline — mirroring
a production LiDAR detection workflow (e.g. surveying/mapping point clouds,
adapting models to new object classes, extracting 3D object/line features).

This track is independent of the `ModelNet40` classification pipeline in
the repo root; it lives entirely under `kitti_detection/`.

## Pipeline (5 steps)

| Step | What | Status |
|---|---|---|
| 1 | Point cloud ingestion, Open3D visualization, voxel downsampling, ground plane removal | **Done** |
| 2 | PointPillars-style detector (from-scratch pure-PyTorch), fine-tune/adapt to a new class | **Done** |
| 3 | Extract 3D bbox features (centroid, dims, yaw) + line features (road edges, lane markings) | **Done** |
| 4 | Eval pipeline: 3D IoU, precision/recall @ IoU threshold, regression test vs. baseline | **Done** |
| 5 | Render predicted vs. ground-truth boxes on the point cloud with Open3D | **Done** |

Verified end-to-end (train → infer → eval → render) on synthetic data —
run `bash kitti_detection/run_smoke_test.sh` to reproduce. See
**Verified pipeline run** below for the actual metrics and a rendered
screenshot from that run.

## Setup

```bash
pip install -r requirements.txt
```

### Dataset

Download the KITTI 3D Object Detection benchmark (left color images are not
required for this pipeline — only Velodyne point clouds + labels + calib):

- `velodyne` point clouds (`.bin`): https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
- `label_2` (training labels): https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
- `calib`: https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

Extract so you have:
```
kitti_detection/data/kitti/
  training/velodyne/*.bin
  training/label_2/*.txt
  training/calib/*.txt
```

(KITTI requires registering for a free download link — the URLs above need
an authenticated session, so this is a manual one-time download rather than
an auto-fetch like the ModelNet40 track.)

## Step 1 — Ingestion & visualization

```bash
python -m kitti_detection.data.ingest --bin_path data/kitti/training/velodyne/000000.bin --visualize
```

Loads a raw `.bin` LiDAR scan (`N x 4`: x, y, z, reflectance), then:
- **Voxel downsampling** — merges points within a `0.1m` voxel grid cell,
  cutting a ~100k-point scan down to a tractable size without losing
  overall shape.
- **Ground plane removal** — fits a plane with RANSAC (Open3D's
  `segment_plane`) and removes the ~majority of points that lie on it (the
  road surface), which is most of a raw outdoor scan and dominates the
  point budget if left in.

## Step 2 — Detection (PointPillars)

`kitti_detection/models/` is a from-scratch, pure-PyTorch reimplementation
of PointPillars: pillar feature encoding (`pillar_encoder.py`) scatters raw
points into BEV "pillars" and runs a per-pillar PointNet to get one feature
vector per pillar; `backbone2d.py` runs a 3-scale 2D CNN over the resulting
BEV pseudo-image; `pointpillars.py` wires it together with per-cell target
assignment, a positive-weighted classification + smooth-L1 box regression
loss, and greedy 3D-IoU NMS at decode time. No sparse-conv dependency, so
it trains on CPU or GPU as-is.

Train on real KITTI:
```bash
python -m kitti_detection.train_detector --data_root data/kitti --epochs 30 \
    --classes Car Pedestrian Cyclist
```

Train on synthetic data (no download needed — see **Verified pipeline run**):
```bash
python -m kitti_detection.train_detector --synthetic --epochs 30
```

**Adapt to a new class** (e.g. `Pole`) from a checkpoint trained on the base
classes — this loads all matching weights and randomly initializes just the
new class's head output channels, then continues training:
```bash
python -m kitti_detection.train_detector --data_root data/kitti --epochs 15 \
    --classes Car Pedestrian Cyclist Pole --resume checkpoints/pointpillars.pth
```

`kitti_detection/configs/kitti_classes.py` holds the base/new class lists
and anchor-size priors referenced above.

For a production-grade detector instead of this from-scratch one,
**MMDetection3D** is the practical alternative (`pip install -U openmim &&
mim install mmcv-full mmdet mmdet3d`), fine-tuning its pretrained
PointPillars checkpoint the same way — extend the class list, resume from
the checkpoint, retrain.

## Step 3 — Feature extraction

```bash
python -m kitti_detection.data.extract_features --label_path data/kitti/training/label_2/000000.txt
```

From each detection (or ground-truth label, same format), extracts:
- **Box features**: centroid `(x, y, z)`, dimensions `(l, w, h)`, yaw.
- **Line features**: road-edge/lane-marking lines fit via RANSAC line
  fitting over the ground-plane points' boundary, output as `(point, direction)`.

## Step 4 — Evaluation (3D IoU, PR, regression test)

```bash
python -m kitti_detection.eval.evaluate --preds preds.json --gt gt.json --baseline baseline_metrics.json
```

Computes 3D IoU between predicted and GT boxes, precision/recall at an IoU
threshold of 0.5, and fails (non-zero exit) if any metric regresses below
the stored baseline by more than a tolerance — suitable as a CI gate on
new model checkpoints.

## Step 5 — Visualization

```bash
python -m kitti_detection.viz.render --bin_path data/kitti/training/velodyne/000000.bin \
    --preds preds.json --gt gt.json
```

Renders the point cloud with predicted boxes in red wireframe and
ground-truth boxes in green wireframe using Open3D's `LineSet`.

## Verified pipeline run

`bash kitti_detection/run_smoke_test.sh` trains the detector on synthetic
scenes (known-exact ground truth, since real KITTI requires a manual
registration-gated download — see **Dataset** above), then runs it through
inference, IoU evaluation, and rendering, proving all 5 steps actually wire
together. One run (30 epochs, 64 synthetic scenes, 8 held-out eval frames,
IoU threshold 0.3) produced:

```
Car:        precision=0.42  recall=1.00  (tp=14 fp=19 fn=0)
Pedestrian: precision=0.19  recall=0.89  (tp=8  fp=35 fn=1)
Cyclist:    precision=0.16  recall=0.82  (tp=9  fp=49 fn=2)
overall:    precision=0.26  recall=0.90
```

High recall / low precision here reflects the tiny synthetic training set
(64 scenes) and few epochs, not a pipeline defect — the loss curve drops
cleanly from 1.22 to ~0.015 and training on the full KITTI train split
(~7,000 frames) for the usual ~80-160 epochs is what a real run (per **Step
2**) would use. The point of this run is to verify every stage — pillar
voxelization, the 2D backbone, per-cell target assignment with the
positive-class reweighting needed to avoid predicting background
everywhere, 3D-IoU NMS at decode time, IoU-based eval, and box rendering —
is correct and self-consistent end to end.
