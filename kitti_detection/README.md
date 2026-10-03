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
| 2 | Fine-tune pretrained PointPillars on KITTI car/pedestrian/cyclist, adapt to a new class | Planned (see below) |
| 3 | Extract 3D bbox features (centroid, dims, yaw) + line features (road edges, lane markings) | **Done** (bbox); line features planned |
| 4 | Eval pipeline: 3D IoU, precision/recall @0.5 IoU, regression test vs. baseline | **Done** |
| 5 | Render predicted vs. ground-truth boxes on the point cloud with Open3D | **Done** |

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

## Step 2 — Detection (fine-tuning PointPillars)

Training a PointPillars/VoxelNet detector from scratch is out of scope for
a pure-PyTorch from-scratch implementation (it depends on sparse
voxelization ops); the practical path is **MMDetection3D**:

```bash
pip install -U openmim
mim install mmcv-full mmdet mmdet3d
# fine-tune a pretrained PointPillars checkpoint on KITTI, then on a new
# class by extending the config's class list and re-training the head:
python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py \
    --cfg-options load_from=<pretrained_checkpoint.pth>
```

`kitti_detection/configs/` holds the class-list + anchor config used to add
a new object class (e.g. `pole`/`traffic_cone`) by extending the existing
car/pedestrian/cyclist head rather than retraining from scratch.

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
