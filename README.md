# 3D Point Cloud Classification with PointNet++

A from-scratch PyTorch implementation of **PointNet++** trained on **ModelNet40**
for 3D point cloud object classification, with full training/evaluation pipeline
and Open3D-based visualization of predictions.

## What this is

- Real 3D point cloud data (ModelNet40, 40 object categories, ~12k meshes sampled to point clouds)
- A PointNet++ model (hierarchical set abstraction with farthest-point sampling +
  ball-query grouping), implemented in pure PyTorch so it runs on CPU or GPU
  without custom CUDA extensions
- A full train/eval loop with data augmentation (random rotation, jitter, scaling)
- Standard 3D metrics: **Overall Accuracy (OA)** and **per-class (mean) accuracy**,
  plus a **mIoU** utility for the segmentation head (ShapeNet part-seg extension)
- Open3D visualization of predicted vs. ground-truth labels on held-out point clouds

## What is a point cloud / what PointNet++ does differently

A point cloud is an unordered set of `(x, y, z, [features])` points sampled from a
3D surface — unlike an image grid, there's no fixed neighborhood structure, so a
2D CNN's convolution (slide a fixed kernel over a regular grid) doesn't apply
directly. PointNet (the predecessor) handles this by processing each point
independently with a shared MLP and pooling the result into a single global
feature (symmetric, permutation-invariant) — but this throws away local
geometric structure. **PointNet++** fixes that by building a hierarchy: it
repeatedly (1) samples a subset of points via farthest-point sampling, (2)
groups each sampled point's local neighborhood via ball-query radius search,
and (3) runs a mini-PointNet over each local group — stacking several of these
"set abstraction" layers to learn features at increasing spatial scale, similar
in spirit to how a CNN's pooling pyramid builds up receptive field.

## Project layout

```
data/
  modelnet40.py       # Dataset download + loading (ModelNet40 HDF5, 2048 pts/sample)
  augmentation.py      # Point cloud augmentations (rotate, jitter, scale)
models/
  pointnet2_utils.py   # FPS, ball query, grouping, set abstraction layer
  pointnet2_cls.py      # Full PointNet++ classification network
utils/
  metrics.py            # OA, per-class accuracy, mIoU
train.py                # Training loop
evaluate.py             # Evaluation on the test split
visualize.py             # Open3D visualization of predictions
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 100 --batch_size 32 --num_points 1024
```

The ModelNet40 HDF5 dataset (~435 MB) is downloaded automatically into `./data/modelnet40_ply_hdf5_2048/`
on first run.

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

Reports Overall Accuracy (OA) and mean per-class accuracy on the official test split.

## Visualize predictions

```bash
python visualize.py --checkpoint checkpoints/best_model.pth --num_samples 8
```

Renders held-out point clouds colored by prediction correctness (green = correct,
red = incorrect) with the predicted/ground-truth class label, using Open3D.

## Status

- [x] Dataset pipeline (ModelNet40 download + loader)
- [x] PointNet++ model (set abstraction hierarchy)
- [x] Training loop with augmentation
- [x] Evaluation (OA, per-class accuracy, mIoU utility)
- [x] Open3D visualization
