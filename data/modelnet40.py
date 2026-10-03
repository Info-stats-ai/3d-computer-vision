"""ModelNet40 dataset (HDF5, 2048 points/sample) download + loading."""
import os
import glob
import h5py
import numpy as np
import urllib.request
import zipfile
from torch.utils.data import Dataset

DATA_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelnet40_ply_hdf5_2048")


def _download_and_extract(root):
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "modelnet40_ply_hdf5_2048.zip")
    if not os.path.exists(DATA_DIR):
        print(f"Downloading ModelNet40 (HDF5, 2048 pts) from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        print("Extracting ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        os.remove(zip_path)


def _load_split(split):
    """split: 'train' or 'test'. Returns (points [N,2048,3], labels [N])."""
    assert split in ("train", "test")
    file_list = os.path.join(DATA_DIR, f"{split}_files.txt")
    all_data, all_label = [], []
    with open(file_list) as f:
        files = [line.strip() for line in f if line.strip()]
    for fname in files:
        h5_name = os.path.join(DATA_DIR, os.path.basename(fname))
        with h5py.File(h5_name, "r") as f:
            data = f["data"][:].astype(np.float32)
            label = f["label"][:].astype(np.int64).reshape(-1)
        all_data.append(data)
        all_label.append(label)
    return np.concatenate(all_data, axis=0), np.concatenate(all_label, axis=0)


def load_class_names():
    names_path = os.path.join(DATA_DIR, "shape_names.txt")
    with open(names_path) as f:
        return [line.strip() for line in f if line.strip()]


class ModelNet40(Dataset):
    """ModelNet40 point cloud classification dataset.

    Each sample is a point cloud of shape [num_points, 3] and an integer
    class label in [0, 39].
    """

    def __init__(self, root=None, split="train", num_points=1024, augment=None):
        global DATA_DIR
        if root is not None:
            DATA_DIR = root
        _download_and_extract(os.path.dirname(DATA_DIR))
        self.data, self.labels = _load_split(split)
        self.num_points = num_points
        self.augment = augment
        self.class_names = load_class_names()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        points = self.data[idx]
        if self.num_points < points.shape[0]:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice]
        points = points.copy()
        if self.augment is not None:
            points = self.augment(points)
        label = int(self.labels[idx])
        return points.astype(np.float32), label
