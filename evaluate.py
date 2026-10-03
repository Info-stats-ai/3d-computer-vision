"""Evaluate a trained PointNet++ checkpoint on the ModelNet40 test split."""
import argparse

import torch
from torch.utils.data import DataLoader

from data.modelnet40 import ModelNet40
from models.pointnet2_cls import PointNet2Classifier
from utils.metrics import ConfusionAccumulator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = ModelNet40(split="test", num_points=args.num_points, augment=None)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt.get("num_classes", len(test_set.class_names))
    model = PointNet2Classifier(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    acc = ConfusionAccumulator(num_classes)
    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            logits = model(points)
            preds = logits.argmax(dim=-1)
            acc.update(preds.cpu().numpy(), labels.cpu().numpy())

    print(f"Overall Accuracy (OA):        {acc.overall_accuracy():.4f}")
    print(f"Mean per-class accuracy:      {acc.mean_class_accuracy():.4f}")
    print(f"Evaluated on {acc.total} test samples, {num_classes} classes.")


if __name__ == "__main__":
    main()
