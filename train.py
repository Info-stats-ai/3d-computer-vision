"""Training loop for PointNet++ classification on ModelNet40."""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.modelnet40 import ModelNet40
from data.augmentation import ComposeAugmentation
from models.pointnet2_cls import PointNet2Classifier, get_loss
from utils.metrics import ConfusionAccumulator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def evaluate(model, loader, device, num_classes):
    model.eval()
    acc = ConfusionAccumulator(num_classes)
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            logits = model(points)
            preds = logits.argmax(dim=-1)
            acc.update(preds.cpu().numpy(), labels.cpu().numpy())
    return acc.overall_accuracy(), acc.mean_class_accuracy()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = ModelNet40(split="train", num_points=args.num_points, augment=ComposeAugmentation())
    test_set = ModelNet40(split="test", num_points=args.num_points, augment=None)
    num_classes = len(train_set.class_names)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    model = PointNet2Classifier(num_classes=num_classes).to(device)
    criterion = get_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    start_epoch = 0
    best_oa = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_oa = ckpt.get("best_oa", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, (points, labels) in enumerate(pbar):
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % args.log_every == 0:
                pbar.set_postfix(loss=total_loss / (i + 1))
        scheduler.step()

        oa, mean_acc = evaluate(model, test_loader, device, num_classes)
        elapsed = time.time() - start
        print(f"Epoch {epoch + 1}: train_loss={total_loss / len(train_loader):.4f} "
              f"test_OA={oa:.4f} test_mean_class_acc={mean_acc:.4f} ({elapsed:.1f}s)")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_oa": best_oa,
            "num_classes": num_classes,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last_model.pth"))
        if oa > best_oa:
            best_oa = oa
            ckpt["best_oa"] = best_oa
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best_model.pth"))
            print(f"  -> new best OA={best_oa:.4f}, checkpoint saved")

    print(f"Training complete. Best test OA: {best_oa:.4f}")


if __name__ == "__main__":
    main()
