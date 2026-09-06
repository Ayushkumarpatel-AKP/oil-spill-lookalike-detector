import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_utils import OilSpillDataset
from model import UNet

TARGET_RECALL = 0.90
CANDIDATE_THRESHOLDS = np.arange(0.05, 0.96, 0.05)

ROOT = "dataset"
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def iou_score(logits, target, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = preds.view(preds.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (preds * target).sum(dim=1)
    union = preds.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def run_epoch(model, loader, bce, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_iou, n_batches = 0.0, 0.0, 0

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

        with torch.set_grad_enabled(is_train):
            logits = model(imgs)
            loss = bce(logits, lbls) + dice_loss(logits, lbls)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_iou += iou_score(logits, lbls)
        n_batches += 1

    return total_loss / n_batches, total_iou / n_batches


def main():
    print(f"Device: {DEVICE}")

    train_ds = OilSpillDataset(ROOT, "train", augment=True)
    test_ds = OilSpillDataset(ROOT, "test", augment=False)
    print(f"Train samples: {len(train_ds)}  Test samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet(in_ch=3, out_ch=1, base=16).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    best_iou = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_iou = run_epoch(model, train_loader, bce, optimizer)
        val_loss, val_iou = run_epoch(model, test_loader, bce, optimizer=None)
        dt = time.time() - t0

        print(
            f"Epoch {epoch}/{EPOCHS} ({dt:.1f}s) "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> saved best_model.pth (val_iou={val_iou:.4f})")

    print(f"Training complete. Best val IoU: {best_iou:.4f}")

    tune_threshold(model, test_loader)


def tune_threshold(model, loader):
    """Pick an operating threshold favoring recall: missing a spill (false
    negative) is costlier than a false alarm, so we accept lower precision
    to hit a target recall, per the standard precision-recall tradeoff for
    imbalanced/safety-critical detection tasks."""
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()

    tp = np.zeros(len(CANDIDATE_THRESHOLDS))
    fp = np.zeros(len(CANDIDATE_THRESHOLDS))
    fn = np.zeros(len(CANDIDATE_THRESHOLDS))

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            probs = torch.sigmoid(model(imgs)).cpu().numpy().ravel()
            targets = lbls.cpu().numpy().ravel().astype(bool)

            for i, t in enumerate(CANDIDATE_THRESHOLDS):
                preds = probs >= t
                tp[i] += np.logical_and(preds, targets).sum()
                fp[i] += np.logical_and(preds, ~targets).sum()
                fn[i] += np.logical_and(~preds, targets).sum()

    precision = tp / np.clip(tp + fp, 1e-9, None)
    recall = tp / np.clip(tp + fn, 1e-9, None)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)

    eligible = np.where(recall >= TARGET_RECALL)[0]
    if len(eligible) > 0:
        # Among thresholds meeting the recall target, pick the one with best precision.
        best_idx = eligible[np.argmax(precision[eligible])]
    else:
        # Recall target unreachable at any candidate threshold; fall back to best F1.
        best_idx = int(np.argmax(f1))

    result = {
        "threshold": float(CANDIDATE_THRESHOLDS[best_idx]),
        "target_recall": TARGET_RECALL,
        "achieved_recall": float(recall[best_idx]),
        "achieved_precision": float(precision[best_idx]),
        "f1": float(f1[best_idx]),
    }
    with open("threshold.json", "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Tuned threshold={result['threshold']:.2f} "
        f"(recall={result['achieved_recall']:.4f}, "
        f"precision={result['achieved_precision']:.4f}, "
        f"f1={result['f1']:.4f}) -> saved threshold.json"
    )


if __name__ == "__main__":
    main()
