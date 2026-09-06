"""Train a real Oil-vs-Look-alike classifier from labeled ground truth,
replacing the fixed-weight heuristic in lookalike.py with a fitted one.

Data source: Zenodo 10.5281/zenodo.13761290, "Sentinel-1 SAR Oil spill image
dataset for train, validate, and test deep learning models. Part III"
(CC-BY-4.0, no login required). After downloading and extracting
02_Test_images_and_ground_truth.7z, the layout is:

    <ZENODO_ROOT>/Images/{Oil,Lookalike,No oil}/NNNNN.tif             -- dual-pol Sigma0 dB, 2048x2048
    <ZENODO_ROOT>/Mask/{Oil,Lookalike,No oil}/NNNNN_segmentation.tif  -- binary mask, 2048x2048

Important asymmetry discovered by inspecting the actual files: the "mask" is
a ground-truth *oil* mask, not a per-class region mask. For "Oil" images it
correctly marks the annotated oil pixels. For every "Lookalike" (and "No
oil") image the mask is **entirely zero** -- because there is no oil, by
definition. The "look-alike" label is therefore an image-level fact ("this
whole scene is a confirmed non-oil look-alike phenomenon"), not a pixel
region, so there is nothing to take connected components of directly.

To get a look-alike *blob* comparable to an oil blob, this script detects
the dominant dark patch in each Lookalike image itself (Gaussian-blurred,
low-percentile threshold, morphological open+close to suppress speckle) and
takes its single largest connected component. Because the image is a
confirmed look-alike (no real oil present, per the dataset label), that
detected dark patch is -- by construction -- a genuine look-alike blob, not
a guess. This mirrors exactly the confusion the model must resolve in
production: a naive dark-pixel detector would flag this same patch as
candidate oil.

Oil blobs come directly from the real annotated mask (all connected
components, since an image can contain multiple separate slicks). The same
shape/texture feature extraction lookalike.py uses at inference time
(lookalike.blob_features) is reused for both, so the only thing that changes
is the combination step: a fitted classifier instead of hand-picked weights.
Images (not blobs) are split train/test so blobs from the same scene never
leak across the split.

Usage:
    python train_lookalike_classifier.py [--root D:/oilspill_zenodo/extracted]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import cv2
import numpy as np
import tifffile
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

import lookalike

CLASSES = {"Oil": 0, "Lookalike": 1}  # "No oil" images contribute no blobs and are skipped


def _to_gray_uint8(tif_path: str) -> np.ndarray:
    """Dual-pol (VV, VH) Sigma0-dB float TIFF -> single-channel uint8.

    Takes the VV channel (index 0) and robust-normalizes via 1st/99th
    percentile clipping, matching how the rest of this project turns SAR
    intensity into a displayable/model-ready grayscale image.
    """
    arr = tifffile.imread(tif_path)
    if arr.ndim == 3:
        # Handle both (H, W, C) and (C, H, W) layouts.
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _load_mask_bin(tif_path: str) -> np.ndarray:
    arr = tifffile.imread(tif_path)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    return (arr > 0).astype(np.uint8)


def _dominant_dark_blob_mask(gray: np.ndarray, dark_percentile: float = 8.0) -> np.ndarray | None:
    """Detect the single largest dark patch in a look-alike image.

    Mirrors what a naive backscatter-threshold detector would flag as
    candidate oil: Gaussian blur (suppress SAR speckle) -> low-percentile
    darkness threshold -> morphological open+close (drop residual speckle,
    close small gaps) -> largest connected component. Returns None if
    nothing meeting MIN_BLOB_AREA is found.
    """
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh_val = np.percentile(blurred, dark_percentile)
    mask = (blurred <= thresh_val).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    if stats[best, cv2.CC_STAT_AREA] < lookalike.MIN_BLOB_AREA:
        return None
    return (labels == best).astype(np.uint8)


def _blob_row(blob_mask: np.ndarray, ctx: dict):
    """(norm_feats, area) for one blob mask, or None if it has no contour."""
    area = int(blob_mask.sum())
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    norm_feats, _ = lookalike.blob_features(blob_mask, contour, area, ctx)
    return norm_feats


def extract_dataset(root: str):
    """Return (X, y, groups): per-blob feature vectors, class labels, and a
    group id per source image (for a leakage-free train/test split).

    Oil: every connected component of the real annotated mask (an image can
    have several separate slicks). Lookalike: the single dominant dark patch
    detected in the image itself, since the provided mask is empty by
    definition for non-oil images -- see module docstring.
    """
    X, y, groups = [], [], []
    group_id = 0
    counts = {}

    for cls_name, cls_label in CLASSES.items():
        img_dir = os.path.join(root, "Images", cls_name)
        mask_dir = os.path.join(root, "Mask", cls_name)
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        counts[cls_name] = {"images": 0, "blobs": 0}

        for img_path in img_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            gray = _to_gray_uint8(img_path)
            ctx = lookalike.image_context(gray)

            blob_masks = []
            if cls_name == "Oil":
                mask_path = os.path.join(mask_dir, f"{stem}_segmentation.tif")
                if not os.path.exists(mask_path):
                    continue
                mask = _load_mask_bin(mask_path)
                if mask.sum() < lookalike.MIN_BLOB_AREA:
                    continue
                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                if n_labels > 1:
                    # One blob per image (the dominant/largest slick), matching
                    # the Lookalike side's methodology -- keeps classes balanced
                    # and avoids swamping the dataset with tiny mask fragments.
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    best = 1 + int(np.argmax(areas))
                    if areas[best - 1] >= lookalike.MIN_BLOB_AREA:
                        blob_masks.append((labels == best).astype(np.uint8))
            else:  # Lookalike
                blob_mask = _dominant_dark_blob_mask(gray)
                if blob_mask is not None:
                    blob_masks.append(blob_mask)

            image_had_blob = False
            for blob_mask in blob_masks:
                norm_feats = _blob_row(blob_mask, ctx)
                if norm_feats is None:
                    continue
                X.append([norm_feats[f] for f in lookalike.FEATURE_NAMES])
                y.append(cls_label)
                groups.append(group_id)
                counts[cls_name]["blobs"] += 1
                image_had_blob = True

            if image_had_blob:
                counts[cls_name]["images"] += 1
            group_id += 1

    return np.array(X), np.array(y), np.array(groups), counts


def _evaluate(name, preds, y_test):
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()
    print(f"{name}: accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} confusion_matrix={cm}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="D:/oilspill_zenodo/extracted")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    X, y, groups, counts = extract_dataset(args.root)
    print(f"Per-class images/blobs: {json.dumps(counts, indent=2)}")
    print(f"Total blobs: {len(X)}  Class balance (0=Oil,1=Lookalike): {np.bincount(y)}")
    if len(X) < 20:
        raise SystemExit(
            f"Only {len(X)} labeled blobs found under {args.root} -- check the extraction path/layout."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train blobs: {len(X_train)}  Test blobs: {len(X_test)} (split by source image, not by blob)")

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
        ),
    }
    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        results[name] = _evaluate(name, clf.predict(X_test), y_test)

    # Baseline: the original fixed-weight heuristic this classifier replaces.
    heuristic_scores = np.clip(
        X_test @ np.array([lookalike.FEATURE_WEIGHTS[n] for n in lookalike.FEATURE_NAMES]), 0.0, 1.0
    )
    heuristic_preds = (heuristic_scores >= lookalike.RISK_POSSIBLE_LOOKALIKE).astype(int)
    results["old_fixed_weight_heuristic"] = _evaluate(
        "old_fixed_weight_heuristic (baseline)", heuristic_preds, y_test
    )

    best_name = max(candidates, key=lambda n: results[n]["f1"])
    print(f"\nBest model: {best_name} (refitting on all {len(X)} blobs for the shipped classifier)")
    final_clf = candidates[best_name].__class__(**candidates[best_name].get_params())
    final_clf.fit(X, y)

    with open("lookalike_classifier.pkl", "wb") as f:
        pickle.dump(final_clf, f)
    with open("lookalike_classifier_metrics.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "held_out_test_blobs": int(len(X_test)),
                "held_out_test_images": int(len(set(groups[test_idx]))),
                "results": results,
                "dataset_source": "Zenodo 10.5281/zenodo.13761290 (Part III test split)",
            },
            f,
            indent=2,
        )
    print("Saved lookalike_classifier.pkl and lookalike_classifier_metrics.json")


if __name__ == "__main__":
    main()
