"""Look-alike risk scoring for detected oil blobs.

Originally this scored each connected component of the model's binary mask
using fixed, hand-picked weights over classic shape/texture discriminators
from the pre-deep-learning SAR oil-spill literature (Solberg et al.,
Topouzelis surveys) -- real oil slicks tend to be elongated/filamentary with
irregular (low-solidity) boundaries, sharp edges, and a homogeneous dark
interior, while common look-alikes (low-wind zones, biogenic slicks, rain
cells) tend to be more compact/rounded with diffuse edges and patchier
interiors. That was a deliberate stand-in because no labeled look-alike data
existed anywhere in this project.

Real labeled look-alike examples (Zenodo 10.5281/zenodo.13761290, MKLab-style
Lookalike/Oil/No-oil classes) are now used to train a real classifier -- see
``train_lookalike_classifier.py``. ``analyze_blobs`` loads the trained model
(``lookalike_classifier.pkl``) when present and falls back to the original
fixed-weight heuristic otherwise, so the app keeps working either way.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

MIN_BLOB_AREA = 20  # px; smaller components are treated as noise, not a blob

# Feature order the trained classifier expects (see train_lookalike_classifier.py).
FEATURE_NAMES = [
    "circularity",
    "low_elongation",
    "high_solidity",
    "large_area_frac",
    "diffuse_edge",
    "patchy_interior",
]

# Fallback: fixed weights combining normalized [0,1] "look-alike-leaning"
# features into a single risk score. Used only when no trained classifier is
# available. See module docstring for the rationale behind each.
FEATURE_WEIGHTS = {
    "circularity": 0.25,      # compact/round shape -> look-alike
    "low_elongation": 0.15,   # not streak-like -> look-alike
    "high_solidity": 0.20,    # smooth, non-jagged boundary -> look-alike
    "large_area_frac": 0.15,  # covers a big chunk of the scene -> look-alike
    "diffuse_edge": 0.15,     # soft boundary -> look-alike
    "patchy_interior": 0.10,  # non-homogeneous interior -> look-alike
}

RISK_LIKELY_OIL = 0.35
RISK_POSSIBLE_LOOKALIKE = 0.60

BLOB_COLORS = {
    "likely_oil": (61, 220, 151),
    "uncertain": (255, 180, 84),
    "possible_look_alike": (255, 92, 92),
}

_CLASSIFIER_PATH = Path(__file__).with_name("lookalike_classifier.pkl")
_classifier = None
_classifier_loaded = False


def _get_classifier():
    """Lazily load the trained classifier, if one has been trained."""
    global _classifier, _classifier_loaded
    if not _classifier_loaded:
        _classifier_loaded = True
        if _CLASSIFIER_PATH.exists():
            with open(_CLASSIFIER_PATH, "rb") as fh:
                _classifier = pickle.load(fh)
    return _classifier


def _normalize(value, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _connected_components(mask_arr):
    mask_bin = (mask_arr > 0).astype(np.uint8)
    return cv2.connectedComponentsWithStats(mask_bin, connectivity=8)


def _boundary_ring(blob_mask):
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(blob_mask, kernel, iterations=1).astype(bool)
    eroded = cv2.erode(blob_mask, kernel, iterations=1).astype(bool)
    return dilated & ~eroded


def _reasons(feats):
    reasons = []
    if feats["circularity"] > 0.6:
        reasons.append("compact/rounded shape")
    if feats["elongation"] < 2.0:
        reasons.append("not elongated/streak-like")
    if feats["solidity"] > 0.9:
        reasons.append("smooth, non-jagged boundary")
    if feats["area_percent"] > 25:
        reasons.append(f"covers {feats['area_percent']:.0f}% of the scene")
    if feats["edge_sharpness_norm"] < 0.35:
        reasons.append("diffuse/soft boundary")
    if feats["interior_homogeneity"] < 0.5:
        reasons.append("patchy interior texture")
    if not reasons:
        reasons.append("elongated, sharp-edged, homogeneous — typical oil signature")
    return reasons[:2]


def image_context(gray_arr: np.ndarray) -> dict:
    """Precompute the whole-image stats blob features are normalized against."""
    gray = gray_arr.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    return {
        "gray": gray,
        "whole_image_std": float(gray.std()),
        "grad_mag": grad_mag,
        "image_grad_scale": float(grad_mag.mean()) + 1e-6,
        "total_px": gray.shape[0] * gray.shape[1],
    }


def blob_features(blob_mask: np.ndarray, contour: np.ndarray, area: int, ctx: dict):
    """Compute the normalized [0,1] look-alike-leaning features for one blob.

    Returns (norm_feats, display_feats): ``norm_feats`` keys match
    FEATURE_NAMES (classifier input order); ``display_feats`` carries the
    unnormalized values used to build human-readable reasons.
    """
    gray, whole_image_std = ctx["gray"], ctx["whole_image_std"]
    grad_mag, image_grad_scale, total_px = ctx["grad_mag"], ctx["image_grad_scale"], ctx["total_px"]

    perimeter = cv2.arcLength(contour, closed=True)
    circularity = _normalize((4 * np.pi * area) / (perimeter ** 2), 0.0, 1.0) if perimeter > 0 else 0.0

    (rw, rh) = cv2.minAreaRect(contour)[1]
    elongation = max(rw, rh) / max(min(rw, rh), 1e-6)
    low_elongation = 1.0 - _normalize(elongation, 1.0, 5.0)

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 1.0
    high_solidity = _normalize(solidity, 0.5, 1.0)

    area_percent = 100.0 * area / total_px
    large_area_frac = _normalize(area_percent, 5.0, 40.0)

    ring = _boundary_ring(blob_mask)
    edge_sharpness_norm = (
        _normalize(float(grad_mag[ring].mean()) / image_grad_scale, 0.0, 3.0) if ring.any() else 0.0
    )
    diffuse_edge = 1.0 - edge_sharpness_norm

    interior = cv2.erode(blob_mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    if interior.sum() < 4:
        interior = blob_mask.astype(bool)
    interior_std = float(gray[interior].std())
    interior_homogeneity = (
        float(np.clip(1.0 - interior_std / whole_image_std, 0.0, 1.0)) if whole_image_std > 0 else 1.0
    )
    patchy_interior = 1.0 - interior_homogeneity

    norm_feats = {
        "circularity": circularity,
        "low_elongation": low_elongation,
        "high_solidity": high_solidity,
        "large_area_frac": large_area_frac,
        "diffuse_edge": diffuse_edge,
        "patchy_interior": patchy_interior,
    }
    display_feats = {
        "circularity": circularity,
        "elongation": elongation,
        "solidity": solidity,
        "area_percent": area_percent,
        "edge_sharpness_norm": edge_sharpness_norm,
        "interior_homogeneity": interior_homogeneity,
    }
    return norm_feats, display_feats


def _score_blob(norm_feats: dict) -> tuple[float, str]:
    """Return (risk in [0,1], label) for one blob's normalized features.

    Uses the trained classifier (lookalike_classifier.pkl) when available;
    otherwise falls back to the fixed-weight heuristic.
    """
    clf = _get_classifier()
    vector = [norm_feats[name] for name in FEATURE_NAMES]

    if clf is not None:
        risk = float(clf.predict_proba([vector])[0][1])
    else:
        risk = float(np.clip(sum(FEATURE_WEIGHTS[name] * norm_feats[name] for name in FEATURE_NAMES), 0.0, 1.0))

    if risk < RISK_LIKELY_OIL:
        label = "likely_oil"
    elif risk < RISK_POSSIBLE_LOOKALIKE:
        label = "uncertain"
    else:
        label = "possible_look_alike"
    return risk, label


def analyze_blobs(mask_arr: np.ndarray, gray_arr: np.ndarray):
    """Score each connected component of a binary oil mask for look-alike risk.

    mask_arr: 2D array, nonzero = predicted oil pixel.
    gray_arr: 2D uint8 grayscale array, same shape as mask_arr.

    Returns (blobs, labels) where blobs is a list of per-blob dicts (largest
    area first) and labels is the int32 connected-component label array
    (same shape as mask_arr, 0 = background), useful for colorizing an
    overlay consistently with the returned blobs' "id" field.
    """
    n_labels, labels, stats, centroids = _connected_components(mask_arr)
    ctx = image_context(gray_arr)

    blobs = []
    for i in range(1, n_labels):  # label 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_BLOB_AREA:
            continue

        blob_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        norm_feats, feats = blob_features(blob_mask, contour, area, ctx)
        risk, label = _score_blob(norm_feats)

        x, y, w, h = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        cx, cy = centroids[i]

        blobs.append({
            "id": int(i),
            "bbox": [x, y, w, h],
            "centroid": [float(cx), float(cy)],
            "area_px": area,
            "area_percent": round(feats["area_percent"], 2),
            "look_alike_risk": round(risk, 3),
            "label": label,
            "reasons": _reasons(feats),
        })

    blobs.sort(key=lambda b: b["area_px"], reverse=True)
    return blobs, labels


def colorize_mask(labels: np.ndarray, blobs: list) -> np.ndarray:
    """Build an RGBA array coloring each blob by its look-alike-risk bucket."""
    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for blob in blobs:
        color = BLOB_COLORS[blob["label"]]
        m = labels == blob["id"]
        rgba[m, 0] = color[0]
        rgba[m, 1] = color[1]
        rgba[m, 2] = color[2]
        rgba[m, 3] = 130
    return rgba
