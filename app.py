import base64
import io
import json
import os
import tempfile

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from dataset_utils import IMG_SIZE
from geo_utils import image_footprint, read_geotiff
from lookalike import analyze_blobs, colorize_mask
from model import UNet

GEOTIFF_EXTENSIONS = (".tif", ".tiff")

# Every image in this project's training data (PALSAR + Sentinel-1) is a
# single SAR backscatter channel replicated across R/G/B -- i.e. exactly
# zero color saturation (verified on the real dataset: mean saturation is
# 0.0 for every sample checked). A real photo (a person, a colored surface)
# carries meaningful saturation even for a plain skin tone (~100/255) or a
# muted color (~30-50/255), so this is a cheap, reliable gate against the
# segmentation model confidently hallucinating "oil" on inputs it was never
# trained to recognize, let alone reject.
SAR_SATURATION_LIMIT = 8.0

app = Flask(__name__)

DEVICE = torch.device("cpu")
model = UNet(in_ch=3, out_ch=1, base=16).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

THRESHOLD = 0.5
if os.path.exists("threshold.json"):
    with open("threshold.json") as f:
        THRESHOLD = json.load(f)["threshold"]


def sar_plausibility_error(pil_img: Image.Image) -> str | None:
    """None if the image is plausibly grayscale SAR data; an error message otherwise."""
    hsv = np.array(pil_img.convert("RGB").convert("HSV"))
    mean_saturation = float(hsv[..., 1].mean())
    if mean_saturation > SAR_SATURATION_LIMIT:
        return (
            "This doesn't look like SAR radar imagery (detected color content -- "
            f"average saturation {mean_saturation:.0f}/255, real SAR chips are ~0). "
            "This model only understands grayscale Sentinel-1/PALSAR backscatter, "
            "where oil shows up as dark patches in radar reflectivity -- it can't "
            "meaningfully classify photos, so it won't guess."
        )
    return None


def predict_mask(pil_img: Image.Image) -> Image.Image:
    orig_size = pil_img.size  # (w, h)
    resized = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].numpy()

    mask = (probs >= THRESHOLD).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L").resize(orig_size, Image.NEAREST)
    return mask_img, float((mask > 0).mean())


def overlay_mask(pil_img: Image.Image, labels: np.ndarray, blobs: list) -> Image.Image:
    """Colors each detected blob by its look-alike-risk bucket: green = likely
    oil, amber = uncertain, red = possible look-alike (see lookalike.py)."""
    base = pil_img.convert("RGBA")
    color_layer = Image.fromarray(colorize_mask(labels, blobs), mode="RGBA")
    return Image.alpha_composite(base, color_layer)


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def mask_centroid(mask_img: Image.Image):
    """Pixel (col, row) centroid of the detected spill region, or None if empty."""
    arr = np.asarray(mask_img)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = (file.filename or "").lower()
    geo_info = {"has_geo": False, "pixel_to_latlon": None}
    downsample_factor = 1

    try:
        if filename.endswith(GEOTIFF_EXTENSIONS):
            # Real satellite GeoTIFFs can be hundreds of MB to a few GB, so
            # geo_utils memory-maps from disk rather than an in-memory stream.
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            try:
                file.save(tmp.name)
                tmp.close()
                rgb_arr, geo_info, downsample_factor = read_geotiff(tmp.name)
            finally:
                os.remove(tmp.name)
            pil_img = Image.fromarray(rgb_arr, mode="RGB")
        else:
            pil_img = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    ood_error = sar_plausibility_error(pil_img)
    if ood_error:
        return jsonify({"error": ood_error}), 422

    mask_img, spill_ratio = predict_mask(pil_img)

    # mask_img is already resized to pil_img's size in predict_mask, so these
    # arrays are pixel-aligned.
    mask_arr = np.asarray(mask_img)
    gray_arr = np.asarray(pil_img.convert("L"))
    blobs, labels = analyze_blobs(mask_arr, gray_arr)
    overlay_img = overlay_mask(pil_img, labels, blobs)

    location = None
    footprint = None
    if geo_info["has_geo"]:
        centroid = mask_centroid(mask_img)
        if centroid is not None:
            col, row = centroid
            lat, lon = geo_info["pixel_to_latlon"](col * downsample_factor, row * downsample_factor)
            location = {"lat": lat, "lon": lon}
        footprint = image_footprint(geo_info)

    return jsonify(
        {
            "original": img_to_b64(pil_img.convert("RGB")),
            "mask": img_to_b64(mask_img),
            "overlay": img_to_b64(overlay_img.convert("RGB")),
            "spill_percent": round(spill_ratio * 100, 2),
            "threshold": THRESHOLD,
            "location": location,
            "footprint": footprint,
            "blobs": blobs,
        }
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
