import os
import random
from glob import glob

import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

IMG_SIZE = 256


class OilSpillDataset(Dataset):
    """Pairs image/label PNGs from one or more sensor folders (palsar, sentinel)."""

    def __init__(self, root, split, sensors=("palsar", "sentinel"), img_size=IMG_SIZE, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.pairs = []
        for sensor in sensors:
            img_dir = os.path.join(root, split, sensor, "image")
            lbl_dir = os.path.join(root, split, sensor, "label")
            for img_path in sorted(glob(os.path.join(img_dir, "*.png"))):
                name = os.path.basename(img_path)
                lbl_path = os.path.join(lbl_dir, name)
                if os.path.exists(lbl_path):
                    self.pairs.append((img_path, lbl_path))

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img, lbl):
        # Flips and 90-degree rotations are lossless and valid for SAR imagery,
        # which has no fixed orientation (ocean scenes, no "up").
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            lbl = lbl.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            k = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
            img = img.transpose(k)
            lbl = lbl.transpose(k)

        # Mild brightness/contrast jitter simulates varying backscatter conditions.
        # Applied to the image only -- the mask is unaffected by intensity changes.
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))

        return img, lbl

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        lbl = Image.open(lbl_path).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST
        )

        if self.augment:
            img, lbl = self._augment(img, lbl)

        img_arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        lbl_arr = (np.asarray(lbl, dtype=np.float32) > 127).astype(np.float32)[None, :, :]

        return img_arr, lbl_arr
