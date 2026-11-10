#!/usr/bin/env python3
"""Convert an image to the accelerator's input format: 224x224x3 uint8, HWC, RGB.

  python3 img2bin.py photo.jpg out.bin            # plain resize to 224x224
  python3 img2bin.py photo.jpg out.bin --crop     # resize short side 256, centre crop

Which of the two matches training must be confirmed by the model team (task M1).
"""
import argparse

import numpy as np
from PIL import Image

SIZE = 224


def load_rgb(path, crop=False):
    img = Image.open(path).convert("RGB")
    if crop:
        w, h = img.size
        s = 256 / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
        w, h = img.size
        left, top = (w - SIZE) // 2, (h - SIZE) // 2
        img = img.crop((left, top, left + SIZE, top + SIZE))
    else:
        img = img.resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--crop", action="store_true")
    a = ap.parse_args()
    load_rgb(a.image, a.crop).tofile(a.out)


if __name__ == "__main__":
    main()
