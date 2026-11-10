#!/usr/bin/env python3
"""Checks the hardware model against PyTorch on real images. Needs torch +
torchvision.

  python3 verify_pytorch.py CKPT.pth ../data/weights.bin VAL_DIR

quant_meta.json must sit next to weights.bin (run_checks.py writes both).

VAL_DIR has non-drones/ (class 0) and drones/ (class 1). Three models run on
every image, all fed the same 224x224 uint8 frame (img2bin.load_rgb):
  1. torchvision shufflenet_v2_x0_5 with the checkpoint, float32 and float64
     (the reference; BatchNorm unfolded, ImageNet normalisation in software).
  2. Folded float model: quantize.fold() weights, no BatchNorm, raw 0..255
     pixels in, stem padded with 255*mean. This is the network the hardware
     implements, before rounding. Must equal (1) to float64 precision.
  3. INT8 model read from weights.bin (ref_int8.py, bit-exact with the C++
     golden model). Compared with (1): argmax agreement, logit error,
     accuracy, and per-layer SQNR against the float tensors.
Writes the report to stdout and to ../data/pytorch_check.txt.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from img2bin import load_rgb  # noqa: E402
from model_def import MEAN, NUM_CLASSES, FloatModel, conv2d, params_from_state_dict, preprocess  # noqa: E402
from quantize import PAD_VALUE, fold, read_bin  # noqa: E402
from ref_int8 import Int8Model  # noqa: E402

DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CLASSES = ("non-drones", "drones")


class FoldedModel(FloatModel):
    """Float model with BatchNorm and input normalisation folded away."""

    def __init__(self, params, pad_value):
        super().__init__(params)
        self.wb = {n: fold(params, n) for n in self.specs if n != "fc"}
        self.pad = pad_value

    def input(self, img_u8):
        return img_u8.astype(np.float64)

    def layer(self, name, x, record=None, init_bn=False):
        s = self.specs[name]
        w, b = self.wb[name]
        groups = s["cin"] if s["kind"] == "dw" else 1
        y = conv2d(x, w, s["stride"], groups, self.pad if name == "conv1" else None) + b
        if s["relu"]:
            y = np.maximum(y, 0)
        if record is not None:
            record[name] = y
        return y


def sqnr(ref, test):
    return 10 * np.log10(np.mean(ref ** 2) / max(np.mean((ref - test) ** 2), 1e-30))


def main():
    ckpt, wbin, val = sys.argv[1:4]
    sd = torch.load(ckpt, map_location="cpu")
    net32 = torchvision.models.shufflenet_v2_x0_5(weights=None, num_classes=NUM_CLASSES)
    net32.load_state_dict(sd)
    net32.eval()
    net64 = torchvision.models.shufflenet_v2_x0_5(weights=None, num_classes=NUM_CLASSES)
    net64.load_state_dict(sd)
    net64.eval().double()

    params = params_from_state_dict({k: v.numpy() for k, v in sd.items()})
    unfolded = FloatModel(params)
    folded_exact = FoldedModel(params, 255.0 * MEAN)          # float pad: exact fold
    folded_hw = FoldedModel(params, PAD_VALUE.astype(np.float64))  # pad rounded as in hardware
    layers, pad, logit_scale = read_bin(wbin)
    int8 = Int8Model(layers, pad)
    with open(os.path.join(os.path.dirname(wbin), "quant_meta.json")) as f:
        scales = json.load(f)["activation_scales"]

    items = [(p, c) for c, cls in enumerate(CLASSES)
             for p in sorted(glob.glob(os.path.join(val, cls, "*")))]
    n = len(items)
    stats = dict(fold_logit=0.0, fold_layer=0.0, padround=0.0, t32=0.0)
    correct = dict(torch=0, int8=0)
    agree = 0
    flips = []
    err = []
    snr = {}
    for path, label in items:
        img = load_rgb(path)
        x = torch.tensor(preprocess(img)).permute(2, 0, 1)[None]
        with torch.no_grad():
            t64 = net64(x)[0].numpy()
            t32 = net32(x.float())[0].double().numpy()
        rec_u, rec_f = {}, {}
        unfolded.forward(img, record=rec_u)
        lf = folded_exact.forward(img, record=rec_f)
        lh = folded_hw.forward(img)
        lq, rec_q = int8.forward(img)
        lq = lq * logit_scale

        stats["fold_logit"] = max(stats["fold_logit"], np.abs(lf - t64).max())
        stats["fold_layer"] = max(stats["fold_layer"], max(
            np.abs(rec_f[k] - rec_u[k]).max() / max(np.abs(rec_u[k]).max(), 1e-12)
            for k in rec_u if k not in ("gap", "logits")))
        stats["padround"] = max(stats["padround"], np.abs(lh - t64).max())
        stats["t32"] = max(stats["t32"], np.abs(t32 - t64).max())

        correct["torch"] += int(np.argmax(t32) == label)
        correct["int8"] += int(np.argmax(lq) == label)
        same = np.argmax(lq) == np.argmax(t32)
        agree += int(same)
        if not same:
            flips.append(f"{os.path.basename(path)}: torch {np.round(t32, 3).tolist()} "
                         f"int8 {np.round(lq, 3).tolist()}")
        err.append((np.abs(lq - t32).max(), np.abs(t32[1] - t32[0])))
        # Per-layer SQNR: INT8 tensor dequantized with its scale vs float tensor.
        for k, v in rec_u.items():
            if k in rec_q and k in scales:
                snr.setdefault(k, []).append(sqnr(v, rec_q[k].astype(np.float64) * scales[k]))

    e = np.array(err)
    lines = [
        f"images: {n} ({', '.join(f'{c} {sum(1 for _, l in items if l == i)}' for i, c in enumerate(CLASSES))})",
        "",
        "BatchNorm / normalisation fold (float64):",
        f"  folded model vs torchvision, max logit difference: {stats['fold_logit']:.2e}",
        f"  folded vs unfolded, worst per-layer difference (relative to layer max): {stats['fold_layer']:.2e}",
        f"  with the stem pad rounded to {PAD_VALUE.tolist()} as in hardware: {stats['padround']:.2e}",
        f"  (torchvision float32 vs float64 for scale: {stats['t32']:.2e})",
        "",
        "INT8 (weights.bin) vs torchvision float32:",
        f"  argmax agreement: {agree}/{n}",
        f"  accuracy: torch {correct['torch']}/{n}, int8 {correct['int8']}/{n}",
        f"  logit error: mean {e[:, 0].mean():.3f}, max {e[:, 0].max():.3f} "
        f"(mean |logit margin| {e[:, 1].mean():.3f})",
    ]
    worst = sorted((np.mean(v), k) for k, v in snr.items())
    lines.append("  per-layer SQNR vs float (dB), mean over images: lowest "
                 + ", ".join(f"{k} {v:.1f}" for v, k in worst[:5])
                 + f"; conv1 {np.mean(snr['conv1']):.1f}, conv5 {np.mean(snr['conv5']):.1f}")
    if flips:
        lines.append("  argmax flips:")
        lines += ["    " + f for f in flips]
    print("\n".join(lines))
    with open(os.path.join(DATA, "pytorch_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
