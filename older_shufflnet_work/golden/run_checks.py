#!/usr/bin/env python3
"""Build the INT8 model, produce golden vectors, and check everything.

  python3 run_checks.py                         # synthetic weights (until the checkpoint arrives)
  python3 run_checks.py --npz shufflenet_2class.npz --calib DATASET/train

Steps:
  1. Float weights: from --npz (see export_npz.py) or synthetic.
  2. Calibrate on --calib (default images/), plus flips, quantize, write
     ../data/weights.bin and ../data/quant_meta.json.
  3. For every test image: NumPy INT8 reference writes all tensors to
     ../data/golden/<image>/; the C++ golden model runs on the same input and
     every tensor must match byte for byte.
  4. Float vs INT8: per-layer signal-to-quantization-noise and logit agreement.
Exit code 0 only if every C++ tensor matches the NumPy reference.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from img2bin import load_rgb  # noqa: E402
from model_def import (FloatModel, maxpool3x3s2, params_from_state_dict,  # noqa: E402
                       synthetic_params, units)
from quantize import calibrate, quantize, read_bin, write_bin, write_meta  # noqa: E402
from ref_int8 import Int8Model  # noqa: E402

DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


def dtype_of(name):
    if name in ("gap", "logits"):
        return "<i4"
    if name.endswith("dw"):
        return "i1"
    return "u1"


def sqnr(ref, test):
    return 10 * np.log10(np.mean(ref ** 2) / max(np.mean((ref - test) ** 2), 1e-30))


def float_noise_baseline(fmodel, img, conv1_sqnr, seed=1):
    """conv5 SQNR when the float model alone gets noise of conv1_sqnr dB at
    conv1. Tells quantization damage apart from the network amplifying noise."""
    rec = {}
    fmodel.forward(img, record=rec)
    c1 = rec["conv1"]
    rng = np.random.default_rng(seed)
    noisy = np.maximum(c1 + rng.normal(0, np.sqrt(np.mean(c1 ** 2)) / 10 ** (conv1_sqnr / 20),
                                       c1.shape), 0)

    def tail(x):
        x = maxpool3x3s2(x)
        for prefix, down in units():
            x = fmodel.unit(prefix, down, x)
        return fmodel.layer("conv5", x)

    return sqnr(tail(c1), tail(noisy))


def calib_set(images):
    out = []
    for img in images:
        out += [img, img[:, ::-1], img[::-1]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="torchvision-style state_dict as .npz")
    ap.add_argument("--images", default=os.path.join(HERE, "images"))
    ap.add_argument("--calib", help="calibration image folder, searched recursively "
                    "(default: the test images)")
    ap.add_argument("--crop", action="store_true", help="resize 256 + centre crop")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.images, "*")))
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    images = [load_rgb(p, a.crop) for p in paths]
    print(f"{len(images)} test images from {a.images}")

    if a.npz:
        params = params_from_state_dict(dict(np.load(a.npz)))
        print(f"weights: {a.npz}")
    else:
        params = synthetic_params(images, seed=a.seed)
        print(f"weights: SYNTHETIC (seed {a.seed}); logits are meaningless until the "
              "trained checkpoint arrives")

    if a.calib:
        cpaths = sorted(p for p in glob.glob(os.path.join(a.calib, "**", "*"), recursive=True)
                        if os.path.isfile(p))
        calib = [load_rgb(p, a.crop) for p in cpaths]
        print(f"calibrating on {len(calib)} images from {a.calib} (plus flips)")
    else:
        calib = images
    sc = calibrate(params, calib_set(calib))
    qlayers, logit_scale = quantize(params, sc)
    os.makedirs(DATA, exist_ok=True)
    build = tempfile.mkdtemp(prefix="golden_build_")

    wpath = os.path.join(DATA, "weights.bin")
    write_bin(wpath, qlayers, logit_scale)
    write_meta(os.path.join(DATA, "quant_meta.json"), qlayers, sc, logit_scale)
    print(f"wrote {wpath} ({os.path.getsize(wpath)} bytes)")

    wpath_local = os.path.join(build, "weights.bin")
    write_bin(wpath_local, qlayers, logit_scale)
    layers, pad, logit_scale = read_bin(wpath_local)
    imodel = Int8Model(layers, pad)
    fmodel = FloatModel(params)

    exe = os.path.join(build, "golden_model")
    src = os.path.join(HERE, "c_model")
    subprocess.run(["g++", "-std=c++11", "-O2", "-Wall", "-Wextra", "-o", exe,
                    os.path.join(src, "golden_main.cpp"), os.path.join(src, "golden_model.cpp")],
                   check=True)

    mismatches = 0
    snr = {}
    agree = 0
    report = []
    for name, img in zip(names, images):
        logits_q, rec = imodel.forward(img)
        blobs = {t: np.asarray(v).astype(dtype_of(t)).tobytes() for t, v in rec.items()}

        # Compare in a local temp dir: the shared folder may lag behind writes.
        with tempfile.TemporaryDirectory() as cdir:
            inp = os.path.join(cdir, "input_image.bin")
            with open(inp, "wb") as f:
                f.write(blobs["input"])
            r = subprocess.run([exe, wpath_local, inp, cdir], capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout, r.stderr)
                mismatches += 1
            bad = []
            for t, ref in blobs.items():
                cp = os.path.join(cdir, f"{t}.bin")
                got = open(cp, "rb").read() if os.path.exists(cp) else b""
                if ref != got:
                    bad.append(t)
            mismatches += len(bad)

        gdir = os.path.join(DATA, "golden", name)
        os.makedirs(gdir, exist_ok=True)
        for t, b in blobs.items():
            with open(os.path.join(gdir, f"{t}.bin"), "wb") as f:
                f.write(b)

        frec = {}
        logits_f = fmodel.forward(img, record=frec)
        for t, v in frec.items():
            if t in rec and t not in ("gap", "logits") and t in sc:
                deq = rec[t].astype(np.float64) * sc[t]
                snr.setdefault(t, []).append(sqnr(v, deq))
        same = int(np.argmax(logits_q) == np.argmax(logits_f))
        agree += same
        line = (f"{name:<22} C vs NumPy: {'MATCH' if not bad else 'MISMATCH ' + ','.join(bad[:5])}"
                f" | int8 logits {logits_q.tolist()} -> {(logits_q * logit_scale).round(3).tolist()}"
                f" | float {np.round(logits_f, 3).tolist()} | argmax same: {bool(same)}")
        print(line)
        report.append(line)

    worst = sorted(((np.mean(v), k) for k, v in snr.items()))[:5]
    report.append("")
    report.append(f"argmax agreement float vs int8: {agree}/{len(images)}")
    report.append("lowest per-layer SQNR (dB): " + ", ".join(f"{k} {v:.1f}" for v, k in worst))
    base = float_noise_baseline(fmodel, images[0], np.mean(snr["conv1"]))
    report.append(f"SQNR at conv1 {np.mean(snr['conv1']):.1f} dB, at conv5 {np.mean(snr['conv5']):.1f} dB; "
                  f"float model with the same noise injected at conv1 gives {base:.1f} dB at conv5")
    if not a.npz:
        report.append("NOTE: synthetic weights. Random networks amplify any small noise, so the "
                      "float-vs-INT8 numbers above say nothing about real accuracy; rerun with --npz.")
    report.append("RESULT: " + ("ALL TENSORS BIT-EXACT" if mismatches == 0 else
                                f"{mismatches} MISMATCHES"))
    print("\n".join(report[len(images):]))
    with open(os.path.join(DATA, "golden", "report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
