#!/usr/bin/env python3
"""Golden test set for the board: data/golden/<D|N|R>NNN/{input.bin,logits.bin,label.txt}.

  python3 tools/make_golden_set.py            # from new_accelrator_3class/, after data/weights.bin
  python3 tools/make_golden_set.py --refresh  # keep every input.bin, recompute logits.bin only

Images: 25 per (CARS / normal test) x (drone / non-drone) from
data/ayush_mixed_cars (board 16:9 squash to 224x224, test images the model never
trained on), plus 4 stress frames R001-R004 (all 0, all 255, two seeded noise
frames). Logits: NumPy INT8 reference (golden/ref_int8.py) from data/weights.bin,
NUM_CLASSES x int32 LE in class order (0 other, 1 bird, 2 drone).
label.txt keeps the source dataset's 2-class label (1 drone, 0 non-drone).
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "golden"))
from model_def import NUM_CLASSES  # noqa: E402
from quantize import read_bin  # noqa: E402
from ref_int8 import Int8Model  # noqa: E402

SRC = os.path.join(ROOT, "data", "ayush_mixed_cars")
OUT = os.path.join(ROOT, "data", "golden")
CLASS_NAMES = ("other", "bird", "drone")


def stress_frames():
    rng = np.random.default_rng(7)
    yield "R001", np.zeros((224, 224, 3), np.uint8), "- stress all-0"
    yield "R002", np.full((224, 224, 3), 255, np.uint8), "- stress all-255"
    yield "R003", rng.integers(0, 256, (224, 224, 3)).astype(np.uint8), "- stress noise"
    yield "R004", (rng.integers(0, 2, (224, 224, 3)) * 255).astype(np.uint8), "- stress binary-noise"


def new_frames():
    from img2bin import load_rgb
    m = json.load(open(os.path.join(SRC, "manifest.json")))
    count = {0: 0, 1: 0}
    for label in (1, 0):
        for st in ("cars", "normal"):
            rows = [r for r in m if r["set"] == st and r["label"] == label][::20][:25]
            for r in rows:
                count[label] += 1
                name = f"{'D' if label else 'N'}{count[label]:03d}"
                yield name, load_rgb(os.path.join(SRC, r["file"])), \
                    f"{label} {st} {r['source']} {r['file']}"


def existing_frames():
    for name in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, name, "input.bin")
        if os.path.isfile(p):
            lab = os.path.join(OUT, name, "label.txt")
            info = open(lab).read().strip() if os.path.exists(lab) else "-"
            yield name, np.fromfile(p, np.uint8).reshape(224, 224, 3), info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true",
                    help="reuse the input.bin files already in data/golden")
    args = ap.parse_args()
    layers, pad, _ = read_bin(os.path.join(ROOT, "data", "weights.bin"))
    im = Int8Model(layers, pad)
    frames = list(existing_frames()) if args.refresh else list(new_frames())
    have = {n for n, _, _ in frames}
    frames += [f for f in stress_frames() if f[0] not in have]
    summary = []
    for name, img, info in frames:
        logits, _ = im.forward(img)
        assert len(logits) == NUM_CLASSES
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "input.bin"), "wb").write(img.astype(np.uint8).tobytes())
        open(os.path.join(d, "logits.bin"), "wb").write(np.asarray(logits, "<i4").tobytes())
        open(os.path.join(d, "label.txt"), "w").write(info + "\n")
        parts = info.split()
        pred = int(np.argmax(logits))
        summary.append(dict(name=name, label=parts[0], info=" ".join(parts[1:3]),
                            logits=[int(v) for v in logits], pred=pred,
                            pred_name=CLASS_NAMES[pred]))
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    preds = {c: sum(s["pred"] == i for s in summary) for i, c in enumerate(CLASS_NAMES)}
    print(f"wrote {len(summary)} golden images to {OUT}, argmax counts {preds}")


if __name__ == "__main__":
    main()
