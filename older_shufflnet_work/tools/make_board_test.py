#!/usr/bin/env python3
"""Labeled 3-class board test set: data/board_test/<tag>/{input.bin,logits.bin,label.txt}.

  python3 tools/make_board_test.py            # from new_accelrator_3class/, after data/weights.bin
  python3 tools/make_board_test.py --src data/clean_v3_iou_qat/board_test

Input (exported from pglab with the QAT weights): <src>/<tag>.bin, already
preprocessed like the board input (224x224 squash, RGB uint8 HWC, 150,528
bytes), and <src>/manifest.jsonl with tag, label (0 other, 1 bird, 2 drone),
set, source, and optionally fp32_pred / int8_pred.
Tag = class letter (O/B/D) + set letter (T test, C CARS, H Halmstad) + digits;
the board app (sw/src/main.c) takes the true class and set from these two letters.
Logits: NumPy INT8 reference (golden/ref_int8.py) from data/weights.bin, written
as NUM_CLASSES x int32 LE. Prints the accuracy the board must reproduce
(bit-exact logits give the same argmax) and writes data/board_test/summary.json.
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "golden"))
from model_def import NUM_CLASSES  # noqa: E402
from quantize import read_bin  # noqa: E402
from ref_int8 import Int8Model  # noqa: E402

OUT = os.path.join(ROOT, "data", "board_test")
CLASS_NAMES = ("other", "bird", "drone")
CLASS_LETTER = "OBD"
SET_NAMES = {"T": "test", "C": "CARS", "H": "Halmstad"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "clean_v3_iou_qat", "board_test"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(args.src, "manifest.jsonl")) if l.strip()]
    layers, pad, _ = read_bin(os.path.join(ROOT, "data", "weights.bin"))
    im = Int8Model(layers, pad)
    os.makedirs(OUT, exist_ok=True)
    summary, conf = [], np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    per_set = {}
    for r in sorted(rows, key=lambda r: r["tag"]):
        tag, label = r["tag"], int(r["label"])
        assert len(tag) <= 8 and tag.isalnum() and tag[0] == CLASS_LETTER[label] \
            and tag[1] in SET_NAMES, f"bad tag {tag} for label {label}"
        img = np.fromfile(os.path.join(args.src, tag + ".bin"), np.uint8)
        assert img.size == 224 * 224 * 3, f"{tag}: {img.size} bytes"
        logits, _ = im.forward(img.reshape(224, 224, 3))
        assert len(logits) == NUM_CLASSES
        pred = int(np.argmax(logits))
        d = os.path.join(OUT, tag)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "input.bin"), "wb").write(img.tobytes())
        open(os.path.join(d, "logits.bin"), "wb").write(np.asarray(logits, "<i4").tobytes())
        open(os.path.join(d, "label.txt"), "w").write(f"{label} {r['set']} {r.get('source', '-')}\n")
        conf[label, pred] += 1
        s = per_set.setdefault(tag[1], [0, 0])
        s[0] += pred == label
        s[1] += 1
        summary.append(dict(tag=tag, label=label, set=r["set"], logits=[int(v) for v in logits],
                            pred=pred, int8_pred=r.get("int8_pred"), fp32_pred=r.get("fp32_pred")))
    n = len(summary)
    acc = int(np.trace(conf))
    mism = [s["tag"] for s in summary if s["int8_pred"] is not None and int(s["int8_pred"]) != s["pred"]]
    json.dump(dict(confusion=conf.tolist(), accuracy=acc / n, per_set=per_set,
                   int8_pred_mismatch=mism, images=summary),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"wrote {n} labeled images to {OUT}")
    print(f"accuracy {acc}/{n} = {100 * acc / n:.1f}%")
    for k, (c, t) in sorted(per_set.items()):
        print(f"  {SET_NAMES[k]:9s} {c}/{t} = {100 * c / t:.1f}%")
    print("confusion (rows true, cols pred: other bird drone)")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:6s} {conf[i].tolist()}  recall {100 * conf[i, i] / max(conf[i].sum(), 1):.1f}%")
    print(f"pglab int8_pred differs from PC golden argmax on {len(mism)} images {mism[:10]}")


if __name__ == "__main__":
    main()
