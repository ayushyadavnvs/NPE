#!/usr/bin/env python3
"""Builds sd_card/: the files the KV260 test app (sw/src/main.c) reads.

  python3 tools/make_sd_card.py            # from new_accelrator_3class/

sd_card/WEIGHTS.BIN        = data/unit_weights.bin
sd_card/IMAGES/<tag>.BIN   = data/board_test/<tag>/input.bin  (labeled, tools/make_board_test.py)
                           + data/golden/R00N/input.bin       (stress frames, unlabelled)
                             224x224x3 RGB, HWC
sd_card/EXPECT/<tag>.BIN   = logits.bin of the same folder (3 x int32 LE: other, bird, drone)
sd_card/NAMES.TXT          tag, folder under data/, expected logits, true label / set / source
Board test tags: class letter O/B/D + set letter T/C/H + digits; the app takes
the true class and set from them (R = unlabelled, bit-exact check only).
8.3 file names, because the app's FatFs has no long names.
"""
import os
import shutil
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "sd_card")


def frames():
    bt = os.path.join(DATA, "board_test")
    for name in sorted(os.listdir(bt)):
        if os.path.isdir(os.path.join(bt, name)):
            yield name, "board_test/" + name
    for name in sorted(os.listdir(os.path.join(DATA, "golden"))):
        if name.startswith("R") and os.path.isdir(os.path.join(DATA, "golden", name)):
            yield name, "golden/" + name


def main():
    for sub in ("IMAGES", "EXPECT"):
        shutil.rmtree(os.path.join(OUT, sub), ignore_errors=True)
        os.makedirs(os.path.join(OUT, sub))
    shutil.copyfile(os.path.join(DATA, "unit_weights.bin"), os.path.join(OUT, "WEIGHTS.BIN"))
    lines = ["# tag   folder               expected logit_other logit_bird logit_drone  label set source"]
    n = 0
    for tag, folder in frames():
        src = os.path.join(DATA, folder)
        assert len(tag) <= 8 and tag.isalnum(), f"{tag}: not an 8.3 name"
        assert os.path.getsize(os.path.join(src, "input.bin")) == 224 * 224 * 3
        shutil.copyfile(os.path.join(src, "input.bin"), os.path.join(OUT, "IMAGES", tag + ".BIN"))
        logits = open(os.path.join(src, "logits.bin"), "rb").read()
        assert len(logits) == 12
        open(os.path.join(OUT, "EXPECT", tag + ".BIN"), "wb").write(logits)
        other, bird, drone = struct.unpack("<iii", logits)
        info = open(os.path.join(src, "label.txt")).read().split() if os.path.exists(os.path.join(src, "label.txt")) else ["-"]
        lines.append(f"{tag:6s}  {folder:20s} {other} {bird} {drone}  {' '.join(info[:3])}")
        n += 1
    assert n <= 512, "the app reads at most MAX_IMAGES = 512 images"
    open(os.path.join(OUT, "NAMES.TXT"), "w").write("\n".join(lines) + "\n")
    print(f"wrote {OUT}: WEIGHTS.BIN, {n} images")


if __name__ == "__main__":
    main()
