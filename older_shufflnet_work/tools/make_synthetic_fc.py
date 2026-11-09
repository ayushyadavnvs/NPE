#!/usr/bin/env python3
"""Stand-in 3-class weights until a trained 3-class model exists.

  python3 tools/make_synthetic_fc.py      # from new_accelrator_3class/

Reads the trained 2-class board model data/weights_2class.bin (copied from
new_accelrator/data/weights.bin) and writes data/weights.bin with the same
backbone and a 3-class fc (class order 0 other, 1 bird, 2 drone):
  row 0 other  synthetic: seeded random INT8 over the full [-127, 127] range
               and a fixed bias of OTHER_BIAS, chosen so that "other" wins the
               argmax on part of the golden set (all three outputs get picked);
  row 1 bird   the trained class-0 row (non-drone) and its bias;
  row 2 drone  the trained class-1 row (drone) and its bias.
So the bird / drone logits equal the 2-class model's logits exactly (an extra
cross-check against new_accelrator), and the other logit is meaningless.
Only bit-exactness is tested with these weights, not accuracy.

Worst-case |logit| = |bias| + 127 * 1024 * 49 * 255 < 1.63e9 + 2^21 < 2^31,
so no fc weight or input can overflow the INT32 accumulator.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "golden"))
from model_def import NUM_CLASSES  # noqa: E402
from quantize import read_bin, write_bin  # noqa: E402

CLASS_OTHER, CLASS_BIRD, CLASS_DRONE = 0, 1, 2
OTHER_BIAS = 1500000  # about the 30th percentile of max(bird, drone) logit


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "weights_2class.bin"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "weights.bin"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    assert NUM_CLASSES == 3
    layers, pad, logit_scale = read_bin(args.src)
    fc = next(l for l in layers if l["name"] == "fc")
    assert fc["cout"] == 2, "source must be the 2-class model"
    rng = np.random.default_rng(args.seed)
    w = np.zeros((3, fc["cin"], 1, 1), np.int8)
    w[CLASS_OTHER, :, 0, 0] = rng.integers(-127, 128, fc["cin"])
    w[CLASS_OTHER, :4, 0, 0] = [127, -127, 127, -127]  # extremes always present
    w[CLASS_BIRD] = fc["w"][0]
    w[CLASS_DRONE] = fc["w"][1]
    bias = np.zeros(3, np.int64)
    bias[CLASS_OTHER] = OTHER_BIAS
    bias[CLASS_BIRD], bias[CLASS_DRONE] = fc["bias"][0], fc["bias"][1]
    fc.update(cout=3, w=w, bias=bias, M=np.zeros(3, np.int64), S=np.zeros(3, np.int64))
    write_bin(args.out, layers, logit_scale)
    back = next(l for l in read_bin(args.out)[0] if l["name"] == "fc")
    assert back["cout"] == 3 and np.array_equal(back["w"], w) and np.array_equal(back["bias"], bias)
    print(f"wrote {args.out}: fc 1024 -> 3, other bias {bias[0]}, "
          f"bird/drone rows = trained 2-class rows")


if __name__ == "__main__":
    main()
