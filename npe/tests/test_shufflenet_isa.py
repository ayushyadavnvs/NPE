"""ShuffleNet on the generic instruction set vs the golden logits.

  python3 -m npe.tests.test_shufflenet_isa [N]     # first N golden images (all if 0)
"""
import os
import sys

import numpy as np

from npe.compiler import isa
from npe.compiler.lower_shufflenet import OLD, lower
from npe.ref.isa_ref import Machine


def main():
    from quantize import read_bin  # noqa: E402
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    layers, pad, _ = read_bin(os.path.join(OLD, "data", "weights.bin"))
    prog, params, names, peak = lower(layers, pad)
    root = os.path.join(OLD, "data", "golden")
    imgs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    imgs = imgs[:n] if n else imgs
    bad = 0
    for d in imgs:
        x = np.fromfile(os.path.join(root, d, "input.bin"), np.uint8).reshape(224, 224, 3)
        gold = np.fromfile(os.path.join(root, d, "logits.bin"), "<i4")
        m = Machine(params, peak)
        got = m.run(prog, x).reshape(-1)
        ok = np.array_equal(got, gold)
        bad += not ok
        print(f"{d:8s} {'PASS' if ok else 'FAIL'} {got.tolist()} golden {gold.tolist()}")
    print(f"{'TEST PASSED' if not bad else 'TEST FAILED'}: {len(imgs) - bad} of {len(imgs)} "
          f"images bit-exact on {len(prog)} generic instructions")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
