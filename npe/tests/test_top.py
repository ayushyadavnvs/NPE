"""A whole model through the engine's top level (npe_top), driven only through
its AXI streams as a host would.

  python3 -m npe.tests.test_top [N] [--model shufflenet|mobilenetv2] [--po N] [--pi N] [--pc N] [--rq N] [--rqc N]

shufflenet   the first N golden images (all if 0) vs the golden logits
mobilenetv2  the MobileNetV2-shaped network on N random images (default 4)
             vs the reference interpreter
"""
import argparse
import os
import subprocess
import sys

import numpy as np

from npe.compiler import isa
from npe.compiler.build import build
from npe.compiler.lower_shufflenet import OLD, lower
from npe.ref.isa_ref import Machine
from npe.tests.test_engine import APDIR, BUILD, NPE


def shufflenet_case(n):
    from quantize import read_bin  # noqa: E402
    layers, pad, _ = read_bin(os.path.join(OLD, "data", "weights.bin"))
    prog, params, names, peak = lower(layers, pad)
    root = os.path.join(OLD, "data", "golden")
    imgs = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
    imgs = imgs[:n] if n else imgs
    return prog, params, peak, [os.path.join(root, x) for x in imgs]


def mobilenetv2_case(n, d):
    """Random images and their reference logits, in the golden-set layout."""
    from npe.compiler.lower_mobilenetv2 import build as mnv2
    prog, params, names, peak, sample = mnv2()
    rng = np.random.default_rng(1)
    dirs = []
    for k in range(n or 4):
        x = rng.integers(0, 256, sample.shape)
        logits = Machine(params, peak).run(prog, x).reshape(-1)
        out = os.path.join(d, "images", f"img{k:02d}")
        os.makedirs(out, exist_ok=True)
        x.astype(np.uint8).tofile(os.path.join(out, "input.bin"))
        logits.astype("<i4").tofile(os.path.join(out, "logits.bin"))
        dirs.append(out)
    return prog, params, peak, dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=0)
    ap.add_argument("--model", default="shufflenet", choices=["shufflenet", "mobilenetv2"])
    ap.add_argument("--po", type=int, default=1)
    ap.add_argument("--pi", type=int, default=1)
    ap.add_argument("--pc", type=int, default=1)
    ap.add_argument("--rq", type=int, default=0)
    ap.add_argument("--rqc", type=int, default=0)
    a = ap.parse_args()
    d = os.path.join(BUILD, f"top_{a.model}_po{a.po}_pi{a.pi}_pc{a.pc}_rq{a.rq}_rqc{a.rqc}")
    os.makedirs(d, exist_ok=True)
    if a.model == "shufflenet":
        prog, params, peak, dirs = shufflenet_case(a.n)
    else:
        prog, params, peak, dirs = mobilenetv2_case(a.n, d)
    cfg, _ = build(prog, params, peak, d, po=a.po, pi=a.pi, pc=a.pc, rq=a.rq or None, rqc=a.rqc or None)
    print(f"{a.model}: {len(prog)} instructions, "
          f"{sum(I.epi == isa.EPI_ADD for I in prog)} with ADD, NPE_HAS_ADD={cfg['NPE_HAS_ADD']}")
    lst = os.path.join(d, "images.txt")
    with open(lst, "w") as f:
        f.write("\n".join(dirs) + "\n")
    exe = os.path.join(d, "tb_top")
    subprocess.check_call(["g++", "-std=c++14", "-O2", "-w", f"-I{d}", f"-I{NPE}/hls",
                           f"-I{NPE}/isa", f"-I{NPE}/tb/shim", f"-I{APDIR}/include",
                           "-o", exe, f"{NPE}/tb/tb_top.cpp", f"{NPE}/hls/npe_top.cpp"])
    r = subprocess.run([exe, os.path.join(d, "load.bin"), lst], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    print("TEST PASSED" if r.returncode == 0 else f"TEST FAILED (exit {r.returncode})")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
