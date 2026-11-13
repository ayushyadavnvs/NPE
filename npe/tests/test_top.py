"""ShuffleNet through the engine's top level (npe_top), driven only through
its AXI streams as a host would, vs the golden logits.

  python3 -m npe.tests.test_top [N]     # first N golden images (all if 0)
"""
import os
import subprocess
import sys

from npe.compiler.build import build
from npe.compiler.lower_shufflenet import OLD, lower
from npe.tests.test_engine import APDIR, BUILD, NPE


def main():
    from quantize import read_bin  # noqa: E402
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    layers, pad, _ = read_bin(os.path.join(OLD, "data", "weights.bin"))
    prog, params, names, peak = lower(layers, pad)
    d = os.path.join(BUILD, "top")
    build(prog, params, peak, d)
    root = os.path.join(OLD, "data", "golden")
    imgs = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
    imgs = imgs[:n] if n else imgs
    lst = os.path.join(d, "images.txt")
    with open(lst, "w") as f:
        f.write("\n".join(os.path.join(root, x) for x in imgs) + "\n")
    exe = os.path.join(d, "tb_top")
    subprocess.check_call(["g++", "-std=c++14", "-O2", "-w", f"-I{d}", f"-I{NPE}/hls",
                           f"-I{NPE}/isa", f"-I{NPE}/tb/shim", f"-I{APDIR}/include",
                           "-o", exe, f"{NPE}/tb/tb_top.cpp", f"{NPE}/hls/npe_top.cpp"])
    r = subprocess.run([exe, os.path.join(d, "load.bin"), lst], capture_output=True, text=True)
    print(r.stdout.strip())
    print("TEST PASSED" if r.returncode == 0 else "TEST FAILED")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
