"""Whole ShuffleNet on the HLS engine (compiled with g++) vs the golden logits.

The engine runs all 59 instructions per image with no help from Python: the
image goes into IN, the logits come out of the OUT stream.

  python3 -m npe.tests.test_engine_e2e [N]     # first N golden images (all if 0)
"""
import os
import struct
import subprocess
import sys

import numpy as np

from npe.compiler import isa, pack
from npe.compiler.lower_shufflenet import OLD, lower
from npe.tests.test_engine import APDIR, BUILD, NPE


def main():
    from quantize import read_bin  # noqa: E402
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    layers, pad, _ = read_bin(os.path.join(OLD, "data", "weights.bin"))
    prog, params, names, peak = lower(layers, pad)
    hw, mems, cfg, skipped = pack.pack(prog, params, peak)
    assert not skipped, f"instructions the engine cannot run: {skipped}"
    d = os.path.join(BUILD, "e2e")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "npe_config.h"), "w") as f:
        f.write(pack.config_header(cfg))

    root = os.path.join(OLD, "data", "golden")
    imgs = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
    imgs = imgs[:n] if n else imgs
    n_out = prog[-1].o_c
    blob = [struct.pack("<3I", len(hw), len(imgs), n_out), isa.encode(hw),
            mems["W"].tobytes(), mems["WD"].tobytes(), mems["Q"].tobytes()]
    for x in imgs:
        img = np.fromfile(os.path.join(root, x, "input.bin"), np.uint8)
        gold = np.fromfile(os.path.join(root, x, "logits.bin"), "<i4")
        assert img.size == cfg["NPE_IN_PIXELS"] * cfg["NPE_IN_C"] and gold.size == n_out
        blob += [img.tobytes(), gold.tobytes()]
    path = os.path.join(d, "run.bin")
    with open(path, "wb") as f:
        f.write(b"".join(blob))

    exe = os.path.join(d, "tb_run")
    subprocess.check_call(["g++", "-std=c++14", "-O2", "-w", f"-I{d}", f"-I{NPE}/hls",
                           f"-I{NPE}/isa", f"-I{NPE}/tb/shim", f"-I{APDIR}/include",
                           "-o", exe, f"{NPE}/tb/tb_run.cpp"])
    r = subprocess.run([exe, path], capture_output=True, text=True)
    print(r.stdout.strip())
    print("TEST PASSED" if r.returncode == 0 else "TEST FAILED")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
