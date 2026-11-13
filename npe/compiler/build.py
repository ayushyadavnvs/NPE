"""Builds everything the engine needs for one model: npe_config.h (sizes the
hardware) and load.bin (what the host streams into it).

  python3 -m npe.compiler.build [--out DIR] [--no-uram]

Today the model is ShuffleNet from the old weights.bin; the ONNX importer
will replace that front end.

load.bin layout (all little-endian):
  u32 n_instr, n_w, n_wd, n_q, n_out     (words per memory; results per run)
  then the 64-bit beats of each memory in this order, as the host sends them
  to load_in: program (mode NPE_LOAD_PROG), W, WD, Q. A word is sent low bits
  first and padded with zeros to a whole number of beats.
"""
import argparse
import os
import struct

import numpy as np

from npe.compiler import isa, pack
from npe.compiler.lower_shufflenet import OLD, ROOT, lower


def beats(words):
    """uint8 [n, bytes] -> bytes of the 64-bit beats, each word padded to 8n."""
    n, b = words.shape
    pad = (-b) % 8
    return np.concatenate([words, np.zeros((n, pad), np.uint8)], axis=1).tobytes()


def build(prog, params, peak, out, no_uram=False):
    hw, mems, cfg, skipped = pack.pack(prog, params, peak)
    assert not skipped, f"instructions this engine cannot run: {skipped}"
    if no_uram:
        cfg["NPE_NO_URAM"] = 1
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "npe_config.h"), "w") as f:
        f.write(pack.config_header(cfg))
    code = np.frombuffer(isa.encode(hw), np.uint8).reshape(len(hw), -1)
    n_out = sum(I.o_c for I in hw if I.o_buf == isa.BUF_OUT)
    with open(os.path.join(out, "load.bin"), "wb") as f:
        f.write(struct.pack("<5I", len(hw), len(mems["W"]), len(mems["WD"]),
                            len(mems["Q"]), n_out))
        for m in (code, mems["W"], mems["WD"], mems["Q"]):
            f.write(beats(m))
    return cfg, n_out


def main():
    from quantize import read_bin  # noqa: E402
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "npe", "build", "shufflenet_hw"))
    ap.add_argument("--no-uram", action="store_true", help="block RAM only (PYNQ-Z2, ZedBoard)")
    a = ap.parse_args()
    layers, pad, _ = read_bin(os.path.join(OLD, "data", "weights.bin"))
    prog, params, names, peak = lower(layers, pad)
    cfg, n_out = build(prog, params, peak, a.out, a.no_uram)
    kb = lambda words, bits: words * bits / 8 / 1024   # noqa: E731
    P = cfg["NPE_P"]
    print(f"wrote {a.out}/npe_config.h and load.bin: {len(prog)} instructions, {n_out} outputs")
    print(f"  W {kb(cfg['NPE_W_WORDS'], 8 * P * P):.0f} KB, WD {kb(cfg['NPE_WD_WORDS'], 8 * P):.0f} KB, "
          f"Q {kb(cfg['NPE_Q_WORDS'], 72 * P):.0f} KB, IN {cfg['NPE_IN_PIXELS'] * cfg['NPE_IN_C'] / 1024:.0f} KB")
    print("  " + ", ".join(f"{b} {kb(cfg[f'NPE_BUF_{b}_WORDS'], 8 * P):.0f} KB"
                           for b in ("S0", "A", "B", "C")) + f", V {kb(cfg['NPE_V_WORDS'], 32 * P):.0f} KB")


if __name__ == "__main__":
    main()
