"""Builds everything the engine needs for one model: npe_config.h (sizes the
hardware) and load.bin (what the host streams into it).

  python3 -m npe.compiler.build [--model shufflenet|mobilenetv2] [--out DIR] [--no-uram]
                                [--po N] [--pi N] [--pc N] [--board kv260|zcu104|pynqz2]

The memory plan (npe/compiler/memplan.py) picks UltraRAM / block RAM /
LUTRAM for each memory so the build fits the board.

Models: ShuffleNet from the old weights.bin, or the MobileNetV2-shaped test
network; the ONNX importer will replace these front ends.

load.bin layout (all little-endian):
  u32 n_instr, n_w, n_wd, n_q, n_qd, n_out  (words per memory; results per run)
  then the 64-bit beats of each memory in this order, as the host sends them
  to load_in: program (mode NPE_LOAD_PROG), W, WD, Q, QD. A word is sent low bits
  first and padded with zeros to a whole number of beats.
"""
import argparse
import os
import struct

import numpy as np

from npe.compiler import isa, memplan, pack
from npe.compiler.lower_shufflenet import OLD, ROOT, lower


def beats(words):
    """uint8 [n, bytes] -> bytes of the 64-bit beats, each word padded to 8n."""
    n, b = words.shape
    pad = (-b) % 8
    return np.concatenate([words, np.zeros((n, pad), np.uint8)], axis=1).tobytes()


def build(prog, params, peak, out, no_uram=False, po=1, pi=1, pc=1, board="kv260"):
    hw, mems, cfg, skipped = pack.pack(prog, params, peak, po, pi, pc)
    assert not skipped, f"instructions this engine cannot run: {skipped}"
    impl, totals = memplan.apply(cfg, board, no_uram)
    cfg["NPE_MEM_ESTIMATE"] = '"URAM %d, BRAM36 %d, LUTRAM LUTs %d"' % totals
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "npe_config.h"), "w") as f:
        f.write(pack.config_header(cfg))
    code = np.frombuffer(isa.encode(hw), np.uint8).reshape(len(hw), -1)
    n_out = sum(I.o_c for I in hw if I.o_buf == isa.BUF_OUT)
    with open(os.path.join(out, "load.bin"), "wb") as f:
        f.write(struct.pack("<6I", len(hw), len(mems["W"]), len(mems["WD"]),
                            len(mems["Q"]), len(mems["QD"]), n_out))
        for m in (code, mems["W"], mems["WD"], mems["Q"], mems["QD"]):
            f.write(beats(m))
    return cfg, n_out


def load_model(name):
    """(prog, params, peak) of a model this repo can lower today."""
    if name == "shufflenet":
        from quantize import read_bin  # noqa: E402
        layers, pad, _ = read_bin(os.path.join(OLD, "data", "weights.bin"))
        prog, params, names, peak = lower(layers, pad)
    else:
        from npe.compiler.lower_mobilenetv2 import build as mnv2
        prog, params, names, peak, _ = mnv2()
    return prog, params, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="shufflenet", choices=["shufflenet", "mobilenetv2"])
    ap.add_argument("--out", help="default npe/build/<model>_hw")
    ap.add_argument("--no-uram", action="store_true", help="block RAM only (PYNQ-Z2, ZedBoard)")
    ap.add_argument("--po", type=int, default=1, help="dense conv output words per cycle")
    ap.add_argument("--pi", type=int, default=1, help="dense conv input words per cycle")
    ap.add_argument("--pc", type=int, default=1, help="depthwise / pool channel words per cycle")
    ap.add_argument("--board", default="kv260", choices=sorted(memplan.BOARDS))
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "npe", "build", f"{a.model}_hw")
    prog, params, peak = load_model(a.model)
    cfg, n_out = build(prog, params, peak, out, a.no_uram, a.po, a.pi, a.pc, a.board)
    kb = lambda words, bits: words * bits / 8 / 1024   # noqa: E731
    P = cfg["NPE_P"]
    po, pi, pc = cfg["NPE_PO"], cfg["NPE_PI"], cfg["NPE_PC"]
    q_bits = (104 if cfg["NPE_HAS_ADD"] else 72) * P * po
    print(f"wrote {out}/npe_config.h and load.bin: {len(prog)} instructions, {n_out} outputs")
    print(f"  PO {po}, PI {pi}, PC {pc}, NB {cfg['NPE_NB']}; memories for {a.board}: "
          + ", ".join(f"{k} {v}" for k, v in cfg.items() if k.startswith("NPE_IMPL_")).replace("NPE_IMPL_", "")
          + f"; estimate {cfg['NPE_MEM_ESTIMATE']}")
    print(f"  W {kb(cfg['NPE_W_WORDS'], 8 * P * P * po * pi):.0f} KB, WD {kb(cfg['NPE_WD_WORDS'], 8 * P * pc):.0f} KB, "
          f"Q {kb(cfg['NPE_Q_WORDS'], q_bits):.0f} KB, QD {kb(cfg['NPE_QD_WORDS'], 72 * P * pc):.0f} KB, IN {cfg['NPE_IN_PIXELS'] * cfg['NPE_IN_C'] / 1024:.0f} KB")
    print("  " + ", ".join(f"{b} {kb(cfg[f'NPE_BUF_{b}_WORDS'], 8 * P):.0f} KB"
                           for b in ("S0", "A", "B", "C")) + f", V {kb(cfg['NPE_V_WORDS'], 32 * P):.0f} KB")


if __name__ == "__main__":
    main()
