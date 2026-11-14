"""Builds everything the engine needs for one model: npe_config.h (sizes the
hardware) and load.bin (what the host streams into it).

  python3 -m npe.compiler.build [--model shufflenet|mobilenetv2] [--out DIR] [--no-uram]
                                [--po N] [--pi N] [--pc N] [--rq N] [--rqc N] [--ppb N] [--board kv260|zcu104|pynqz2]
                                [--stream none|all|auto] [--w-budget KB] [--ring WORDS]

The memory plan (npe/compiler/memplan.py) picks UltraRAM / block RAM /
LUTRAM for each memory so the build fits the board.

Models: ShuffleNet from the old weights.bin, or the MobileNetV2-shaped test
network; the ONNX importer will replace these front ends.

load.bin layout (all little-endian):
  u32 n_instr, n_w, n_wd, n_q, n_qd, n_out  (words per memory; results per run)
  then the 64-bit beats of each memory in this order, as the host sends them
  to load_in: program (mode NPE_LOAD_PROG), W, WD, Q, QD. A word is sent low bits
  first and padded with zeros to a whole number of beats.

weights_dram.bin: the streamed layers' weights (npe/hls/npe_wstream.h), the
  128-bit beats npe_wfetch reads from DRAM, once per inference. Empty when no
  layer streams. W in load.bin then holds only the resident layers.
"""
import argparse
import json
import os
import struct

import numpy as np

from npe.compiler import isa, memplan, pack, wstream
from npe.compiler.lower_shufflenet import OLD, ROOT, lower


def beats(words):
    """uint8 [n, bytes] -> bytes of the 64-bit beats, each word padded to 8n."""
    n, b = words.shape
    pad = (-b) % 8
    return np.concatenate([words, np.zeros((n, pad), np.uint8)], axis=1).tobytes()


def build(prog, params, peak, out, no_uram=False, po=1, pi=1, pc=1, board="kv260", rq=None, rqc=None, ppb=4,
          stream="none", w_budget=None, ring=None):
    streamed = wstream.choose(prog, stream, w_budget, po, pi, pc)
    hw, mems, cfg, skipped = pack.pack(prog, params, peak, po, pi, pc, rq, rqc, ppb, streamed, ring)
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
    with open(os.path.join(out, "weights_dram.bin"), "wb") as f:
        f.write(mems["WS"].tobytes())     # word bytes in order = 128-bit beats
    # What a host driver (npe/board/npe_driver.py) needs, without parsing C.
    loads, off = [], 24
    for mode, m in zip((1, 2, 3, 4, 5), (code, mems["W"], mems["WD"], mems["Q"], mems["QD"])):
        n_beats = len(m) * (-(-m.shape[1] // 8))
        loads.append(dict(mode=mode, count=len(m), offset=off, beats=n_beats))
        off += 8 * n_beats
    info = dict(n_instr=len(hw), n_out=n_out, loads=loads, in_c=cfg["NPE_IN_C"],
                in_pixels=cfg["NPE_IN_PIXELS"], in_ppb=cfg["NPE_IN_PPB"],
                in_beat_bits=cfg["NPE_IN_BEAT_BITS"],
                ws_beats=mems["WS"].size // 16, streamed=sorted(streamed),
                knobs={k: cfg[f"NPE_{k.upper()}"] for k in ("po", "pi", "pc", "rq", "rqc")})
    with open(os.path.join(out, "board.json"), "w") as f:
        json.dump(info, f, indent=1)
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
    ap.add_argument("--rq", type=int, help="depthwise words requantized per cycle (default PC)")
    ap.add_argument("--rqc", type=int, help="dense conv words requantized per cycle (default PO)")
    ap.add_argument("--ppb", type=int, default=4, help="image pixels per input beat (and IN banks)")
    ap.add_argument("--board", default="kv260", choices=sorted(memplan.BOARDS))
    ap.add_argument("--stream", default="none", choices=["none", "all", "auto"],
                    help="which dense conv layers stream their weights from DRAM (npe/compiler/wstream.py)")
    ap.add_argument("--w-budget", type=float, help="auto: KB of W kept on chip")
    ap.add_argument("--ring", type=int, help="W ring words (power of two)")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "npe", "build", f"{a.model}_hw")
    prog, params, peak = load_model(a.model)
    cfg, n_out = build(prog, params, peak, out, a.no_uram, a.po, a.pi, a.pc, a.board, a.rq, a.rqc, a.ppb,
                       a.stream, a.w_budget, a.ring)
    kb = lambda words, bits: words * bits / 8 / 1024   # noqa: E731
    P = cfg["NPE_P"]
    po, pi, pc = cfg["NPE_PO"], cfg["NPE_PI"], cfg["NPE_PC"]
    q_bits = (104 if cfg["NPE_HAS_ADD"] else 72) * P * po
    print(f"wrote {out}/npe_config.h and load.bin: {len(prog)} instructions, {n_out} outputs")
    print(f"  PO {po}, PI {pi}, PC {pc}, RQ {cfg['NPE_RQ']}, RQC {cfg['NPE_RQC']}, NB {cfg['NPE_NB']}; memories for {a.board}: "
          + ", ".join(f"{k} {v}" for k, v in cfg.items() if k.startswith("NPE_IMPL_")).replace("NPE_IMPL_", "")
          + f"; estimate {cfg['NPE_MEM_ESTIMATE']}")
    wb = 8 * P * P * po * pi
    if cfg["NPE_W_RING"]:
        print(f"  streamed weights: {kb(cfg['NPE_WS_WORDS'], wb):.0f} KB in DRAM (weights_dram.bin), "
              f"ring {cfg['NPE_W_RING']} words ({kb(cfg['NPE_W_RING'], wb):.0f} KB) after "
              f"{kb(cfg['NPE_W_RING_BASE'], wb):.0f} KB resident")
    print(f"  W {kb(cfg['NPE_W_WORDS'], 8 * P * P * po * pi):.0f} KB, WD {kb(cfg['NPE_WD_WORDS'], 8 * P * pc):.0f} KB, "
          f"Q {kb(cfg['NPE_Q_WORDS'], q_bits):.0f} KB, QD {kb(cfg['NPE_QD_WORDS'], 72 * P * pc):.0f} KB, IN {cfg['NPE_IN_PIXELS'] * cfg['NPE_IN_C'] / 1024:.0f} KB")
    print("  " + ", ".join(f"{b} {kb(cfg[f'NPE_BUF_{b}_WORDS'], 8 * P):.0f} KB"
                           for b in ("S0", "A", "B", "C")) + f", V {kb(cfg['NPE_V_WORDS'], 32 * P):.0f} KB")


if __name__ == "__main__":
    main()
