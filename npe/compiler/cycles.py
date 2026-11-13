"""Cycle model of the engine: clock cycles per instruction and frames per
second, from the loop structure of the HLS units (every loop runs at II = 1,
as synthesis reported). Each loop also pays its pipeline depth once, and each
instruction a small fixed cost for decode and the call.

  python3 -m npe.compiler.cycles [--model shufflenet|mobilenetv2] [--mhz 200] [--list]
                                 [--po N] [--pi N] [--pc N] [--rq N]

This is an estimate: it does not include AXI back-pressure from the host or
DRAM, and frames are processed one after another (no overlap).
"""
import argparse

from npe.compiler import isa, pack
from npe.compiler.build import load_model

P = pack.P
# pipeline depths from the csynth log (7c0b738)
DEPTH = {"conv": 13, "dw": 11, "pool": 5, "gap": 4, "fc": 4}
CALL = 10   # decode + call per instruction


def cdiv(a, b):
    return -(-a // b)


def instr_cycles(I, po=1, pi=1, pc=1, rq=None):
    """(unit, cycles, MACs) of one packed instruction on a build with knobs
    po, pi, pc (see npe/hls/npe_types.h)."""
    pix = I.o_h * I.o_w
    cout_w = cdiv(I.o_c, P)
    taps = I.kh * I.kw
    if pack.is_fc(I):
        cin_w = I.a_c // P
        return "fc", cout_w * (cin_w + DEPTH["fc"] + P) + CALL, I.o_c * I.a_c
    if I.op == isa.OP_POOL_MAX:
        return "pool", pix * cdiv(cout_w, pc) * taps + DEPTH["pool"] + CALL, 0
    if I.op == isa.OP_GAP:
        return "gap", I.a_h * I.a_w * cout_w + DEPTH["gap"] + CALL, 0
    if pack.is_depthwise(I):
        return "dw", pix * cdiv(cout_w, pc) * taps + cdiv(pc, rq or pc) + DEPTH["dw"] + CALL, pix * I.o_c * taps
    cin_w = 1 if I.a_buf == isa.BUF_IN else I.a_c // P
    macs = pix * I.o_c * taps * I.a_c
    return "conv", pix * cdiv(cout_w, po) * taps * cdiv(cin_w, pi) + DEPTH["conv"] + CALL, macs


def frame_cycles(prog, cfg):
    k = (cfg.get("NPE_PO", 1), cfg.get("NPE_PI", 1), cfg.get("NPE_PC", 1), cfg.get("NPE_RQ"))
    per = [instr_cycles(I, *k) for I in prog]
    io = cfg["NPE_IN_PIXELS"] + cfg["NPE_OUT_MAX"]   # read image, write results
    return per, sum(c for _, c, _ in per) + io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="shufflenet", choices=["shufflenet", "mobilenetv2"])
    ap.add_argument("--mhz", type=float, default=200)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--po", type=int, default=1)
    ap.add_argument("--pi", type=int, default=1)
    ap.add_argument("--pc", type=int, default=1)
    ap.add_argument("--rq", type=int)
    a = ap.parse_args()
    prog, params, peak = load_model(a.model)
    hw, mems, cfg, skipped = pack.pack(prog, params, peak, a.po, a.pi, a.pc, a.rq)
    per, total = frame_cycles(hw, cfg)
    by = {}
    for u, c, m in per:
        by.setdefault(u, [0, 0])
        by[u][0] += c
        by[u][1] += m
    macs = sum(m for _, _, m in per)
    print(f"PO {a.po}, PI {a.pi}, PC {a.pc}: {P * P * a.po * a.pi} conv MACs/cycle, "
          f"{P * a.pc} depthwise MACs/cycle")
    print(f"{a.model}: {len(hw)} instructions, {macs / 1e6:.1f}M MACs, {total / 1e6:.3f}M cycles "
          f"per frame -> {total / a.mhz / 1e3:.2f} ms, {a.mhz * 1e6 / total:.0f} fps at {a.mhz:g} MHz")
    print(f"  image in: {cfg['NPE_IN_PIXELS']} cycles")
    for u, (c, m) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        eff = f", {m / c:.1f} MACs/cycle" if m else ""
        print(f"  {u:5s} {c / 1e6:7.3f}M cycles ({100 * c / total:4.1f}%){eff}")
    if a.list:
        for n, (I, (u, c, m)) in enumerate(zip(hw, per)):
            print(f"  {n:3d} {u:5s} {c:9d} cycles {m / 1e6:7.2f}M MACs")


if __name__ == "__main__":
    main()
