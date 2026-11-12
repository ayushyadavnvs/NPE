#!/usr/bin/env python3
"""Cycle model of the unit engine, read from the generated layer program.

Uses the same per-layer formulas as src/layers/units.h, so with the default
knobs it reproduces UNIT_IDEAL_CYCLES exactly. Then it estimates what each
planned change would buy before any RTL is written:

  base        the program as generated (merge layers, if any, cost cycles)
  no_merge    concat + shuffle folded into write addresses (merge = 0 cycles)
  dw_skip     stride-2 dw visits only the windows it outputs
  concurrent  pw and dw engines overlap inside a unit (bound: per unit, the
              busier engine sets the time)

Knobs: --pin / --pout (pointwise MACs per cycle = pin x pout), --pdw (depthwise
channels per cycle, 9 MACs each), --stem cycles, --mhz.

  python3 tools/ppa_model.py
  python3 tools/ppa_model.py --pin 16 --pout 16 --pdw 16
"""
import argparse
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(HERE, "..", "src", "weights", "unit_program.h")
CW = 8  # channels per buffer word in the current program
PW, DW, MERGE, HEAD = 0, 1, 2, 3


def read_program(path):
    text = open(path).read()
    rows = re.findall(r"\{(\d+(?:, \d+){12})\}, // (\S+)", text)
    keys = ("kind src dst aux size stride cin_g src_off src_tot cout_g "
            "w_base q_base signed").split()
    return [dict(zip(keys, map(int, r.split(", "))), name=n) for r, n in rows]


def layer_cycles(L, pin, pout, pdw, dw_skip):
    s = L["size"]
    if L["kind"] == PW:
        cin, cout = L["cin_g"] * CW, L["cout_g"] * CW
        return s * s * math.ceil(cout / pout) * math.ceil(cin / pin)
    if L["kind"] == DW:
        groups = math.ceil(L["cin_g"] * CW / pdw)
        if dw_skip and L["stride"] == 2:
            o = s // 2
            return groups * o * o
        return groups * (s + 1) * (s + 1)
    if L["kind"] == MERGE:
        return s * s * L["cout_g"]
    return s * s * L["cin_g"] + L["cin_g"]  # head: gap + fc


def layer_macs(L):
    s = L["size"]
    if L["kind"] == PW:
        return s * s * L["cin_g"] * CW * L["cout_g"] * CW
    if L["kind"] == DW:
        o = s // L["stride"]
        return o * o * L["cin_g"] * CW * 9
    return 0


def units(prog):
    """Groups program layers into shuffle units (ends at each merge)."""
    cur = []
    for L in prog:
        cur.append(L)
        if L["kind"] in (MERGE, HEAD) or L["name"] == "conv5":
            yield cur
            cur = []
    if cur:
        yield cur


def run(prog, pin, pout, pdw, merge=True, dw_skip=False, concurrent=False):
    total = 0
    for u in units(prog):
        pw = sum(layer_cycles(L, pin, pout, pdw, dw_skip) for L in u if L["kind"] == PW)
        dw = sum(layer_cycles(L, pin, pout, pdw, dw_skip) for L in u if L["kind"] == DW)
        other = sum(layer_cycles(L, pin, pout, pdw, dw_skip) for L in u
                    if L["kind"] == HEAD or (L["kind"] == MERGE and merge))
        total += (max(pw, dw) if concurrent else pw + dw) + other
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prog", default=PROG)
    ap.add_argument("--pin", type=int, default=8)
    ap.add_argument("--pout", type=int, default=16)
    ap.add_argument("--pdw", type=int, default=8)
    ap.add_argument("--stem", type=int, default=75264, help="stem cycles per frame")
    ap.add_argument("--mhz", type=float, default=200)
    a = ap.parse_args()
    prog = read_program(a.prog)
    macs = sum(layer_macs(L) for L in prog)
    inst = a.pin * a.pout + a.pdw * 9
    print(f"engine: pw {a.pin}x{a.pout}, dw {a.pdw}x9 = {inst} MACs/cycle, "
          f"{len(prog)} layers, {macs / 1e6:.1f} M engine MACs, {a.mhz:.0f} MHz")
    print(f"{'variant':<34}{'engine cyc':>11}{'MAC busy':>10}{'fps seq':>9}{'fps ovl':>9}")
    variants = [
        ("base", {}),
        ("no_merge", dict(merge=False)),
        ("no_merge + dw_skip", dict(merge=False, dw_skip=True)),
        ("no_merge + dw_skip + concurrent", dict(merge=False, dw_skip=True, concurrent=True)),
    ]
    for name, kw in variants:
        c = run(prog, a.pin, a.pout, a.pdw, **kw)
        busy = macs / (c * inst)
        seq = a.mhz * 1e6 / (c + a.stem)        # stem and engine back to back
        ovl = a.mhz * 1e6 / max(c, a.stem)       # frames overlapped
        print(f"{name:<34}{c:>11}{busy:>10.0%}{seq:>9.0f}{ovl:>9.0f}")
    print("fps seq: frames do not overlap (today). fps ovl: stem of frame n+1 "
          "runs during the engine of frame n.")


if __name__ == "__main__":
    main()
