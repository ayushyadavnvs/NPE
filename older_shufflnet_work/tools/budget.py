#!/usr/bin/env python3
"""MAC, parameter and cycle budget for ShuffleNetV2 x0.5 (2-class head).

Examples:
  python3 tools/budget.py                    # summary at several resolutions
  python3 tools/budget.py --res 224 --layers # per-layer table
  python3 tools/budget.py --res 224 --pin 8 --pout 16 --pdw 16 --pstem 54

Cycle model: a pointwise engine of pin x pout MACs/cycle, a depthwise engine
of pdw channels x 9 MACs/cycle, and a stem of pstem MACs/cycle, run one layer
after another (no overlap, no efficiency loss). Real hardware lands at roughly
50-70 % of this ideal, so treat the FPS column as an upper bound.
"""
import argparse
import math

CH = (24, 48, 96, 192, 1024)
REPEATS = (4, 8, 4)


def model(res, ncls=2):
    layers = []  # (name, kind, out_size, cin, cout, k, stride, macs, params)

    def conv(name, kind, size_in, cin, cout, k, stride):
        out = math.ceil(size_in / stride)
        if kind == "dw":
            macs, params = out * out * cin * k * k, cin * k * k
        else:
            macs, params = out * out * cin * cout * k * k, cin * cout * k * k
        layers.append((name, kind, out, cin, cout, k, stride, macs, params))
        return out

    size = conv("conv1", "std", res, 3, CH[0], 3, 2)
    size = math.ceil(size / 2)
    layers.append(("maxpool", "pool", size, CH[0], CH[0], 3, 2, 0, 0))
    cin = CH[0]
    for stage, (c, reps) in enumerate(zip(CH[1:4], REPEATS), start=2):
        half = c // 2
        conv(f"s{stage}u0.b1.dw", "dw", size, cin, cin, 3, 2)
        conv(f"s{stage}u0.b1.pw", "pw", math.ceil(size / 2), cin, half, 1, 1)
        conv(f"s{stage}u0.b2.pw1", "pw", size, cin, half, 1, 1)
        size = conv(f"s{stage}u0.b2.dw", "dw", size, half, half, 3, 2)
        conv(f"s{stage}u0.b2.pw2", "pw", size, half, half, 1, 1)
        for u in range(1, reps):
            conv(f"s{stage}u{u}.pw1", "pw", size, half, half, 1, 1)
            conv(f"s{stage}u{u}.dw", "dw", size, half, half, 3, 1)
            conv(f"s{stage}u{u}.pw2", "pw", size, half, half, 1, 1)
        cin = c
    conv("conv5", "pw", size, cin, CH[4], 1, 1)
    layers.append(("fc", "fc", 1, CH[4], ncls, 1, 1, CH[4] * ncls, CH[4] * ncls + ncls))
    return layers


def cycles(layers, pin, pout, pdw, pstem):
    total = 0
    for name, kind, out, cin, cout, k, s, macs, _ in layers:
        if kind == "std":
            total += macs / pstem
        elif kind == "dw":
            total += out * out * math.ceil(cin / pdw)
        elif kind in ("pw", "fc"):
            total += out * out * math.ceil(cin / pin) * math.ceil(cout / pout)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="*", default=[96, 128, 160, 224, 320, 640])
    ap.add_argument("--pin", type=int, default=8)
    ap.add_argument("--pout", type=int, default=16)
    ap.add_argument("--pdw", type=int, default=16)
    ap.add_argument("--pstem", type=int, default=54)
    ap.add_argument("--mhz", type=float, default=200)
    ap.add_argument("--layers", action="store_true")
    a = ap.parse_args()
    print(f"engine: pw {a.pin}x{a.pout}, dw {a.pdw}x9, stem {a.pstem} MAC/cycle @ {a.mhz:.0f} MHz")
    print(f"{'res':>5} {'MMAC':>8} {'params_k':>9} {'GMAC/s@60':>10} {'Mcycles':>8} {'ideal_fps':>9}")
    for r in a.res:
        L = model(r)
        macs = sum(x[7] for x in L)
        params = sum(x[8] for x in L)
        cyc = cycles(L, a.pin, a.pout, a.pdw, a.pstem)
        print(f"{r:>5} {macs/1e6:>8.2f} {params/1e3:>9.1f} {macs*60/1e9:>10.2f} "
              f"{cyc/1e6:>8.3f} {a.mhz*1e6/cyc:>9.0f}")
        if a.layers:
            for name, kind, out, cin, cout, k, s, m, p in L:
                print(f"    {name:<14} {kind:<4} out {out:>3}x{out:<3} {cin:>4}->{cout:<5}"
                      f" k{k} s{s} {m/1e6:7.3f} MMAC {p:>7} params")


if __name__ == "__main__":
    main()
