#!/usr/bin/env python3
"""DRAM weight-streaming model for the generic NPE engine.

Per layer: engine cycles (same loop counts as npe/hls: conv = oh*ow*cout_w*kh*kw*cin_w,
dw = oh*ow*cw*kh*kw, fc = cout_w*cin_w, pool/gap by pixels) and weight + Q bytes.
Then an event simulation of a weight fetcher (one AXI read stream of BW bytes/cycle,
LAT cycles per tile request) feeding the engine through a ring of K slots of S bytes.
Conv layers are split into output-channel tiles (whole output words, all input
channels, so no partial sums); each tile is fetched once and reused over every pixel.

  python3 npe/tools/wstream_model.py                 # summary, all models
  python3 npe/tools/wstream_model.py --layers resnet18
"""
import argparse
import math

# ---------------------------------------------------------------- model zoo
def L(kind, name, oh, ow, cin, cout, k=1, s=1, img=False):
    return dict(kind=kind, name=name, oh=oh, ow=ow, cin=cin, cout=cout, k=k, s=s, img=img)


def resnet18(res=224, classes=1000):
    ls, h = [], res
    if res >= 128:
        h //= 2; ls.append(L("conv", "conv1", h, h, 3, 64, 7, 2, img=True))
        h //= 2; ls.append(L("pool", "maxpool", h, h, 64, 64, 3, 2))
    else:
        ls.append(L("conv", "conv1", h, h, 3, 64, 3, 1, img=True))
    cin = 64
    for si, c in enumerate([64, 128, 256, 512]):
        for b in range(2):
            s = 2 if (b == 0 and si > 0) else 1
            h2 = h // s
            ls.append(L("conv", f"l{si+1}.{b}.c1", h2, h2, cin, c, 3, s))
            ls.append(L("conv", f"l{si+1}.{b}.c2", h2, h2, c, c, 3, 1))
            if s == 2 or cin != c:
                ls.append(L("conv", f"l{si+1}.{b}.ds", h2, h2, cin, c, 1, s))
            cin, h = c, h2
    ls.append(L("gap", "gap", 1, 1, 512, 512, h))
    ls.append(L("fc", "fc", 1, 1, 512, classes))
    return ls


def resnet8():
    ls, h = [L("conv", "conv1", 32, 32, 3, 16, 3, 1, img=True)], 32
    cin = 16
    for si, c in enumerate([16, 32, 64]):
        s = 1 if si == 0 else 2
        h //= s
        ls.append(L("conv", f"s{si}.c1", h, h, cin, c, 3, s))
        ls.append(L("conv", f"s{si}.c2", h, h, c, c, 3, 1))
        if s == 2:
            ls.append(L("conv", f"s{si}.ds", h, h, cin, c, 1, s))
        cin = c
    ls += [L("gap", "gap", 1, 1, 64, 64, h), L("fc", "fc", 1, 1, 64, 10)]
    return ls


def dscnn():
    ls = [L("conv", "conv1", 25, 5, 1, 64, 10, 2, img=True)]  # 10x4 kernel ~ k=6.3; use 40 taps below
    ls[0]["kh"], ls[0]["kw"] = 10, 4
    for i in range(4):
        ls.append(L("dw", f"dw{i}", 25, 5, 64, 64, 3, 1))
        ls.append(L("conv", f"pw{i}", 25, 5, 64, 64, 1, 1))
    ls += [L("gap", "gap", 1, 1, 64, 64, 11), L("fc", "fc", 1, 1, 64, 12)]
    return ls


def mobilenetv2(res=224, classes=1000):
    h = res // 2
    ls, cin = [L("conv", "conv1", h, h, 3, 32, 3, 2, img=True)], 32
    for bi, (t, c, n, s) in enumerate([(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
                                       (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]):
        for r in range(n):
            st = s if r == 0 else 1
            e = cin * t
            if t != 1:
                ls.append(L("conv", f"b{bi}.{r}.exp", h, h, cin, e, 1, 1))
            h //= st
            ls.append(L("dw", f"b{bi}.{r}.dw", h, h, e, e, 3, st))
            ls.append(L("conv", f"b{bi}.{r}.prj", h, h, e, c, 1, 1))
            cin = c
    ls += [L("conv", "conv_last", h, h, cin, 1280, 1, 1), L("gap", "gap", 1, 1, 1280, 1280, h),
           L("fc", "fc", 1, 1, 1280, classes)]
    return ls


def shufflenetv2(width=0.5, res=224, classes=1000):
    ch = {0.5: [48, 96, 192, 1024], 1.0: [116, 232, 464, 1024]}[width]
    h = res // 2
    ls, cin = [L("conv", "conv1", h, h, 3, 24, 3, 2, img=True)], 24
    h //= 2; ls.append(L("pool", "maxpool", h, h, 24, 24, 3, 2))
    for si, (c, n) in enumerate(zip(ch[:3], [4, 8, 4])):
        half = c // 2
        for r in range(n):
            if r == 0:
                h //= 2
                ls.append(L("dw", f"s{si}.0.b1dw", h, h, cin, cin, 3, 2))
                ls.append(L("conv", f"s{si}.0.b1pw", h, h, cin, half, 1, 1))
                ls.append(L("conv", f"s{si}.0.pw1", h * 2, h * 2, cin, half, 1, 1))
            else:
                ls.append(L("conv", f"s{si}.{r}.pw1", h, h, half, half, 1, 1))
            ls.append(L("dw", f"s{si}.{r}.dw", h, h, half, half, 3, 2 if r == 0 else 1))
            ls.append(L("conv", f"s{si}.{r}.pw2", h, h, half, half, 1, 1))
        cin = c
    ls += [L("conv", "conv5", h, h, cin, ch[3], 1, 1), L("gap", "gap", 1, 1, ch[3], ch[3], h),
           L("fc", "fc", 1, 1, ch[3], classes)]
    return ls


MODELS = {
    "shufflenetv2_x0.5": lambda: shufflenetv2(0.5),
    "shufflenetv2_x1.0": lambda: shufflenetv2(1.0),
    "mobilenetv2": mobilenetv2,
    "resnet8": resnet8,
    "dscnn": dscnn,
    "resnet18_cifar": lambda: resnet18(32, 10),
    "resnet18_imagenet": resnet18,
}

# ---------------------------------------------------------------- engine model
def cost(l, P, wbits):
    """(cycles, weight words, bytes per weight word, words per output word, out words, Q bytes)"""
    cw = lambda c: math.ceil(c / P)
    kh, kw = l.get("kh", l["k"]), l.get("kw", l["k"])
    pix = l["oh"] * l["ow"]
    wb = P * P * wbits // 8
    qb = 9 * P  # bias 4 B, M 4 B, S 1 B per channel
    if l["kind"] == "conv":
        ciw = 1 if l["img"] else cw(l["cin"])
        per = kh * kw * ciw
        return pix * cw(l["cout"]) * per, cw(l["cout"]) * per, wb, per, cw(l["cout"]), cw(l["cout"]) * qb
    if l["kind"] == "fc":
        per = cw(l["cin"])
        return cw(l["cout"]) * per, cw(l["cout"]) * per, wb, per, cw(l["cout"]), cw(l["cout"]) * qb
    if l["kind"] == "dw":
        # WD words are P bytes (one tap of P channels); fetch per channel word
        per = kh * kw
        return pix * cw(l["cout"]) * per, cw(l["cout"]) * per, P * wbits // 8, per, cw(l["cout"]), cw(l["cout"]) * qb
    if l["kind"] == "pool":
        return pix * cw(l["cout"]) * l["k"] ** 2, 0, 0, 0, 0, 0
    if l["kind"] == "gap":
        return l["k"] ** 2 * cw(l["cout"]), 0, 0, 0, 0, 0
    raise ValueError(l)


def tiles(layers, P, wbits, slot):
    """Work items in program order: (layer, cycles, bytes). Weighted layers are cut
    into output-word tiles of at most `slot` bytes (weights + Q)."""
    out = []
    for l in layers:
        cyc, nw, wb, per, ow, qbytes = cost(l, P, wbits)
        if nw == 0:
            out.append((l["name"], cyc, 0))
            continue
        per_ow_bytes = per * wb + qbytes / ow
        n = max(1, int(slot // per_ow_bytes))  # output words per tile
        if per_ow_bytes > slot:
            raise SystemExit(f"{l['name']}: one output word needs {per_ow_bytes/1024:.1f} KB > slot")
        left = ow
        while left:
            k = min(n, left)
            out.append((l["name"], cyc * k / ow, k * per_ow_bytes))
            left -= k
    return out


def simulate(items, bw, lat, K):
    """Fetcher runs ahead by up to K slots. Returns (total cycles, stall cycles, per-layer stall)."""
    fetch_done, comp_end = [], []
    t_fetch = 0.0
    t_comp = 0.0
    stall = 0.0
    per = {}
    widx = []  # indices of items that use a slot
    for i, (name, cyc, by) in enumerate(items):
        if by:
            j = len(widx)
            start = t_fetch
            if j >= K:  # wait for slot j-K to be released (end of that tile's compute)
                start = max(start, comp_end[widx[j - K]])
            t_fetch = start + lat + by / bw
            widx.append(i)
            ready = t_fetch
        else:
            ready = 0.0
        s = max(0.0, ready - t_comp)
        stall += s
        per[name] = per.get(name, 0) + s
        t_comp = max(t_comp, ready) + cyc
        comp_end.append(t_comp)
    return t_comp, stall, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--bw", type=float, default=12.0, help="DRAM bytes per engine cycle")
    ap.add_argument("--lat", type=int, default=150, help="cycles per tile request")
    ap.add_argument("--slot", type=int, default=64 * 1024, help="bytes per ring slot")
    ap.add_argument("--K", type=int, default=2, help="ring slots (2 = ping-pong)")
    ap.add_argument("--mhz", type=float, default=276)
    ap.add_argument("--layers", help="print per-layer table for this model")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    for wbits in (8, 4):
        print(f"\n== int{wbits}, P={a.P}, BW={a.bw} B/cyc ({a.bw*a.mhz/1e3:.2f} GB/s @ {a.mhz:.0f} MHz), "
              f"ring {a.K} x {a.slot//1024} KB")
        print(f"{'model':20s} {'weights':>9s} {'compute':>9s} {'all-DRAM':>9s} {'stream':>9s} {'stall':>7s} {'naive':>9s}")
        for m, fn in MODELS.items():
            layers = fn()
            items = tiles(layers, a.P, wbits, a.slot)
            comp = sum(c for _, c, _ in items)
            wbytes = sum(b for _, _, b in items)
            tot, st, per = simulate(items, a.bw, a.lat, a.K)
            naive, _, _ = simulate(items, a.bw, a.lat, 1)  # no overlap: fetch, then compute
            ms = lambda c: c / a.mhz / 1e3
            print(f"{m:20s} {wbytes/2**20:7.2f}MB {ms(comp):7.2f}ms {ms(wbytes/a.bw):7.2f}ms "
                  f"{ms(tot):7.2f}ms {100*st/tot:6.2f}% {ms(naive):7.2f}ms")
            if a.layers == m:
                print(f"   {'layer':14s} {'cycles':>10s} {'KB':>8s} {'B/cyc need':>10s} {'bound':>7s} {'stall cyc':>10s}")
                for l in layers:
                    cyc, nw, wb, per_, ow, qb = cost(l, a.P, wbits)
                    by = nw * wb + qb
                    need = by / cyc if cyc else 0
                    print(f"   {l['name']:14s} {cyc:10.0f} {by/1024:8.1f} {need:10.2f} "
                          f"{'DRAM' if need > a.bw else 'MAC':>7s} {per.get(l['name'],0):10.0f}")
    if a.sweep:
        print("\n== sweep (int8, resnet18_imagenet / mobilenetv2): stall % vs ring")
        for P in (8, 16, 32):
            for K, slot in ((2, 32), (2, 64), (2, 128), (4, 64), (8, 64), (16, 64)):
                row = []
                for m in ("resnet18_imagenet", "resnet18_cifar", "mobilenetv2"):
                    try:
                        items = tiles(MODELS[m](), P, 8, slot * 1024)
                    except SystemExit:
                        row.append("  n/a"); continue
                    tot, st, _ = simulate(items, a.bw, a.lat, K)
                    row.append(f"{100*st/tot:5.1f}")
                print(f"P={P:2d} K={K:2d} slot={slot:3d}KB ring={K*slot:4d}KB  r18in {row[0]}%  r18c {row[1]}%  mbv2 {row[2]}%")


if __name__ == "__main__":
    main()
