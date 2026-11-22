"""Frame time with streamed weights: how long the engine waits for DRAM.

Walks the packed program like the engine does (npe/hls/npe_wstream.h):
conv and depthwise instructions take in weight-stream beats while they
compute (one per cycle at most, only while the ring has room); a streamed
conv waits before each output group until the group's words are all in.
The fetcher (npe_wfetch) delivers `bw` 128-bit beats per engine cycle into an
AXI-Stream FIFO of `fifo` beats and stops when the FIFO is full. Compute
cycles per instruction come from npe/compiler/cycles.py; a streamed conv's
cycles are split evenly over its groups. An estimate, not a cycle-exact model.

  python3 -m npe.compiler.wsim [--model shufflenet|mobilenetv2] [--po N] [--pi N] [--pc N]
                               [--stream none|all|auto] [--w-budget KB] [--ring WORDS]
                               [--bw BEATS_PER_CYCLE] [--fifo BEATS] [--mhz F] [--list]
"""
import argparse

from npe.compiler import cycles, pack, wstream


def simulate(hw, cfg, bw=0.75, fifo=64):
    """(frame cycles, wait cycles, per-instruction wait) for a packed program."""
    beats = cfg["NPE_P"] ** 2 * cfg["NPE_PO"] * cfg["NPE_PI"] // 16   # beats per W word
    ring = cfg["NPE_W_RING"]
    k = (cfg["NPE_PO"], cfg["NPE_PI"], cfg["NPE_PC"], cfg["NPE_RQ"], cfg["NPE_RQC"], cfg.get("NPE_IN_PPW", 1))
    st = dict(wr=0.0, fifo=0.0)          # words in the ring, beats in the FIFO

    def run(D, keep, takes):
        """D cycles of compute; takes: the unit takes in beats."""
        if not takes or not ring:
            st["fifo"] = min(fifo, st["fifo"] + D * bw)
            return
        room = max(0.0, keep + ring - st["wr"]) * beats       # beats the ring can take
        got = min(room, st["fifo"] + D * bw, D)
        st["wr"] += got / beats
        st["fifo"] = min(fifo, st["fifo"] + D * bw - got)

    def wait_for(need, keep):
        """Cycles until the ring holds word `need` - 1 (the unit takes beats)."""
        short = (need - st["wr"]) * beats
        if short <= 0:
            return 0.0
        t = 0.0
        from_fifo = min(st["fifo"], short)         # one beat per cycle from the FIFO
        t += from_fifo
        st["fifo"] -= from_fifo
        short -= from_fifo
        st["fifo"] = min(fifo, st["fifo"] + from_fifo * bw)
        if short > 0:                               # then at the fetch rate
            t += short / bw
        st["wr"] = need
        return t

    total = waits = 0.0
    per = []
    pos = 0
    for I in hw:
        unit, c, _ = cycles.instr_cycles(I, *k)
        w = 0.0
        if unit == "conv" and I.w_stream:
            cin_g = pack.cdiv(1 if I.a_buf == pack.isa.BUF_IN else I.a_c // pack.P, cfg["NPE_PI"])
            per_g = I.kh * I.kw * cin_g
            cout_g = pack.cdiv(pack.cdiv(I.o_c, pack.P), cfg["NPE_PO"])
            for g in range(cout_g):
                w += wait_for(pos + per_g, pos)
                run(c / cout_g, pos, True)
                pos += per_g
        else:
            run(c, pos, unit in ("conv", "dw"))
        total += c + w
        waits += w
        per.append(w)
    io = pack.cdiv(cfg["NPE_IN_PIXELS"], cfg.get("NPE_IN_PPB", 1)) + cfg["NPE_OUT_MAX"]
    return total + io, waits, per


def main():
    from npe.compiler.build import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="shufflenet", help="shufflenet, mobilenetv2 or mobilenetv2_w<W>_<IMG>")
    for kn in ("po", "pi", "pc"):
        ap.add_argument(f"--{kn}", type=int, default=1)
    ap.add_argument("--stream", default="all", choices=["none", "all", "auto"])
    ap.add_argument("--w-budget", type=float)
    ap.add_argument("--ring", type=int)
    ap.add_argument("--bw", type=float, default=0.75, help="128-bit beats per engine cycle from DRAM")
    ap.add_argument("--fifo", type=int, default=64)
    ap.add_argument("--mhz", type=float, default=200)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    prog, params, peak = load_model(a.model)
    s = wstream.choose(prog, a.stream, a.w_budget, a.po, a.pi, a.pc)
    hw, mems, cfg, _ = pack.pack(prog, params, peak, a.po, a.pi, a.pc, stream=s, ring=a.ring)
    total, waits, per = simulate(hw, cfg, a.bw, a.fifo)
    wb = mems["W"].shape[1] if len(mems["W"]) else 0
    base = cycles.frame_cycles(hw, cfg)[1]
    print(f"{a.model} PO {a.po} PI {a.pi} PC {a.pc}: {len(s)} of {sum(map(pack.streamable, prog))} streamable layers "
          f"stream, {mems['WS'].nbytes / 1024:.0f} KB from DRAM per frame, W on chip "
          f"{cfg['NPE_W_WORDS'] * wb / 1024:.0f} KB (ring {cfg['NPE_W_RING']} words)")
    print(f"  {base / 1e3:.1f}k cycles without waits, {total / 1e3:.1f}k with: waits {waits / 1e3:.1f}k "
          f"({100 * waits / total:.1f}%), {total / a.mhz / 1e3:.3f} ms at {a.mhz:g} MHz, "
          f"DRAM {mems['WS'].nbytes * a.mhz * 1e6 / total / 1e9:.2f} GB/s average")
    if a.list:
        for n, (I, w) in enumerate(zip(hw, per)):
            if w:
                print(f"  {n:3d} waits {w:8.0f} cycles")


if __name__ == "__main__":
    main()
