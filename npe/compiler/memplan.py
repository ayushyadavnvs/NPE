"""Memory plan: which RAM technology (UltraRAM, block RAM, LUTRAM) each
on-chip memory of a build uses, so the build fits its board. Wider knobs
(npe/hls/npe_types.h) make W, Q, WD, QD wider and the feature buffers more
banked, which changes the best mapping.

Every combination is tried (9 memories, 2 or 3 choices each) and the one
that fits the board with the fewest LUTRAM LUTs, then the fewest RAM
blocks, wins. The counts are the usual primitive tiling estimates; Vitis
reports the real numbers.
"""
import itertools
import math

from npe.compiler import isa

P = 8
# UltraRAM, block RAM (36 Kb), LUT budgets; BRAM keeps 10% for the rest of
# the design (FIFOs, AXI).
BOARDS = {
    "kv260": dict(part="xck26-sfvc784-2LV-c", uram=64, bram=144, lut=117120),
    "zcu104": dict(part="xczu7ev-ffvc1156-2-e", uram=96, bram=312, lut=230400),
    "pynqz2": dict(part="xc7z020clg400-1", uram=0, bram=140, lut=53200),
}


def uram(w, d):
    return math.ceil(w / 72) * math.ceil(d / 4096)


def bram(w, d, true_dual):
    modes = [(36, 1024), (18, 2048), (9, 4096), (4, 8192), (2, 16384), (1, 32768)]
    if not true_dual:
        modes.append((72, 512))
    return min(math.ceil(w / mw) * math.ceil(d / md) for mw, md in modes)


def lutram(w, d):   # RAM64M: 3 bits x 64 words in 4 LUTs (simple dual port)
    return math.ceil(d / 64) * math.ceil(w / 3) * 4


def memories(cfg):
    """name -> (banks, width bits, depth, feature buffer)."""
    nb = cfg["NPE_NB"]
    po, pi, pc = cfg["NPE_PO"], cfg["NPE_PI"], cfg["NPE_PC"]
    ql = (104 if cfg["NPE_HAS_ADD"] else 72) * P
    m = {b: (nb, 8 * P, cfg[f"NPE_BUF_{b}_WORDS"] // nb, True) for b in ("S0", "A", "B", "C")}
    ppb = cfg.get("NPE_IN_PPB", 1)
    m["IN"] = (ppb, 8 * cfg["NPE_IN_C"], -(-cfg["NPE_IN_PIXELS"] // ppb) + 1, False)
    m["W"] = (1, 8 * P * P * po * pi, cfg["NPE_W_WORDS"], False)
    m["WD"] = (1, 8 * P * pc, cfg["NPE_WD_WORDS"], False)
    m["Q"] = (1, ql * po, cfg["NPE_Q_WORDS"], False)
    m["QD"] = (1, 72 * P * pc, cfg["NPE_QD_WORDS"], False)
    return m


def cost(spec, impl):
    banks, w, d, feat = spec
    if impl == "uram":
        return (banks * uram(w, d), 0, 0)
    if impl == "bram":
        return (0, banks * bram(w, d, feat), 0)
    return (0, 0, banks * lutram(w, d))


def plan(cfg, board="kv260", no_uram=False):
    """Returns ({memory: impl}, (uram, bram36, lutram LUTs)) or raises."""
    b = dict(BOARDS[board])
    if no_uram:
        b["uram"] = 0
    mems = memories(cfg)
    names = list(mems)
    choices = []
    for n in names:
        c = ["bram"] + (["uram"] if b["uram"] else [])
        if not mems[n][3] and mems[n][2] <= 1024:   # small read-only memories
            c.append("lutram")
        choices.append(c)
    best = None
    for combo in itertools.product(*choices):
        u = br = lut = 0
        for n, impl in zip(names, combo):
            cu, cb, cl = cost(mems[n], impl)
            u, br, lut = u + cu, br + cb, lut + cl
        if u > b["uram"] or br > 0.9 * b["bram"] or lut > 0.25 * b["lut"]:
            continue
        key = (lut, u + br)
        if best is None or key < best[0]:
            best = (key, dict(zip(names, combo)), (u, br, lut))
    if best is None:
        raise ValueError(f"this build does not fit {board}: memories {mems}")
    return best[1], best[2]


def apply(cfg, board="kv260", no_uram=False):
    """Adds NPE_IMPL_<memory> macros to cfg; returns the plan and totals."""
    impl, totals = plan(cfg, board, no_uram)
    for n, v in impl.items():
        cfg[f"NPE_IMPL_{n}"] = v
    return impl, totals
