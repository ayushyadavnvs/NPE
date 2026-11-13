"""Packs a program and its parameters into the memory images of the HLS engine.

The reference interpreter works on elements and parameter dicts. The engine
works on words of P channels and four parameter memories. Their widths depend
on the parallelism knobs of the build (see npe/hls/npe_types.h):
PO, PI (dense conv output / input words per cycle), PC (depthwise
channel words per cycle), RQ and RQC (depthwise / dense conv output words
requantized per cycle).

  W  dense conv weights. One word = PO x PI blocks of P x P int8, the
     weights of one cycle. Layer with cout_w output words, cin_w input words:
     output group og = output words PO*og .. PO*og+PO-1, input group ig
     likewise with PI. Word w_base + ((og * kh + ky) * kw + kx) * cin_g + ig
     holds weight[P*(PO*og+po) + o][ky][kx][P*(PI*ig+k) + c] in byte
     ((po * PI + k) * P + o) * P + c. Words past the layer's channels are 0.
     A PAD_CONST layer has one more word after its weights: the padding
     value of each input channel in its low P bytes.
  Q  dense conv constants. One word = PO blocks, block po for output word
     PO*og+po: P int32 biases, P int32 multipliers M, P uint8 shifts S and,
     only in builds whose program uses the ADD epilogue, P int32 residual
     multipliers Mb (zero for layers without ADD). Word q_base + og.
  WD depthwise weights. One word = one kernel tap of PC channel words:
     word w_base + cg * kh * kw + ky * kw + kx holds weight[P*(PC*cg+k) + c]
     of tap (ky, kx) in byte P * k + c.
  QD depthwise constants: one word = PC blocks of P x (bias, M, S), word
     q_base + cg.

  fc   a fully connected layer (1x1 conv reading int32 buffer V, raw int32
     out) uses only the low P x P bytes of W words (block po = k = 0) and
     block 0 of Q words: word w_base + ow * cin_w + iw, Q word q_base + ow.
     Output channels are padded with zero weights to a multiple of P; only
     the biases are used.

Buffers: S0, A, B, C hold words of P channels, in NB banks. V holds words of
P int32 (GAP sums, fc input). IN holds the input image, one pixel per word
with only its own channels (3 for RGB) to save memory; the engine widens a
pixel to a P-channel word with zeros, and the weights of the missing channels
are zero.

pack() returns a copy of the program whose w_base / q_base fields are word
addresses in W / Q or WD / QD (in the reference program w_base is a parameter
index). Instructions the engine cannot run yet are returned unchanged and
listed.
"""
import dataclasses

import numpy as np

from npe.compiler import isa

P = 8              # channels per word
HW_BUFS = [isa.BUF_S0, isa.BUF_A, isa.BUF_B, isa.BUF_C]   # word buffers


def is_depthwise(I):
    return I.op == isa.OP_CONV and I.groups > 1 and I.groups == I.a_c == I.o_c


def is_fc(I):
    return (I.op == isa.OP_CONV and I.a_buf == isa.BUF_V and I.a_bits == 32
            and I.raw_out == 1)


def has_add(prog):
    return any(I.epi == isa.EPI_ADD for I in prog)


def supported(I):
    """What the engine executes: dense and depthwise conv on 8-bit data, max
    pool, GAP, and fc (int32 in, raw int32 out)."""
    if I.op == isa.OP_POOL_MAX or I.op == isa.OP_GAP or is_fc(I):
        return True
    return (I.op == isa.OP_CONV and (I.groups == 1 or is_depthwise(I))
            and I.a_bits == 8 and I.o_bits == 8 and I.w_bits == 8 and not I.raw_out)


def check(I):
    """Restrictions of the word-based hardware on a supported instruction."""
    def words(*vals):
        assert all(v % P == 0 for v in vals), f"not a multiple of {P} channels: {I}"
    if is_fc(I):
        assert I.groups == 1 and I.kh == I.kw == 1 and I.a_h == I.a_w == I.o_h == I.o_w == 1
        assert I.o_buf == isa.BUF_OUT and I.epi == isa.EPI_NONE and I.w_bits == 8
        words(I.a_c, I.a_base, I.a_off)
        assert I.a_pix in (0, I.a_c)
        return
    if I.op == isa.OP_GAP:
        assert I.a_buf in HW_BUFS and I.o_buf == isa.BUF_V and I.o_bits == 32
        words(I.a_c, I.a_pix, I.a_off, I.a_base, I.o_base, I.o_off)
        assert I.o_c == I.a_c and I.o_pix in (0, I.o_c) and I.epi == isa.EPI_NONE
        return
    if I.op == isa.OP_POOL_MAX:
        assert I.a_buf in HW_BUFS and I.o_buf in HW_BUFS and I.a_buf != I.o_buf
        assert I.a_bits == 8 and I.o_c == I.a_c and I.epi == isa.EPI_NONE
        assert max(I.pad_t, I.pad_b) < I.kh and max(I.pad_l, I.pad_r) < I.kw, \
            "a window that is all padding has no maximum"
        words(I.a_c, I.a_pix, I.a_off, I.a_base, I.o_pix, I.o_off, I.o_base)
        return
    if I.a_buf == isa.BUF_IN:
        assert I.a_c <= P and I.a_pix in (0, I.a_c) and I.a_off == 0 and I.a_base == 0
    else:
        assert I.a_buf in HW_BUFS
        words(I.a_c, I.a_pix, I.a_off, I.a_base)
    words(I.o_c, I.o_pix, I.o_off, I.o_base)
    assert I.act in (isa.ACT_NONE, isa.ACT_RELU, isa.ACT_RELU6)
    if I.act == isa.ACT_RELU6:
        assert 0 < I.act_max < 1 << I.o_bits
    assert I.o_buf in HW_BUFS
    if is_depthwise(I):
        assert I.a_buf != isa.BUF_IN, "depthwise on the image is not supported"
        assert I.pad_mode == isa.PAD_ZERO and I.epi == isa.EPI_NONE
    if I.pad_mode == isa.PAD_CONST:
        assert I.a_c <= P, "constant padding only on a one-word input (the image)"
    if I.epi in (isa.EPI_INTERLEAVE, isa.EPI_ADD):
        words(I.b_pix, I.b_off, I.b_base)
        assert I.b_buf in HW_BUFS
        assert I.b_buf != I.a_buf, "A and B operands must be in different buffers"
    if I.epi == isa.EPI_ADD:
        assert not is_depthwise(I), "ADD is an epilogue of the dense conv unit"
        if I.b_buf == I.o_buf:   # in place: only over exactly the B view
            assert (I.b_base + I.b_off, I.b_pix or I.o_c) == \
                (I.o_base + I.o_off, I.o_pix or I.o_c), "ADD output overlaps B elsewhere"
        else:
            assert I.o_buf != I.a_buf, "a plain conv cannot write the buffer it reads"
    elif I.epi != isa.EPI_INTERLEAVE:
        assert I.epi == isa.EPI_NONE
        assert I.o_buf != I.a_buf, "a plain conv cannot write the buffer it reads"


def cdiv(a, b):
    return -(-a // b)


def pack(prog, params, peak, po=1, pi=1, pc=1, rq=None, rqc=None, ppb=4):
    """Returns (hw_prog, mems, cfg, skipped).
    mems: {"W", "WD", "Q", "QD": uint8 [words, bytes per word]},
    cfg: {macro name: value}."""
    rq, rqc = rq or pc, rqc or po
    assert 1 <= rq <= pc and 1 <= rqc <= po and ppb & (ppb - 1) == 0
    in_c = {I.a_c for I in prog if I.a_buf == isa.BUF_IN and supported(I)}
    assert len(in_c) <= 1, "every layer reading the image must use all its channels"
    in_c = in_c.pop() if in_c else 1
    ppw = max(min(P // in_c, ppb), 1)          # image pixels packed per word
    nb = 2
    while nb < max(2 * po, pi, pc):
        nb *= 2
    add = has_add(prog)
    ql = 13 * P if add else 9 * P             # bytes of one Q block
    mem = {k: [] for k in ("W", "WD", "Q", "QD")}
    hw, skipped = [], []
    for n, I in enumerate(prog):
        if not supported(I):
            hw.append(I)
            skipped.append(n)
            continue
        check(I)
        if I.op in (isa.OP_POOL_MAX, isa.OP_GAP):     # no parameters
            hw.append(I)
            continue
        p = params[I.w_base]
        if is_fc(I):
            hw.append(pack_fc(I, p, mem, add, po, pi))
        elif is_depthwise(I):
            assert I.kh * I.kw * rq >= pc, \
                f"depthwise {I.kh}x{I.kw} cannot requantize {pc} words in time at RQ={rq}"
            hw.append(pack_depthwise(I, p, mem, pc))
        else:
            hw.append(pack_conv(I, p, mem, add, po, pi, ppw))

    def buf_words(elems):   # NB banks, plus NB words of slack for vector reads
        return cdiv(cdiv(elems, P) + nb, nb) * nb

    cfg = {"NPE_P": P, "NPE_PO": po, "NPE_PI": pi, "NPE_PC": pc, "NPE_RQ": rq, "NPE_RQC": rqc, "NPE_NB": nb,
           "NPE_LOG2NB": nb.bit_length() - 1, "NPE_HAS_ADD": int(add)}
    for k in mem:
        cfg[f"NPE_{k}_WORDS"] = max(len(mem[k]), 1)
    for b in HW_BUFS:
        cfg[f"NPE_BUF_{isa.BUF_NAMES[b]}_WORDS"] = buf_words(peak.get(b, 0))
    cfg["NPE_PROG_WORDS"] = max(len(prog), 1)
    cfg["NPE_OUT_MAX"] = max(sum(I.o_c for I in prog if I.o_buf == isa.BUF_OUT), 1)
    cfg["NPE_V_WORDS"] = max((peak.get(isa.BUF_V, 0) + P - 1) // P, 1)
    cfg["NPE_IN_C"] = in_c
    cfg["NPE_IN_PPB"], cfg["NPE_LOG2_IN_PPB"], cfg["NPE_IN_PPW"] = ppb, ppb.bit_length() - 1, ppw
    beat = 32
    while beat < 8 * in_c * ppb:
        beat *= 2
    cfg["NPE_IN_BEAT_BITS"] = beat             # image AXI-Stream width
    cfg["NPE_IN_PIXELS"] = max(peak.get(isa.BUF_IN, 0) // cfg["NPE_IN_C"], 1)
    width = {"W": P * P * po * pi, "WD": P * pc, "Q": ql * po, "QD": 9 * P * pc}
    mems = {k: np.frombuffer(b"".join(mem[k]), np.uint8).reshape(-1, width[k]) for k in mem}
    return hw, mems, cfg, skipped


def quant_block(p, chans, add):
    """Bytes of one quant block: bias, M, S (and Mb if add) of the output
    channels `chans` (P indices; an index past the layer gives zeros)."""
    def get(k, dtype, default=0):
        v = np.asarray(p[k] if k in p else np.zeros(1)).astype(np.int64)
        out = np.full(P, default, np.int64)
        real = [i for i, c in enumerate(chans) if c < len(v)] if k in p else []
        out[real] = v[[chans[i] for i in real]]
        return out.astype(dtype).tobytes()
    return (get("bias", "<i4") + get("M", "<i4") + get("S", np.uint8, 1)
            + (get("Mb", "<i4") if add else b""))


def check_quant(p):
    M, S, bias = (np.asarray(p[k]).astype(np.int64) for k in ("M", "S", "bias"))
    assert M.min() >= 0 and M.max() < 2 ** 31 and S.min() >= 1 and S.max() <= 63
    assert np.abs(bias).max() < 2 ** 31
    if "Mb" in p:
        assert np.abs(np.asarray(p["Mb"]).astype(np.int64)).max() < 2 ** 31


def pack_conv(I, p, mem, add, po, pi, ppw=1):
    """An image layer (reading IN) packs ppw pixels of a kernel row into one
    input word: lane s * a_c + c holds channel c of kernel column
    kx * ppw + s, so a kernel row takes kw_c = ceil(kw / ppw) words."""
    check_quant(p)
    cin_w = cdiv(I.a_c, P)
    cout_w = I.o_c // P
    cin_g, cout_g = cdiv(cin_w, pi), cdiv(cout_w, po)
    wt = np.asarray(p["w"]).reshape(I.o_c, I.a_c, I.kh, I.kw).transpose(0, 2, 3, 1)  # [o][ky][kx][c]
    if I.a_buf == isa.BUF_IN:
        kw_c = cdiv(I.kw, ppw)
        w = np.zeros((cout_g * po * P, I.kh, kw_c, cin_g * pi * P), np.int64)
        for kx in range(I.kw):
            s = kx % ppw
            w[:I.o_c, :, kx // ppw, s * I.a_c:(s + 1) * I.a_c] = wt[:, :, kx, :]
    else:
        kw_c = I.kw
        w = np.zeros((cout_g * po * P, I.kh, I.kw, cin_g * pi * P), np.int64)  # [cout][ky][kx][cin]
        w[:I.o_c, :, :, :I.a_c] = wt
    assert w.min() >= -128 and w.max() <= 127
    w_base, q_base = len(mem["W"]), len(mem["Q"])
    for og in range(cout_g):
        for ky in range(I.kh):
            for kx in range(kw_c):
                for ig in range(cin_g):
                    # [po][o][k][c] -> byte ((po * PI + k) * P + o) * P + c
                    blk = w[og * po * P:(og + 1) * po * P, ky, kx, ig * pi * P:(ig + 1) * pi * P]
                    blk = blk.reshape(po, P, pi, P).transpose(0, 2, 1, 3)
                    mem["W"].append(blk.astype(np.int8).tobytes())
        mem["Q"].append(b"".join(quant_block(p, range(P * (po * og + j), P * (po * og + j + 1)), add)
                                 for j in range(po)))
    if I.pad_mode == isa.PAD_CONST:
        pad = np.zeros(P * P * po * pi, np.uint8)
        pad[:I.a_c] = np.broadcast_to(np.asarray(p["pad"]), I.a_c) & 0xFF
        mem["W"].append(pad.tobytes())
    assert len(mem["W"]) <= 1 << 16 and q_base < 1 << 16
    return dataclasses.replace(I, w_base=w_base, q_base=q_base)


def pack_fc(I, p, mem, add, po, pi):
    cout_p = cdiv(I.o_c, P) * P                            # padded to words
    w = np.zeros((cout_p, I.a_c), np.int64)
    w[:I.o_c] = np.asarray(p["w"]).reshape(I.o_c, I.a_c)
    assert w.min() >= -128 and w.max() <= 127
    assert np.abs(np.asarray(p["bias"]).astype(np.int64)).max() < 2 ** 31
    w_base, q_base = len(mem["W"]), len(mem["Q"])
    wbytes, qbytes = P * P * po * pi, (13 if add else 9) * P * po
    for ob in range(cout_p // P):
        for iw in range(I.a_c // P):
            blk = w[P * ob:P * ob + P, P * iw:P * iw + P].astype(np.int8).tobytes()
            mem["W"].append(blk + bytes(wbytes - P * P))
        q = quant_block(dict(bias=np.asarray(p["bias"])), range(P * ob, P * ob + P), add)
        mem["Q"].append(q + bytes(qbytes - len(q)))
    assert len(mem["W"]) <= 1 << 16 and q_base < 1 << 16
    return dataclasses.replace(I, w_base=w_base, q_base=q_base)


def pack_depthwise(I, p, mem, pc):
    check_quant(p)
    ch_g = cdiv(I.o_c // P, pc)
    w = np.zeros((ch_g * pc * P, I.kh, I.kw), np.int64)   # one kernel per channel
    w[:I.o_c] = np.asarray(p["w"]).reshape(I.o_c, I.kh, I.kw)
    assert w.min() >= -128 and w.max() <= 127
    w_base, q_base = len(mem["WD"]), len(mem["QD"])
    for cg in range(ch_g):
        ch = slice(cg * pc * P, (cg + 1) * pc * P)
        for ky in range(I.kh):
            for kx in range(I.kw):
                mem["WD"].append(w[ch, ky, kx].astype(np.int8).tobytes())
        mem["QD"].append(b"".join(quant_block(p, range(P * (pc * cg + j), P * (pc * cg + j + 1)), False)
                                  for j in range(pc)))
    assert len(mem["WD"]) <= 1 << 16 and q_base < 1 << 16
    return dataclasses.replace(I, w_base=w_base, q_base=q_base)


def config_header(cfg):
    lines = ["// Generated by npe/compiler/pack.py. Do not edit.",
             "#ifndef NPE_CONFIG_H_", "#define NPE_CONFIG_H_", ""]
    lines += [f"#define {k} {v}" for k, v in cfg.items()]
    lines += ["", "#endif // NPE_CONFIG_H_", ""]
    return "\n".join(lines)
