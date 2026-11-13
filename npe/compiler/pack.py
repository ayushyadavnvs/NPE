"""Packs a program and its parameters into the memory images of the HLS engine.

The reference interpreter works on elements and parameter dicts. The engine
works on words of P channels and two flat memories:

  W  weights. One word = P x P int8: P output channels x P input channels.
     Dense conv layer: word w_base + ob * cin_w + iw holds
     weight[P * ob + o][P * iw + i] in byte P * o + i   (ob: output word,
     iw: input word, cin_w = input channels / P). With a kh x kw kernel the
     order is output word, kernel position (row by row), input word:
     word w_base + (ob * kh * kw + ky * kw + kx) * cin_w + iw.
     A PAD_CONST layer has one more word after its weights: the padding
     value of each input channel in its low P bytes.
  Q  per-output-channel constants. One word = P lanes, laid out as P int32
     biases, then P int32 multipliers M, then P uint8 shifts S.
     Word q_base + ob serves output channels P * ob .. P * ob + P - 1.

Buffers: S0, A, B, C hold words of P channels. IN holds the input image, one
pixel per word with only its own channels (3 for RGB) to save memory; the
engine widens a pixel to a P-channel word with zeros, and the weights of the
missing channels are zero.

  WD depthwise weights. One word = P int8: one kernel tap of P channels.
     Word w_base + cw * kh * kw + ky * kw + kx holds weight[P * cw + c] of
     tap (ky, kx) in byte c (cw: channel word). Its biases and multipliers
     are in Q like a dense layer's.

pack() returns a copy of the program whose w_base / q_base fields are word
addresses in W / Q (in the reference program w_base is a parameter index).
Instructions the engine cannot run yet are returned unchanged and listed.
"""
import dataclasses

import numpy as np

from npe.compiler import isa

P = 8              # channels per word
W_BYTES = P * P    # one weight word
Q_BYTES = 9 * P    # one quant word: P * (4 + 4 + 1) bytes
HW_BUFS = [isa.BUF_S0, isa.BUF_A, isa.BUF_B, isa.BUF_C]   # word buffers


def is_depthwise(I):
    return I.op == isa.OP_CONV and I.groups > 1 and I.groups == I.a_c == I.o_c


def supported(I):
    """What the engine executes at this step: dense and depthwise conv on
    8-bit data."""
    return (I.op == isa.OP_CONV and (I.groups == 1 or is_depthwise(I))
            and I.a_bits == 8 and I.o_bits == 8 and I.w_bits == 8 and not I.raw_out)


def check(I):
    """Restrictions of the word-based hardware on a supported instruction."""
    def words(*vals):
        assert all(v % P == 0 for v in vals), f"not a multiple of {P} channels: {I}"
    if I.a_buf == isa.BUF_IN:
        assert I.a_c <= P and I.a_pix in (0, I.a_c) and I.a_off == 0 and I.a_base == 0
    else:
        assert I.a_buf in HW_BUFS
        words(I.a_c, I.a_pix, I.a_off, I.a_base)
    words(I.o_c, I.o_pix, I.o_off, I.o_base)
    assert I.act in (isa.ACT_NONE, isa.ACT_RELU), "ReLU6 comes with MobileNetV2"
    assert I.o_buf in HW_BUFS
    if is_depthwise(I):
        assert I.a_buf != isa.BUF_IN, "depthwise on the image is not supported"
        assert I.pad_mode == isa.PAD_ZERO and I.epi == isa.EPI_NONE
    if I.pad_mode == isa.PAD_CONST:
        assert I.a_c <= P, "constant padding only on a one-word input (the image)"
    if I.epi == isa.EPI_INTERLEAVE:
        words(I.b_pix, I.b_off, I.b_base)
        assert I.b_buf in HW_BUFS
        assert I.b_buf != I.a_buf, "A and B operands must be in different buffers"
    else:
        assert I.epi == isa.EPI_NONE
        assert I.o_buf != I.a_buf, "a plain conv cannot write the buffer it reads"


def pack(prog, params, peak):
    """Returns (hw_prog, mems, cfg, skipped).
    mems: {"W": uint8 [n, W_BYTES], "WD": uint8 [n, P], "Q": uint8 [n, Q_BYTES]},
    cfg: {macro name: value}."""
    W, WD, Q, hw, skipped = [], [], [], [], []
    for n, I in enumerate(prog):
        if not supported(I):
            hw.append(I)
            skipped.append(n)
            continue
        check(I)
        p = params[I.w_base]
        if is_depthwise(I):
            hw.append(pack_depthwise(I, p, WD, Q))
            continue
        cin_w = (I.a_c + P - 1) // P
        w = np.zeros((I.o_c, I.kh, I.kw, cin_w * P), np.int64)  # [cout][ky][kx][cin]
        w[..., :I.a_c] = np.asarray(p["w"]).reshape(I.o_c, I.a_c, I.kh, I.kw).transpose(0, 2, 3, 1)
        assert w.min() >= -128 and w.max() <= 127
        w_base, q_base = len(W), len(Q)
        for ob in range(I.o_c // P):
            ch = slice(P * ob, P * ob + P)
            for ky in range(I.kh):
                for kx in range(I.kw):
                    for iw in range(cin_w):
                        W.append(w[ch, ky, kx, P * iw:P * iw + P].astype(np.int8).tobytes())
        quant(p, I.o_c, Q)
        if I.pad_mode == isa.PAD_CONST:
            pad = np.zeros(W_BYTES, np.uint8)
            pad[:I.a_c] = np.broadcast_to(np.asarray(p["pad"]), I.a_c) & 0xFF
            W.append(pad.tobytes())
        assert w_base < 1 << 16 and q_base < 1 << 16
        hw.append(dataclasses.replace(I, w_base=w_base, q_base=q_base))

    def even_words(elems):   # buffers are two banks, so an even word count
        n = max((elems + P - 1) // P, 2)
        return n + (n & 1)

    cfg = {"NPE_P": P, "NPE_W_WORDS": max(len(W), 1), "NPE_WD_WORDS": max(len(WD), 1),
           "NPE_Q_WORDS": max(len(Q), 1)}
    for b in HW_BUFS:
        cfg[f"NPE_BUF_{isa.BUF_NAMES[b]}_WORDS"] = even_words(peak.get(b, 0))
    in_c = {I.a_c for n, I in enumerate(prog) if I.a_buf == isa.BUF_IN and n not in skipped}
    assert len(in_c) <= 1, "every layer reading the image must use all its channels"
    cfg["NPE_IN_C"] = in_c.pop() if in_c else 1
    cfg["NPE_IN_PIXELS"] = max(peak.get(isa.BUF_IN, 0) // cfg["NPE_IN_C"], 1)
    mems = {"W": np.frombuffer(b"".join(W), np.uint8).reshape(-1, W_BYTES),
            "WD": np.frombuffer(b"".join(WD), np.uint8).reshape(-1, P),
            "Q": np.frombuffer(b"".join(Q), np.uint8).reshape(-1, Q_BYTES)}
    return hw, mems, cfg, skipped


def quant(p, cout, Q):
    """Appends the Q words of one layer: bias, M, S of each output channel."""
    M, S, bias = (np.asarray(p[k]).astype(np.int64) for k in ("M", "S", "bias"))
    assert M.min() >= 0 and M.max() < 2 ** 31 and S.min() >= 1 and S.max() <= 63
    assert np.abs(bias).max() < 2 ** 31
    for ob in range(cout // P):
        ch = slice(P * ob, P * ob + P)
        Q.append(bias[ch].astype("<i4").tobytes() + M[ch].astype("<i4").tobytes()
                 + S[ch].astype(np.uint8).tobytes())


def pack_depthwise(I, p, WD, Q):
    w = np.asarray(p["w"]).reshape(I.o_c, I.kh, I.kw)      # one kernel per channel
    assert w.min() >= -128 and w.max() <= 127
    w_base, q_base = len(WD), len(Q)
    for cw in range(I.o_c // P):
        for ky in range(I.kh):
            for kx in range(I.kw):
                WD.append(w[P * cw:P * cw + P, ky, kx].astype(np.int8).tobytes())
    quant(p, I.o_c, Q)
    assert len(WD) < 1 << 16 and q_base < 1 << 16
    return dataclasses.replace(I, w_base=w_base, q_base=q_base)


def config_header(cfg):
    lines = ["// Generated by npe/compiler/pack.py. Do not edit.",
             "#ifndef NPE_CONFIG_H_", "#define NPE_CONFIG_H_", ""]
    lines += [f"#define {k} {v}" for k, v in cfg.items()]
    lines += ["", "#endif // NPE_CONFIG_H_", ""]
    return "\n".join(lines)
