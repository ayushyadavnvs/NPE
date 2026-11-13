"""A MobileNetV2-shaped test network on the generic instruction set.

There is no trained, quantized MobileNetV2 in the repo yet (that comes with
the ONNX importer), so this builds the real MobileNetV2 layer structure with
random int8 weights. Every layer's requant constants are calibrated on a
sample image so activations use the int8 range instead of saturating. The
reference interpreter is the ground truth the engine is checked against.

Structure (Sandler et al. 2018): conv 3x3 s2 + ReLU6, 17 inverted-residual
blocks (expand 1x1 + ReLU6, depthwise 3x3 + ReLU6, project 1x1 linear, plus
the block input when stride is 1 and channels match: the fused ADD
epilogue), conv 1x1 to 1280 + ReLU6, GAP, fc.

  python3 -m npe.compiler.lower_mobilenetv2 [--list]
"""
import argparse
import dataclasses

import numpy as np

from npe.compiler import isa
from npe.compiler.isa import Instr
from npe.ref.isa_ref import Machine

# (expansion t, output channels c, repeats n, first stride s)
BLOCKS = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
          (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


def divisible(v, d=8):
    """Channel rounding of the reference implementation (multiples of 8)."""
    n = max(d, int(v + d / 2) // d * d)
    return n + d if n < 0.9 * v else n


def build(width=0.5, img=96, classes=10, seed=0):
    """Returns (prog, params, names, peak, sample image)."""
    rng = np.random.default_rng(seed)
    prog, params, names, peak = [], [], [], {}
    sample = rng.integers(0, 256, (img, img, 3))
    big = 1 << 22
    m = Machine(params, {b: np.zeros(0) for b in ()})    # sized below
    m.buf = {b: np.zeros(big, np.int64) for b in range(len(isa.BUF_NAMES))}
    m.write(isa.BUF_IN, 0, sample, 0, 0)

    def calibrate(I, p, target):
        """Per-channel M, S so that |out| reaches about `target` on the sample
        (99.9th percentile); for ADD also Mb (the residual at ~0.7 weight)."""
        a = m.read(I.a_buf, I.a_base, I.a_h, I.a_w, I.a_c, I.a_pix, I.a_off)
        acc = m._conv(a, dataclasses.replace(I, raw_out=1, epi=isa.EPI_NONE))
        top = np.maximum(np.percentile(np.abs(acc), 99.9, axis=(0, 1)), 1.0)
        ratio = target / top
        # S <= 30 on ADD layers keeps Mb (about 0.7 * 2^S) inside int32
        S = np.clip(np.floor(np.log2(2 ** 30 / ratio)), 1,
                    30 if I.epi == isa.EPI_ADD else 40).astype(np.int64)
        p["M"] = np.round(ratio * 2.0 ** S).astype(np.int64)
        p["S"] = S
        if I.epi == isa.EPI_ADD:
            p["Mb"] = np.round(rng.uniform(0.5, 0.9, len(S)) * 2.0 ** S).astype(np.int64)

    def conv(name, a_buf, o_buf, h, cin, cout, *, k=1, s=1, groups=1, act=isa.ACT_RELU6,
             a_signed=0, epi=isa.EPI_NONE, pad_const=False):
        p = k // 2
        ho = (h + 2 * p - k) // s + 1
        relu = act != isa.ACT_NONE
        I = Instr(op=isa.OP_CONV, epi=epi, act=act, kh=k, kw=k, sh=s, sw=s,
                  pad_t=p, pad_l=p, pad_b=p, pad_r=p, groups=groups,
                  pad_mode=isa.PAD_CONST if pad_const else isa.PAD_ZERO,
                  a_buf=a_buf, a_h=h, a_w=h, a_c=cin, a_signed=a_signed,
                  o_buf=o_buf, o_h=ho, o_w=ho, o_c=cout, o_signed=0 if relu else 1,
                  w_base=len(params))
        if act == isa.ACT_RELU6:
            I.act_max = int(rng.integers(60, 140))        # quantized 6.0: some clipping
        if epi == isa.EPI_ADD:   # in place over the block input
            I.b_buf, I.b_signed = o_buf, 1
        w = np.clip(np.round(rng.normal(0, 40, (cout, cin // groups, k, k))), -127, 127)
        par = dict(w=w.astype(np.int64), bias=rng.integers(-2000, 2000, cout))
        if pad_const:
            par["pad"] = rng.integers(0, 256, cin)
        params.append(par)
        calibrate(I, par, 60 if epi == isa.EPI_ADD else 110)
        emit(name, I)
        return ho

    def emit(name, I):
        prog.append(I)
        names.append(name)
        m.step(I)
        need = I.o_base + I.o_h * I.o_w * (I.o_pix or I.out_channels())
        peak[I.o_buf] = max(peak.get(I.o_buf, 0), need)
        if I.a_buf != isa.BUF_IN:
            need = I.a_base + I.a_h * I.a_w * (I.a_pix or I.a_c)
            peak[I.a_buf] = max(peak.get(I.a_buf, 0), need)

    c = divisible(32 * width)
    h = conv("conv1", isa.BUF_IN, isa.BUF_S0, img, 3, c, k=3, s=2, pad_const=True)
    src, signed = isa.BUF_S0, 0      # where the block input is, and its type
    n_block = 0
    for t, cb, n, s0 in BLOCKS:
        cout = divisible(cb * width)
        for i in range(n):
            s = s0 if i == 0 else 1
            pre = f"b{n_block}"
            hid = c * t
            if t != 1:
                conv(f"{pre}.expand", src, isa.BUF_B, h, c, hid, a_signed=signed)
                dw_in, dw_signed = isa.BUF_B, 0
            else:
                dw_in, dw_signed = src, signed
            ho = conv(f"{pre}.dw", dw_in, isa.BUF_C, h, hid, hid, k=3, s=s, groups=hid,
                      a_signed=dw_signed)
            residual = s == 1 and c == cout
            assert not residual or src == isa.BUF_A
            conv(f"{pre}.project", isa.BUF_C, isa.BUF_A, ho, hid, cout, act=isa.ACT_NONE,
                 epi=isa.EPI_ADD if residual else isa.EPI_NONE)
            src, signed, c, h = isa.BUF_A, 1, cout, ho
            n_block += 1

    last = 1280 if width <= 1.0 else divisible(1280 * width)
    conv("conv_last", isa.BUF_A, isa.BUF_B, h, c, last, a_signed=1)
    emit("gap", Instr(op=isa.OP_GAP, a_buf=isa.BUF_B, a_h=h, a_w=h, a_c=last,
                      o_buf=isa.BUF_V, o_c=last, o_bits=32, o_signed=1))
    w = rng.integers(-127, 128, (classes, last, 1, 1))
    params.append(dict(w=w, bias=rng.integers(-5000, 5000, classes),
                       M=np.zeros(classes, np.int64), S=np.ones(classes, np.int64)))
    emit("logits", Instr(op=isa.OP_CONV, a_buf=isa.BUF_V, a_c=last, a_bits=32, a_signed=1,
                         o_buf=isa.BUF_OUT, o_c=classes, raw_out=1, o_bits=32, o_signed=1,
                         w_base=len(params) - 1))
    peak[isa.BUF_IN] = img * img * 3
    return prog, params, names, peak, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    prog, params, names, peak, _ = build()
    n_add = sum(I.epi == isa.EPI_ADD for I in prog)
    print(f"{len(prog)} instructions ({n_add} with the ADD epilogue), "
          f"{sum(np.asarray(p['w']).size for p in params) / 1e6:.2f}M weights")
    print("buffer peaks (elements):", {isa.BUF_NAMES[b]: n for b, n in sorted(peak.items())})
    if a.list:
        for n, I in zip(names, prog):
            print(f"{n:14s} op={I.op} epi={I.epi} act={I.act} k={I.kh} s={I.sh} g={I.groups} "
                  f"{isa.BUF_NAMES[I.a_buf]}{I.a_h}x{I.a_w}x{I.a_c} -> "
                  f"{isa.BUF_NAMES[I.o_buf]}{I.o_h}x{I.o_w}x{I.out_channels()}")


if __name__ == "__main__":
    main()
