"""Lowers the quantized ShuffleNetV2 (weights.bin from the old flow) to the
generic NPE instruction set.

This is a stand-in for the ONNX importer (plan step 3): it proves that the
generic instruction set expresses the whole network, stem and head included,
with nothing ShuffleNet-specific in the ops.

  python3 -m npe.compiler.lower_shufflenet [--bin weights.bin] [--list]
"""
import argparse
import os
import sys

import numpy as np

from npe.compiler import isa
from npe.compiler.isa import Instr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OLD = os.path.join(ROOT, "older_shufflnet_work")
sys.path.insert(0, os.path.join(OLD, "golden"))


def lower(layers, pad_value, img=224):
    from model_def import STAGE_OUT, units  # noqa: E402
    L = {l["name"]: l for l in layers}
    prog, params, names = [], [], []
    peak = {}

    def param(name, **extra):
        l = L[name]
        cout = l["cout"]
        params.append(dict(w=l["w"].reshape(cout, -1, l["k"], l["k"]), bias=l["bias"],
                           M=l["M"], S=l["S"], **extra))
        return len(params) - 1

    def emit(name, ins):
        prog.append(ins)
        names.append(name)
        oc = ins.out_channels()
        need = ins.o_base + ins.o_h * ins.o_w * (ins.o_pix or oc)
        peak[ins.o_buf] = max(peak.get(ins.o_buf, 0), need)

    def conv(name, a_buf, o_buf, h, cin, *, k=1, s=1, groups=1, a_off=0, a_pix=0,
             a_signed=0, o_pix=0, epi=isa.EPI_NONE, b_buf=isa.BUF_A, b_pix=0, b_off=0,
             pad_mode=isa.PAD_ZERO, extra=None):
        l = L[name]
        p = k // 2
        ho = (h + 2 * p - k) // s + 1
        relu = l["relu"]
        emit(name, Instr(
            op=isa.OP_CONV, epi=epi, act=isa.ACT_RELU if relu else isa.ACT_NONE,
            kh=k, kw=k, sh=s, sw=s, pad_t=p, pad_l=p, pad_b=p, pad_r=p, pad_mode=pad_mode,
            groups=groups, a_buf=a_buf, a_h=h, a_w=h, a_c=cin, a_pix=a_pix, a_off=a_off,
            a_signed=a_signed, b_buf=b_buf, b_pix=b_pix, b_off=b_off,
            o_buf=o_buf, o_h=ho, o_w=ho, o_c=l["cout"], o_pix=o_pix, o_signed=0 if relu else 1,
            w_base=param(name, **(extra or {}))))
        return ho

    # Stem: conv1 3x3 s2 with the folded input mean as padding, then maxpool.
    h = conv("conv1", isa.BUF_IN, isa.BUF_S0, img, 3, k=3, s=2, pad_mode=isa.PAD_CONST,
             extra=dict(pad=pad_value.astype(np.int64)))
    c = L["conv1"]["cout"]
    ho = (h + 2 - 3) // 2 + 1
    emit("maxpool", Instr(op=isa.OP_POOL_MAX, kh=3, kw=3, sh=2, sw=2, pad_t=1, pad_l=1,
                          pad_b=1, pad_r=1, a_buf=isa.BUF_S0, a_h=h, a_w=h, a_c=c,
                          o_buf=isa.BUF_A, o_h=ho, o_w=ho, o_c=c))
    h = ho

    # Units. A holds the unit input and output (in place), B the pw1 output,
    # C the depthwise output.
    for prefix, down in units():
        cout = STAGE_OUT[int(prefix[1]) - 1]
        half = cout // 2
        if down:
            conv(f"{prefix}.b1.dw", isa.BUF_A, isa.BUF_C, h, c, k=3, s=2, groups=c)
            conv(f"{prefix}.pw1", isa.BUF_A, isa.BUF_B, h, c)
            ho = (h + 2 - 3) // 2 + 1
            # branch 1 into the first half of each output pixel of A
            conv(f"{prefix}.b1.pw", isa.BUF_C, isa.BUF_A, ho, c, a_signed=1, o_pix=cout)
            conv(f"{prefix}.dw", isa.BUF_B, isa.BUF_C, h, half, k=3, s=2, groups=half)
            h = ho
        else:
            conv(f"{prefix}.pw1", isa.BUF_A, isa.BUF_B, h, half, a_off=half, a_pix=cout)
            conv(f"{prefix}.dw", isa.BUF_B, isa.BUF_C, h, half, k=3, groups=half)
        # pw2 + concat + shuffle in one instruction, in place over A
        conv(f"{prefix}.pw2", isa.BUF_C, isa.BUF_A, h, half, a_signed=1, o_pix=cout,
             epi=isa.EPI_INTERLEAVE, b_buf=isa.BUF_A, b_pix=cout, b_off=0)
        names[-1] = f"{prefix}.out"
        c = cout

    # Head: conv5, global average pool (int32 sums into V), fc with raw int32
    # logits.
    conv("conv5", isa.BUF_A, isa.BUF_B, h, c)
    c5 = L["conv5"]["cout"]
    emit("gap", Instr(op=isa.OP_GAP, a_buf=isa.BUF_B, a_h=h, a_w=h, a_c=c5,
                      o_buf=isa.BUF_V, o_c=c5, o_bits=32, o_signed=1))
    fc = L["fc"]
    params.append(dict(w=fc["w"].reshape(fc["cout"], -1, 1, 1), bias=fc["bias"],
                       M=fc["M"], S=fc["S"]))
    emit("logits", Instr(op=isa.OP_CONV, a_buf=isa.BUF_V, a_c=c5, a_bits=32, a_signed=1,
                         o_buf=isa.BUF_OUT, o_c=fc["cout"], raw_out=1, o_bits=32,
                         o_signed=1, w_base=len(params) - 1))
    peak[isa.BUF_IN] = img * img * 3
    return prog, params, names, peak


def main():
    from quantize import read_bin  # noqa: E402
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(OLD, "data", "weights.bin"))
    ap.add_argument("--list", action="store_true", help="print the program")
    a = ap.parse_args()
    layers, pad, _ = read_bin(a.bin)
    prog, params, names, peak = lower(layers, pad)
    print(f"{len(prog)} instructions, {len(isa.FIELDS)} fields each, "
          f"{len(isa.encode(prog))} bytes")
    print("buffer peaks (elements):", {isa.BUF_NAMES[b]: n for b, n in sorted(peak.items())})
    if a.list:
        for n, I in zip(names, prog):
            print(f"{n:12s} op={I.op} epi={I.epi} k={I.kh}x{I.kw} s={I.sh} g={I.groups} "
                  f"{isa.BUF_NAMES[I.a_buf]}{I.a_h}x{I.a_w}x{I.a_c}+{I.a_off} -> "
                  f"{isa.BUF_NAMES[I.o_buf]}{I.o_h}x{I.o_w}x{I.out_channels()}")


if __name__ == "__main__":
    main()
