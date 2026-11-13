"""The HLS engine (compiled with g++) vs the reference interpreter.

Two suites, each packed and compiled on its own:
  shufflenet  every instruction the engine supports, on one golden image,
              starting from the buffer state the reference had at that point
  random      random dense and depthwise convolutions: kernel sizes 1-5 (also non-square),
              strides, asymmetric zero padding, signed / unsigned inputs,
              channel-offset views, plain and INTERLEAVE (also in place),
              and image-input layers with constant padding

  python3 -m npe.tests.test_engine [seed]
"""
import os
import struct
import subprocess
import sys

import numpy as np

from npe.compiler import isa, pack
from npe.compiler.isa import Instr
from npe.compiler.lower_shufflenet import OLD, ROOT, lower
from npe.ref.isa_ref import Machine

NPE = os.path.join(ROOT, "npe")
BUILD = os.path.join(NPE, "build")
APDIR = os.path.join(BUILD, "hlsap")


def image(m, cfg, b):
    """Byte image of reference buffer b, padded to the hardware buffer size."""
    n = cfg[f"NPE_BUF_{isa.BUF_NAMES[b]}_WORDS"] * pack.P
    out = np.zeros(n, np.uint8)
    out[:len(m.buf[b])] = m.buf[b] & 0xFF
    return out.tobytes()


def run_suite(name, prog, params, peak, inp=None, state=None):
    """Steps the reference through prog; every supported instruction becomes a
    case. state(m) may randomize the buffers before each instruction."""
    hw, mems, cfg, skipped = pack.pack(prog, params, peak)
    d = os.path.join(BUILD, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "npe_config.h"), "w") as f:
        f.write(pack.config_header(cfg))
    sizes = dict(peak)
    for b in pack.HW_BUFS:
        sizes[b] = cfg[f"NPE_BUF_{isa.BUF_NAMES[b]}_WORDS"] * pack.P
    m = Machine(params, sizes)
    if inp is not None:
        m.write(isa.BUF_IN, 0, inp.astype(np.int64), 0, 0)
    cases = []
    for n, I in enumerate(prog):
        if state:
            state(m)
        before = b"".join(image(m, cfg, b) for b in pack.HW_BUFS)
        m.step(I)
        if n not in skipped:
            after = b"".join(image(m, cfg, b) for b in pack.HW_BUFS)
            cases.append(isa.encode([hw[n]]) + before + after)
    path = os.path.join(d, "cases.bin")
    img = np.zeros(cfg["NPE_IN_PIXELS"] * cfg["NPE_IN_C"], np.uint8)
    if isa.BUF_IN in m.buf:
        n = min(len(img), len(m.buf[isa.BUF_IN]))
        img[:n] = m.buf[isa.BUF_IN][:n] & 0xFF
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(cases)) + mems["W"].tobytes() + mems["WD"].tobytes() + mems["Q"].tobytes()
                + img.tobytes()
                + b"".join(cases))
    if not os.path.exists(os.path.join(APDIR, "include", "ap_int.h")):
        subprocess.check_call(["git", "clone", "-q", "--depth", "1",
                               "https://github.com/Xilinx/HLS_arbitrary_Precision_Types", APDIR])
    exe = os.path.join(d, "tb_engine")
    subprocess.check_call(["g++", "-std=c++14", "-O2", "-w", f"-I{d}", f"-I{NPE}/hls",
                           f"-I{NPE}/isa", f"-I{APDIR}/include", "-o", exe,
                           f"{NPE}/tb/tb_engine.cpp"])
    r = subprocess.run([exe, path], capture_output=True, text=True)
    print(f"[{name}] {len(prog)} instructions, {len(skipped)} not supported yet; "
          f"{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'no output'}")
    if r.returncode:
        print(r.stdout)
    return r.returncode


def shufflenet():
    from quantize import read_bin  # noqa: E402
    layers, pad, _ = read_bin(os.path.join(OLD, "data", "weights.bin"))
    prog, params, names, peak = lower(layers, pad)
    root = os.path.join(OLD, "data", "golden")
    img = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))[0]
    x = np.fromfile(os.path.join(root, img, "input.bin"), np.uint8).reshape(224, 224, 3)
    return run_suite("shufflenet", prog, params, peak, inp=x)


def random_suite(seed, count=80):
    rng = np.random.default_rng(seed)
    P = pack.P
    prog, params, peak = [], [], {}
    bufs = pack.HW_BUFS
    in_c = 3
    r = lambda lo, hi: int(rng.integers(lo, hi + 1))   # noqa: E731
    for n in range(count):
        image = n % 4 == 0                  # every 4th case reads the RGB image
        dw = n % 4 == 1 or n % 4 == 2       # half the cases are depthwise
        cin = in_c if image else P * r(1, 3)
        cout = cin if dw else P * r(1, 3)
        kh, kw = r(1, 5), r(1, 5)
        sh, sw = r(1, 2), r(1, 2)
        pt, pb, pl, pr = (r(0, k - 1) for k in (kh, kh, kw, kw))
        oh, ow = r(1, 5), r(1, 5)
        ah = max((oh - 1) * sh + kh - pt - pb, 1)
        aw = max((ow - 1) * sw + kw - pl - pr, 1)
        oh = (ah + pt + pb - kh) // sh + 1      # recompute: a_h >= 1 may change it
        ow = (aw + pl + pr - kw) // sw + 1
        if image:
            ah, aw = 12, 12                     # one fixed image for all cases
            oh = (ah + pt + pb - kh) // sh + 1
            ow = (aw + pl + pr - kw) // sw + 1
            a_buf, a_base, a_off, a_pix, a_signed = isa.BUF_IN, 0, 0, 0, 0
            o_buf, b_buf = (int(b) for b in rng.permutation(bufs)[:2])
        else:
            a_buf, o_buf, b_buf = (int(b) for b in rng.permutation(bufs)[:3])
            a_base, a_off = P * r(0, 2), P * r(0, 2)
            a_pix = a_off + cin + P * r(0, 2)
            a_signed = r(0, 1)
        relu = bool(r(0, 1))
        I = Instr(op=isa.OP_CONV, act=isa.ACT_RELU if relu else isa.ACT_NONE,
                  kh=kh, kw=kw, sh=sh, sw=sw, pad_t=pt, pad_l=pl, pad_b=pb, pad_r=pr,
                  pad_mode=isa.PAD_CONST if image and r(0, 1) else isa.PAD_ZERO,
                  a_buf=a_buf, a_base=a_base, a_h=ah, a_w=aw, a_c=cin, a_pix=a_pix,
                  a_off=a_off, a_signed=a_signed, o_buf=o_buf, o_base=P * r(0, 2),
                  o_h=oh, o_w=ow, o_c=cout, o_signed=0 if relu else r(0, 1),
                  groups=cin if dw else 1, w_base=len(params))
        if r(0, 1) and not dw:
            I.epi = isa.EPI_INTERLEAVE
            if r(0, 1):      # in place: B is the first half of the output pixel
                I.b_buf, I.b_base, I.b_pix, I.b_off = o_buf, I.o_base, 2 * cout, 0
            else:
                I.b_buf, I.b_base = b_buf, P * r(0, 2)
                I.b_pix = cout + P * r(0, 1)
                I.o_pix = 2 * cout + P * r(0, 1)
        else:
            I.o_off = P * r(0, 1)
            I.o_pix = I.o_off + cout + P * r(0, 1)
        params.append(dict(w=rng.integers(-128, 128, (cout, 1 if dw else cin, kh, kw)),
                           bias=rng.integers(-2 ** 15, 2 ** 15, cout),
                           M=rng.integers(1, 2 ** 31, cout),
                           S=rng.integers(36, 45, cout) if dw else rng.integers(34, 46, cout),
                           pad=rng.integers(0, 256, cin)))
        prog.append(I)
        uses = [(I.o_buf, I.o_base + oh * ow * (I.o_pix or I.out_channels())),
                (I.b_buf, I.b_base + oh * ow * (I.b_pix or cout))]
        if not image:
            uses.append((a_buf, a_base + ah * aw * a_pix))
        for buf, size in uses:
            peak[buf] = max(peak.get(buf, 0), size)
    peak[isa.BUF_IN] = 12 * 12 * in_c
    image = rng.integers(0, 256, (12, 12, in_c))
    it = iter(prog)

    def state(m):   # fresh random bytes before each instruction, signed if A is
        I = next(it)
        for b in bufs:
            m.buf[b][:] = rng.integers(0, 256, len(m.buf[b]))
        if I.a_signed:
            v = m.buf[I.a_buf]
            v[v > 127] -= 256
    return run_suite("random", prog, params, peak, inp=image, state=state)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    bad = shufflenet() | random_suite(seed)
    print("TEST FAILED" if bad else "TEST PASSED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
