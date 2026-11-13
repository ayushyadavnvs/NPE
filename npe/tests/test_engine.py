"""The HLS engine (compiled with g++) vs the reference interpreter.

Two suites, each packed and compiled on its own:
  shufflenet  every instruction the engine supports, on one golden image,
              starting from the buffer state the reference had at that point
  random      random 1x1 convolutions: shapes, strides, signed / unsigned,
              channel-offset views, plain and INTERLEAVE (also in place)

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
    hw, W, Q, cfg, skipped = pack.pack(prog, params, peak)
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
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(cases)) + W.tobytes() + Q.tobytes() + b"".join(cases))
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


def random_suite(seed, count=60):
    rng = np.random.default_rng(seed)
    P = pack.P
    prog, params, peak = [], [], {}
    bufs = pack.HW_BUFS
    for _ in range(count):
        cin, cout = (P * int(rng.integers(1, 5)) for _ in range(2))
        sh, sw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        oh, ow = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        ah, aw = (oh - 1) * sh + 1, (ow - 1) * sw + 1   # 1x1, no padding
        a_off = P * int(rng.integers(0, 3))
        a_pix = a_off + cin + P * int(rng.integers(0, 3))
        relu = bool(rng.integers(0, 2))
        inter = bool(rng.integers(0, 2))
        a_buf, o_buf, b_buf = (int(b) for b in rng.permutation(bufs))
        I = Instr(op=isa.OP_CONV, act=isa.ACT_RELU if relu else isa.ACT_NONE, sh=sh, sw=sw,
                  a_buf=a_buf, a_base=P * int(rng.integers(0, 3)), a_h=ah, a_w=aw, a_c=cin,
                  a_pix=a_pix, a_off=a_off, a_signed=int(rng.integers(0, 2)),
                  o_buf=o_buf, o_base=P * int(rng.integers(0, 3)), o_h=oh, o_w=ow, o_c=cout,
                  o_signed=0 if relu else int(rng.integers(0, 2)), w_base=len(params))
        if inter:
            I.epi = isa.EPI_INTERLEAVE
            if rng.integers(0, 2):      # in place: B is the first half of the output pixel
                I.b_buf, I.b_base, I.b_pix, I.b_off = o_buf, I.o_base, 2 * cout, 0
            else:
                I.b_buf, I.b_base = b_buf, P * int(rng.integers(0, 3))
                I.b_pix = cout + P * int(rng.integers(0, 2))
                I.o_pix = 2 * cout + P * int(rng.integers(0, 2))
        else:
            I.o_off = P * int(rng.integers(0, 2))
            I.o_pix = I.o_off + cout + P * int(rng.integers(0, 2))
        params.append(dict(w=rng.integers(-128, 128, (cout, cin, 1, 1)),
                           bias=rng.integers(-2 ** 15, 2 ** 15, cout),
                           M=rng.integers(1, 2 ** 31, cout), S=rng.integers(32, 45, cout)))
        prog.append(I)
        for buf, n in ((I.a_buf, I.a_base + ah * aw * a_pix),
                       (I.o_buf, I.o_base + oh * ow * (I.o_pix or I.out_channels())),
                       (I.b_buf, I.b_base + oh * ow * (I.b_pix or cout))):
            peak[buf] = max(peak.get(buf, 0), n)

    def state(m):   # fresh random bytes everywhere before each instruction
        for b in bufs:
            m.buf[b][:] = rng.integers(0, 256, len(m.buf[b]))

    def as_signed(m, I):  # the reference stores signed values as negative numbers
        if I.a_signed:
            v = m.buf[I.a_buf]
            v[v > 127] -= 256
    # wrap state so each instruction's A buffer has the right signedness
    it = iter(prog)

    def state2(m):
        state(m)
        as_signed(m, next(it))
    return run_suite("random", prog, params, peak, state=state2)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    bad = shufflenet() | random_suite(seed)
    print("TEST FAILED" if bad else "TEST PASSED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
