"""ONNX importer checks.

  python3 -m npe.tests.test_onnx_import [N] [--models a,b] [--hw]

  golden   ShuffleNetV2 x0.5 exported to QDQ ONNX and imported back: all 104
           golden images bit-exact (also reported without the exact
           requant metadata, from float32 scales alone)
  models   each target model imported, run on the reference interpreter for
           N inputs, vs ONNX Runtime running the same QDQ model (and vs the
           original TFLite model for ResNet-8 / DS-CNN). Not bit-exact by
           design: ONNX Runtime and TFLite requantize in float / gemmlowp
           and requantize tensors the engine keeps at full precision.
  --hw     each model through the HLS top level (C++ build of npe_top, CPU
           only) vs the reference interpreter: bit-exact.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

from npe.compiler.build import build
from npe.frontend import models
from npe.frontend.onnx_import import import_model
from npe.ref.isa_ref import Machine
from npe.tests.test_engine import APDIR, BUILD, NPE


def golden():
    from npe.frontend.export_golden import export
    bad = 0
    for exact in (True, False):
        p = import_model(export(exact))
        ok = 0
        for d in models.golden_images():
            x = np.fromfile(os.path.join(d, "input.bin"), np.uint8).reshape(224, 224, 3)
            gold = np.fromfile(os.path.join(d, "logits.bin"), "<i4")
            ok += np.array_equal(Machine(p.params, p.peak).run(p.prog, x).reshape(-1), gold)
        n = len(models.golden_images())
        print(f"golden ShuffleNetV2 x0.5 via ONNX{'' if exact else ' (float32 scales only)'}: "
              f"{ok} of {n} images bit-exact on {len(p.prog)} instructions")
        bad += exact and ok != n
    return bad


def ort_input(p, x):
    s = np.broadcast_to(np.asarray(p.in_scale, np.float64).reshape(-1), (p.in_shape[2],))
    z = np.broadcast_to(np.asarray(p.in_zp, np.float64).reshape(-1), (p.in_shape[2],))
    real = (x - p.in_offset - z) * s
    return real.astype(np.float32).transpose(2, 0, 1)[None]


def compare(name, n):
    import onnx
    import onnxruntime as ort
    path = models.prepare(name)
    p = import_model(path)
    so = ort.SessionOptions()   # no fusion: ONNX Runtime's fused int8 kernels may saturate
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    tfl = None
    if models.MODELS[name]["kind"] == "tflite":
        from ai_edge_litert.interpreter import Interpreter
        tfl = Interpreter(models.fetch(name), experimental_preserve_all_tensors=True)
        tfl.allocate_tensors()
        ind = tfl.get_input_details()[0]
        fc = [o for o in tfl._get_ops_details() if o["op_name"] == "FULLY_CONNECTED"][-1]["outputs"][0]
        fs, fz = tfl._get_tensor_details(fc, 0)["quantization"]
    xs = models.inputs(name, n, p)
    agree = agree_t = 0
    err = []
    for x in xs:
        e = Machine(p.params, p.peak).run(p.prog, x).reshape(-1) * p.out_scale
        o = sess.run(None, {sess.get_inputs()[0].name: ort_input(p, x)})[0].reshape(-1)
        agree += e.argmax() == o.argmax()
        err.append(np.abs(e - o).max() / max(np.abs(o).max(), 1e-9))
        if tfl is not None:
            q = (x - p.in_offset).reshape(1, *x.shape).astype(np.int8)
            tfl.set_tensor(ind["index"], q)
            tfl.invoke()
            t = (tfl.get_tensor(fc).astype(np.float64).reshape(-1) - fz) * fs
            agree_t += e.argmax() == t.argmax()
    extra = f", top-1 same as TFLite {agree_t}/{len(xs)}" if tfl is not None else ""
    print(f"{name:16s} {len(p.prog):3d} instructions: top-1 same as ONNX Runtime {agree}/{len(xs)}"
          f"{extra}; logit error vs ONNX Runtime median {np.median(err):.1%} max {np.max(err):.1%}")
    for note in p.notes:
        print("   ", note)
    return 0


def layerwise(name, n):
    """Each instruction fed ONNX Runtime's own quantized inputs: its output vs
    ONNX Runtime's quantized output (float semantics, no fusion). Differences
    are rounding only: float32 conv sums and round-half-even there, 64-bit
    fixed point here; at most 1 LSB except where the engine's ADD skips the
    conv's own rounding."""
    import onnx
    import onnxruntime as ort
    from onnx import shape_inference
    from npe.frontend.onnx_import import Importer, Lowering
    from npe.compiler import isa
    model = onnx.load(models.prepare(name))
    imp = Importer(model).run()
    low = Lowering(imp).run()
    p = import_model(model)
    vi = {v.name: v for v in shape_inference.infer_shapes(model).graph.value_info}
    roots = {}
    for L in imp.layers:
        for t in [L.out] + list(L.parts or []):
            if t is not None and t.bits == 8 and t.onnx_q is not None:
                roots[t.onnx_q[0]] = t

    def eng(t, q):
        """ONNX quantized values -> the engine's numbers for tensor t."""
        _, s, z = t.onnx_q
        v = q.astype(np.int64) - z
        if t.onnx_idx is not None:
            v = v[..., t.onnx_idx]
        if abs(s - t.scale) > 1e-6 * s:     # widened zero point: rescaled
            v = np.round(v * s / t.scale).astype(np.int64)
        return v

    m2 = onnx.ModelProto()
    m2.CopyFrom(model)
    for nm in roots:
        m2.graph.output.append(vi[nm])
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(m2.SerializeToString(), so, providers=["CPUExecutionProvider"])
    worst, diff, total = 0, 0, 0
    for x in models.inputs(name, n, p):
        outs = sess.run(None, {sess.get_inputs()[0].name: ort_input(p, x)})
        val = {o.name: v for o, v in zip(sess.get_outputs(), outs)}
        m = Machine(low.params, low.peak)
        m.write(isa.BUF_IN, 0, x, 0, 0)

        def put(t):
            r = t.base_root()
            if r.onnx_q is None or r.onnx_q[0] not in val:
                return
            v = eng(r, val[r.onnx_q[0]][0].transpose(1, 2, 0))
            e = np.zeros(v.shape[:2] + (r.width,), np.int64)
            e[..., r.perm] = v
            m.write(r.grp.buf, r.grp.base, e, r.pix, r.off)

        for L, I in zip([L for L in imp.layers if L.t >= 0], low.prog):
            if L.kind in ("gap", "fc"):
                continue
            for t in (L.x, L.add, L.ib):
                if t is not None:
                    put(t)
            r = m.step(I)
            if L.epi == isa.EPI_INTERLEAVE:
                r = r[..., 1::2]
            got = r[..., L.out.perm]
            ref = eng(L.out, val[L.out.onnx_q[0]][0].transpose(1, 2, 0))
            d = np.abs(got - ref)
            worst = max(worst, int(d.max()))
            diff += int((d > 1).sum())
            total += d.size
    print(f"{name:16s} per instruction on ONNX Runtime's inputs: max difference {worst} LSB, "
          f"{diff / total:.4%} of values off by more than 1")
    return 0


def hw(name, n):
    p = import_model(models.prepare(name))
    d = os.path.join(BUILD, f"onnx_{name}")
    os.makedirs(d, exist_ok=True)
    dirs = []
    for k, x in enumerate(models.inputs(name, n, p)):
        out = os.path.join(d, "images", f"img{k:02d}")
        os.makedirs(out, exist_ok=True)
        (x & 0xFF).astype(np.uint8).tofile(os.path.join(out, "input.bin"))
        Machine(p.params, p.peak).run(p.prog, x).reshape(-1).astype("<i4").tofile(
            os.path.join(out, "logits.bin"))
        dirs.append(out)
    from npe.compiler import memplan
    fits = None
    for board in ("kv260", "zcu104"):
        try:
            build(p.prog, p.params, p.peak, d, board=board)
            fits = board
            break
        except ValueError:
            pass
    if fits is None:   # C simulation only: memories sized without a board limit
        memplan.BOARDS.setdefault("csim", dict(part="", uram=1 << 20, bram=1 << 20, lut=1 << 30))
        build(p.prog, p.params, p.peak, d, board="csim")
    lst = os.path.join(d, "images.txt")
    with open(lst, "w") as f:
        f.write("\n".join(dirs) + "\n")
    if not os.path.exists(os.path.join(APDIR, "include", "ap_int.h")):
        subprocess.check_call(["git", "clone", "-q", "--depth", "1",
                               "https://github.com/Xilinx/HLS_arbitrary_Precision_Types", APDIR])
    exe = os.path.join(d, "tb_top")
    subprocess.check_call(["g++", "-std=c++14", "-O2", "-w", f"-I{d}", f"-I{NPE}/hls",
                           f"-I{NPE}/isa", f"-I{NPE}/tb/shim", f"-I{APDIR}/include",
                           "-o", exe, f"{NPE}/tb/tb_top.cpp", f"{NPE}/hls/npe_top.cpp"])
    r = subprocess.run([exe, os.path.join(d, "load.bin"), lst], capture_output=True, text=True)
    last = (r.stdout.strip() or r.stderr.strip()).splitlines()[-1:]
    fit = f"fits {fits} on chip" if fits else "does not fit on chip yet (weights need DRAM streaming)"
    print(f"{name:16s} HLS top level vs reference: {'PASS' if r.returncode == 0 else 'FAIL'} {last}; {fit}")
    return r.returncode != 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=20)
    ap.add_argument("--models", default=",".join(models.MODELS))
    ap.add_argument("--hw", action="store_true")
    a = ap.parse_args()
    names = a.models.split(",")
    bad = 0
    if a.hw:
        for nm in names:
            bad += hw(nm, a.n)
    else:
        bad += golden()
        for nm in names:
            bad += compare(nm, a.n)
        for nm in names:
            bad += layerwise(nm, min(a.n, 4))
    print("TEST PASSED" if not bad else "TEST FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
