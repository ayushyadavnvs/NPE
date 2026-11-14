"""Hardware-aware post-training quantization: a float ONNX CNN to a QDQ ONNX
model the importer (onnx_import.py) takes as is.

  python3 -m npe.frontend.ptq float.onnx out.onnx --calib DIR [--n 64]
          [--mean 0.485,0.456,0.406 --std 0.229,0.224,0.225]

What it does, in the engine's terms:
  * BatchNormalization folded into the conv before it.
  * Input normalization folded into the first conv: the quantized model eats
    raw 0..255 pixels (per-channel zero point round(255 * mean), so zero
    padding stays exact as constant padding).
  * Weights int8 symmetric, per output channel (fc: one scale, so the raw
    int32 logits of all classes compare directly). Biases int32.
  * Activations zero point 0: uint8 where the value is never negative (ReLU,
    Clip, max pool or concat of those), int8 otherwise. Scale from the
    calibration range (a high percentile, so rare outliers clip).
  * Tensors that the engine joins without requantizing share one scale:
    concat inputs and output, max pool in and out, and everything a split /
    reshape / transpose only rearranges.
Calibration images: input.bin files (H x W x 3 uint8) under DIR, or .npy.
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LAYOUT = {"Split", "Slice", "Reshape", "Transpose", "Flatten", "Squeeze", "Unsqueeze", "Identity"}
SAME_SCALE = {"MaxPool", "Concat"}
ACT = {"Relu", "Clip"}


def load_images(d, n, hw):
    out = []
    for root, _, files in sorted(os.walk(d)):
        for f in sorted(files):
            if f == "input.bin":
                a = np.fromfile(os.path.join(root, f), np.uint8)
                out.append(a.reshape(hw[0], hw[1], -1))
            elif f.endswith(".npy"):
                out.append(np.load(os.path.join(root, f)))
            if len(out) >= n:
                return out
    return out


def fix_batch(model):
    for t in list(model.graph.input) + list(model.graph.output):
        d = t.type.tensor_type.shape.dim
        if d and not d[0].dim_value:
            d[0].dim_value = 1
    return model


def fold(model, mean=None, std=None):
    """BN into conv, normalization into the first conv, shape subgraphs into
    constants. Returns (model, name of the raw-pixel input)."""
    model = fix_batch(model)
    if model.opset_import[0].version < 13:
        from onnx import version_converter
        model = version_converter.convert_version(model, 13)
    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph
    init = {t.name: numpy_helper.to_array(t).astype(np.float32) if t.data_type == TensorProto.FLOAT
            else numpy_helper.to_array(t) for t in g.initializer}
    shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
              for vi in list(g.value_info) + list(g.output)}
    prod = {o: n for n in g.node for o in n.output}
    nodes = []
    rename = {}
    for n in g.node:
        n.input[:] = [rename.get(i, i) for i in n.input]
        if n.op_type == "BatchNormalization":
            c = prod.get(n.input[0])
            if c is None or c.op_type != "Conv":
                raise ValueError(f"{n.name}: BatchNormalization not after a Conv")
            gm, bt, mu, var = (init[i] for i in n.input[1:5])
            eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
            k = gm / np.sqrt(var + eps)
            w = init[c.input[1]] * k[:, None, None, None]
            b0 = init[c.input[2]] if len(c.input) > 2 else np.zeros_like(k)
            b = (b0 - mu) * k + bt
            init[c.input[1]] = w.astype(np.float32)
            bn = c.input[1] + ".bias"
            init[bn] = b.astype(np.float32)
            if len(c.input) > 2:
                c.input[2] = bn
            else:
                c.input.append(bn)
            rename[n.output[0]] = c.output[0]
            continue
        if n.op_type in ("Shape", "Gather", "Unsqueeze", "Constant") and all(
                o not in shapes or True for o in n.output):
            nodes.append(n)
            continue
        if n.op_type == "Reshape" and any(u.op_type == "Gemm" for u in g.node if n.output[0] in u.input):
            nodes.append(helper.make_node("Flatten", [n.input[0]], [n.output[0]], axis=1))
            continue
        if n.op_type == "Reshape" and n.output[0] in shapes and all(shapes[n.output[0]]):
            nm = n.output[0] + ".shape"
            init[nm] = np.array(shapes[n.output[0]], np.int64)
            n.input[1] = nm
        nodes.append(n)
    for o in g.output:
        o.name = rename.get(o.name, o.name)
    # drop nodes nothing uses (the shape subgraphs)
    need = {o.name for o in g.output}
    keep = []
    for n in reversed(nodes):
        if any(o in need for o in n.output):
            keep.append(n)
            need.update(n.input)
    keep.reverse()
    inp = [i for i in g.input if i.name not in init][0]
    if mean is not None:   # normalization into the first conv
        c = next(n for n in keep if n.op_type == "Conv" and inp.name in n.input)
        w = init[c.input[1]]
        init[c.input[1]] = (w / (255.0 * np.asarray(std, np.float32))[None, :, None, None]).astype(np.float32)
    inits = [numpy_helper.from_array(v, k) for k, v in init.items() if k in need]
    g2 = helper.make_graph(keep, g.name, [inp], list(g.output), inits)
    m = helper.make_model(g2, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return onnx.shape_inference.infer_shapes(m), inp.name


def calibrate(model, inp, images, zp, pct):
    import onnxruntime as ort
    g = model.graph
    names = [o for n in g.node for o in n.output]
    m = onnx.ModelProto()
    m.CopyFrom(model)
    del m.graph.output[:]
    for nm in names:
        m.graph.output.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    vals = {nm: [] for nm in names}
    mins = {nm: 0.0 for nm in names}
    for im in images:
        x = (im.astype(np.float32) - zp.astype(np.float32)).transpose(2, 0, 1)[None]
        outs = sess.run(names, {inp: x})
        for nm, v in zip(names, outs):
            a = np.abs(v).reshape(-1)
            vals[nm].append(np.percentile(a, pct) if a.size > 1000 else a.max())
            mins[nm] = min(mins[nm], float(v.min()))
    return {nm: float(np.mean(v)) for nm, v in vals.items()}, mins


def quantize(model, inp, ranges, mins, zp):
    g = model.graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)
    # scale groups (union-find)
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for n in g.node:
        if n.op_type in LAYOUT or n.op_type in SAME_SCALE:
            for o in n.output:
                union(o, n.input[0])
            if n.op_type == "Concat":
                for i in n.input:
                    union(i, n.output[0])
    # quant points: outputs where a compute chain ends
    def is_qpoint(t):
        n = prod.get(t)
        if n is None or n.op_type in LAYOUT or n.op_type in ("Gemm", "MatMul", "Constant"):
            return False
        us = users.get(t, [])
        if n.op_type in ("Conv", "Add") and len(us) == 1 and us[0].op_type in ACT:
            return False
        if n.op_type == "Conv" and len(us) == 1 and us[0].op_type == "Add":
            return False      # the engine adds the accumulator (ADD epilogue)
        return True

    qp = [o for n in g.node for o in n.output if is_qpoint(o)]
    grp_max, grp_neg = {}, {}
    for t in qp:
        r = find(t)
        grp_max[r] = max(grp_max.get(r, 0.0), ranges[t])
        n = prod[t]
        nonneg = n.op_type in ACT or (n.op_type in ("MaxPool", "Concat", "GlobalAveragePool", "ReduceMean")
                                      and mins[t] >= 0)
        grp_neg[r] = grp_neg.get(r, False) or not nonneg
    scale = {}
    for t in qp:
        r = find(t)
        scale[t] = (max(grp_max[r], 1e-8) / (127.0 if grp_neg[r] else 255.0), grp_neg[r])
    # tensors seen through layout ops get their group's scale
    def tscale(t):
        t = t.removesuffix(".dq")
        if t in scale:
            return scale[t]
        if t == inp:
            return 1.0, False
        n = prod.get(t)
        if n is not None and (n.op_type in LAYOUT or n.op_type in SAME_SCALE):
            return tscale(n.input[0])
        if n is not None and n.op_type in ACT:
            return tscale(n.output[0])
        raise KeyError(t)

    new_nodes, inits = [], []
    f32 = lambda a: np.asarray(a, np.float32)   # noqa: E731

    def c(name, arr):
        inits.append(numpy_helper.from_array(arr, name))
        return name

    def qdq(t, s, signed, axis=None, z=None):
        z = np.zeros((), np.int8 if signed else np.uint8) if z is None else z
        kw = {} if axis is None else {"axis": axis}
        c(t + ".qs", f32(s))
        c(t + ".qz", z)
        new_nodes.append(helper.make_node("QuantizeLinear", [t, t + ".qs", t + ".qz"], [t + ".q"], **kw))
        new_nodes.append(helper.make_node("DequantizeLinear", [t + ".q", t + ".qs", t + ".qz"],
                                          [t + ".dq"], **kw))
        return t + ".dq"

    remap = {}
    remap[inp] = qdq(inp, np.ones(len(zp)), False, axis=1, z=np.asarray(zp, np.uint8))
    for n in g.node:
        n.input[:] = [remap.get(i, i) for i in n.input]
        if n.op_type in ("Conv", "Gemm"):
            w = init[n.input[1]].astype(np.float64)
            s_in = tscale(n.input[0].removesuffix(".dq"))[0] if n.input[0].endswith(".dq") else \
                tscale(n.input[0])[0]
            if n.op_type == "Gemm":
                tb = next((a.i for a in n.attribute if a.name == "transB"), 0)
                if not tb:
                    w = w.T
                    for a in n.attribute:
                        if a.name == "transB":
                            a.i = 1
                    if not any(a.name == "transB" for a in n.attribute):
                        n.attribute.append(helper.make_attribute("transB", 1))
                sw = np.full(w.shape[0], max(np.abs(w).max(), 1e-12) / 127.0)
            else:
                sw = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(axis=1), 1e-12) / 127.0
                sw = np.maximum(sw, 1e-4 * sw.max())
            sh = (-1,) + (1,) * (w.ndim - 1)
            wq = np.clip(np.round(w / sw.reshape(sh)), -127, 127).astype(np.int8)
            nm = n.name or n.output[0]
            per = n.op_type == "Conv"
            c(nm + ".w", wq)
            c(nm + ".ws", f32(sw if per else sw[0]))
            c(nm + ".wz", np.zeros(w.shape[0] if per else (), np.int8))
            new_nodes.append(helper.make_node("DequantizeLinear", [nm + ".w", nm + ".ws", nm + ".wz"],
                                              [nm + ".wf"], **({"axis": 0} if per else {})))
            n.input[1] = nm + ".wf"
            if len(n.input) > 2 and n.input[2]:
                b = init[n.input[2]].astype(np.float64)
                bs = f32(f32(sw) * np.float32(s_in))
                bq = np.round(b / bs).astype(np.int32)
                c(nm + ".b", bq)
                c(nm + ".bs", bs if per else bs[0])
                c(nm + ".bz", np.zeros(len(bq) if per else (), np.int32))
                new_nodes.append(helper.make_node("DequantizeLinear", [nm + ".b", nm + ".bs", nm + ".bz"],
                                                  [nm + ".bf"], **({"axis": 0} if per else {})))
                n.input[2] = nm + ".bf"
        new_nodes.append(n)
        for o in n.output:
            if o in scale:
                remap[o] = qdq(o, *scale[o])
    # graph outputs keep their names
    for o in g.output:
        if o.name in remap:
            new_nodes.append(helper.make_node("Identity", [remap[o.name]], [o.name + ".out"]))
            o.name = o.name + ".out"
    keep_init = [t for t in g.initializer if t.name not in
                 {i for n in g.node if n.op_type in ("Conv", "Gemm") for i in n.input[1:]}]
    used = {i for n in new_nodes for i in n.input}
    inits += [t for t in keep_init if t.name in used]
    raw = helper.make_tensor_value_info("image", TensorProto.FLOAT, None)
    for nd in new_nodes:
        nd.input[:] = ["image" if i == inp else i for i in nd.input]
    shp = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    raw = helper.make_tensor_value_info("image", TensorProto.FLOAT, shp)
    g2 = helper.make_graph(new_nodes, g.name + "_qdq", [raw], list(g.output), inits)
    m = helper.make_model(g2, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def run(src, out, calib, n=64, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), pct=99.99):
    model, inp = fold(onnx.load(src), mean, std)
    shp = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    zp = np.round(255.0 * np.asarray(mean)).astype(np.int64)
    images = load_images(calib, n, shp[2:])
    ranges, mins = calibrate(model, inp, images, zp, pct)
    m = quantize(model, inp, ranges, mins, zp)
    onnx.save(m, out)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--pct", type=float, default=99.99)
    a = ap.parse_args()
    run(a.src, a.out, a.calib, a.n, pct=a.pct)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
