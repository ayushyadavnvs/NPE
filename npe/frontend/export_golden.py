"""Exports the golden quantized ShuffleNetV2 x0.5 (old weights.bin) as a QDQ
ONNX model in the torchvision graph layout (Split, Concat, Reshape /
Transpose / Reshape channel shuffle), so the ONNX importer can be checked
against the 104 golden logits.

Integers come from weights.bin unchanged; float scales from quant_meta.json
(the weight scale of each output channel is derived from M / 2^S so the
scales are consistent). The exact M, S go into metadata "npe.requant":
float32 scales cannot hold a 31-bit multiplier.

  python3 -m npe.frontend.export_golden [--out shufflenet_golden.onnx] [--no-exact]
"""
import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from npe.compiler.lower_shufflenet import OLD

sys.path.insert(0, os.path.join(OLD, "golden"))
META = os.path.join(OLD, "data", "clean_v3_iou_qat", "quant_meta.json")


def export(exact=True):
    from model_def import STAGE_OUT, units  # noqa: E402
    from quantize import read_bin  # noqa: E402
    layers, pad, logit_scale = read_bin(os.path.join(OLD, "data", "weights.bin"))
    L = {l["name"]: l for l in layers}
    meta = json.load(open(META))
    sc = meta["activation_scales"]
    nodes, inits, requant = [], [], {}
    f32 = lambda a: np.asarray(a, np.float32)   # noqa: E731

    def const(name, arr):
        inits.append(numpy_helper.from_array(arr, name))
        return name

    def q(x, s, signed, name):
        """Q then DQ of activation x with scale s (zero point 0)."""
        zp = np.zeros((), np.int8 if signed else np.uint8)
        const(name + ".s", f32(s))
        const(name + ".z", zp)
        nodes.append(helper.make_node("QuantizeLinear", [x, name + ".s", name + ".z"], [name + ".q"]))
        nodes.append(helper.make_node("DequantizeLinear", [name + ".q", name + ".s", name + ".z"], [name]))
        return name

    def conv(name, x, s_in, s_out, *, k=1, s=1, groups=1, bias_add=None, out_name=None):
        l = L[name]
        cout = l["cout"]
        m = l["M"] / 2.0 ** l["S"]
        s_w = m * s_out / s_in
        w = l["w"].reshape(cout, -1, k, k).astype(np.int8)
        b = l["bias"].astype(np.int64) + (0 if bias_add is None else bias_add(w))
        const(name + ".w", w)
        const(name + ".ws", f32(s_w))
        const(name + ".wz", np.zeros(cout, np.int8))
        const(name + ".b", b.astype(np.int32))
        const(name + ".bs", f32(f32(s_w) * np.float32(s_in)))
        const(name + ".bz", np.zeros(cout, np.int32))
        nodes.append(helper.make_node("DequantizeLinear", [name + ".w", name + ".ws", name + ".wz"],
                                      [name + ".wf"], axis=0))
        nodes.append(helper.make_node("DequantizeLinear", [name + ".b", name + ".bs", name + ".bz"],
                                      [name + ".bf"], axis=0))
        p = k // 2
        y = name + ".acc"
        nodes.append(helper.make_node("Conv", [x, name + ".wf", name + ".bf"], [y], name=name,
                                      kernel_shape=[k, k], strides=[s, s], pads=[p] * 4, group=groups))
        if l["relu"]:
            nodes.append(helper.make_node("Relu", [y], [name + ".relu"]))
            y = name + ".relu"
        requant[name] = dict(M=l["M"].tolist(), S=l["S"].tolist())
        return q(y, s_out, not l["relu"], out_name or name + ".out")

    # input: raw pixels; real value = pixel - pad (zero padding = mean padding)
    const("in.s", f32(np.ones(3)))
    const("in.z", pad.astype(np.uint8))
    nodes.append(helper.make_node("QuantizeLinear", ["input", "in.s", "in.z"], ["in.q"], axis=1))
    nodes.append(helper.make_node("DequantizeLinear", ["in.q", "in.s", "in.z"], ["in.f"], axis=1))
    padsum = lambda w: (w.astype(np.int64) * pad.astype(np.int64)[None, :, None, None]).sum((1, 2, 3))  # noqa
    x = conv("conv1", "in.f", 1.0, sc["conv1"], k=3, s=2, bias_add=padsum)
    nodes.append(helper.make_node("MaxPool", [x], ["maxpool.f"], name="maxpool", kernel_shape=[3, 3],
                                  strides=[2, 2], pads=[1, 1, 1, 1]))
    x = q("maxpool.f", sc["maxpool"], False, "maxpool")
    s_x = sc["maxpool"]
    h = 56
    for prefix, down in units():
        cout = STAGE_OUT[int(prefix[1]) - 1]
        half = cout // 2
        s_out = sc[f"{prefix}.out"]
        if down:
            b1 = conv(f"{prefix}.b1.dw", x, s_x, sc[f"{prefix}.b1.dw"], k=3, s=2, groups=L[f"{prefix}.b1.dw"]["cout"])
            b1 = conv(f"{prefix}.b1.pw", b1, sc[f"{prefix}.b1.dw"], s_out)
            src = x
            h //= 2
        else:
            sp = [f"{prefix}.x1", f"{prefix}.x2"]
            const(f"{prefix}.split", np.array([half, half], np.int64))
            nodes.append(helper.make_node("Split", [x, f"{prefix}.split"], sp, axis=1))
            b1, src = sp
        y = conv(f"{prefix}.pw1", src, s_x, sc[f"{prefix}.pw1"])
        y = conv(f"{prefix}.dw", y, sc[f"{prefix}.pw1"], sc[f"{prefix}.dw"], k=3, s=2 if down else 1, groups=half)
        y = conv(f"{prefix}.pw2", y, sc[f"{prefix}.dw"], s_out)
        cat = f"{prefix}.cat"
        nodes.append(helper.make_node("Concat", [b1, y], [cat], axis=1))
        const(f"{prefix}.s5", np.array([1, 2, half, h, h], np.int64))
        const(f"{prefix}.s4", np.array([1, cout, h, h], np.int64))
        nodes.append(helper.make_node("Reshape", [cat, f"{prefix}.s5"], [cat + "5"]))
        nodes.append(helper.make_node("Transpose", [cat + "5"], [cat + "t"], perm=[0, 2, 1, 3, 4]))
        nodes.append(helper.make_node("Reshape", [cat + "t", f"{prefix}.s4"], [cat + "s"]))
        x = q(cat + "s", s_out, False, f"{prefix}.out")
        s_x = s_out
    x = conv("conv5", x, s_x, sc["conv5"])
    nodes.append(helper.make_node("GlobalAveragePool", [x], ["gap.f"], name="gap"))
    s_gap = sc["conv5"] / 49   # the GAP stays an int32 sum: no Q here
    nodes.append(helper.make_node("Flatten", ["gap.f"], ["flat"]))
    fc = L["fc"]
    s_w = logit_scale / s_gap
    const("fc.w", fc["w"].reshape(fc["cout"], -1).astype(np.int8))
    const("fc.ws", f32(s_w))
    const("fc.wz", np.zeros((), np.int8))
    const("fc.b", fc["bias"].astype(np.int32))
    const("fc.bs", f32(np.float32(s_w) * np.float32(s_gap)))
    const("fc.bz", np.zeros((), np.int32))
    nodes.append(helper.make_node("DequantizeLinear", ["fc.w", "fc.ws", "fc.wz"], ["fc.wf"]))
    nodes.append(helper.make_node("DequantizeLinear", ["fc.b", "fc.bs", "fc.bz"], ["fc.bf"]))
    nodes.append(helper.make_node("Gemm", ["flat", "fc.wf", "fc.bf"], ["logits"], name="fc", transB=1))
    g = helper.make_graph(nodes, "shufflenetv2_x0.5_golden",
                          [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, fc["cout"]])],
                          inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    if exact:
        helper.set_model_props(m, {"npe.requant": json.dumps(requant)})
    onnx.checker.check_model(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="shufflenet_golden.onnx")
    ap.add_argument("--no-exact", action="store_true", help="leave out the exact M, S metadata")
    a = ap.parse_args()
    onnx.save(export(not a.no_exact), a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
