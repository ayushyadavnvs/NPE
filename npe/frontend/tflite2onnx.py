"""TFLite int8 model to QDQ ONNX, keeping every integer and scale.

  python3 -m npe.frontend.tflite2onnx model.tflite out.onnx

For the MLPerf Tiny reference models (ResNet-8, DS-CNN) and other fully int8
TFLite CNNs: CONV_2D, DEPTHWISE_CONV_2D (multiplier 1), ADD, AVERAGE_POOL_2D,
MAX_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX (dropped: the engine returns
logits). NHWC becomes NCHW; fused activations become Relu / Clip. Tensors
keep TFLite's asymmetric quantization; the importer maps it to the engine
(see onnx_import.py).
"""
import argparse

import numpy as np
import onnx
import tflite
from onnx import TensorProto, helper, numpy_helper

OPS = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith("_")}
F32 = np.float32


def convert(path):
    buf = open(path, "rb").read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    g = m.Subgraphs(0)
    nodes, inits = [], []
    names = {}

    def tensor(i):
        return g.Tensors(i)

    def data(i):
        t = tensor(i)
        b = m.Buffers(t.Buffer()).DataAsNumpy()
        dt = {tflite.TensorType.INT8: np.int8, tflite.TensorType.INT32: np.int32,
              tflite.TensorType.UINT8: np.uint8, tflite.TensorType.FLOAT32: np.float32,
              tflite.TensorType.INT64: np.int64}[t.Type()]
        return np.frombuffer(b.tobytes(), dt).reshape(t.ShapeAsNumpy())

    def quant(i):
        q = tensor(i).Quantization()
        return q.ScaleAsNumpy().astype(np.float64), q.ZeroPointAsNumpy().astype(np.int64)

    def c(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def act_q(i):
        """Q + DQ of activation tensor i (output of a float op named f)."""
        s, z = quant(i)
        dt = np.int8 if tensor(i).Type() == tflite.TensorType.INT8 else np.uint8
        nm = f"t{i}"
        c(nm + ".s", F32(s[0]))
        c(nm + ".z", np.array(z[0], dt))
        nodes.append(helper.make_node("QuantizeLinear", [nm + ".f", nm + ".s", nm + ".z"], [nm + ".q"]))
        nodes.append(helper.make_node("DequantizeLinear", [nm + ".q", nm + ".s", nm + ".z"], [nm]))
        names[i] = nm
        return nm

    def weights(i, w_nchw, per_axis, s_in, nm):
        s, z = quant(i)
        assert np.all(z == 0), "weights need zero point 0"
        c(nm + ".w", w_nchw.astype(np.int8))
        c(nm + ".ws", F32(s if per_axis else s[0]))
        c(nm + ".wz", np.zeros(len(s) if per_axis else (), np.int8))
        nodes.append(helper.make_node("DequantizeLinear", [nm + ".w", nm + ".ws", nm + ".wz"],
                                      [nm + ".wf"], **({"axis": 0} if per_axis else {})))
        return nm + ".wf", s

    def bias(i, s_w, s_in, nm, per_axis):
        b = data(i).astype(np.int32)
        bs = F32(F32(s_w) * F32(s_in))
        c(nm + ".b", b)
        c(nm + ".bs", bs if per_axis else bs[0])
        c(nm + ".bz", np.zeros(len(b) if per_axis else (), np.int32))
        nodes.append(helper.make_node("DequantizeLinear", [nm + ".b", nm + ".bs", nm + ".bz"],
                                      [nm + ".bf"], **({"axis": 0} if per_axis else {})))
        return nm + ".bf"

    def act(y, fused, out_i, nm):
        if fused == tflite.ActivationFunctionType.RELU:
            nodes.append(helper.make_node("Relu", [y], [nm + ".act"]))
            return nm + ".act"
        if fused == tflite.ActivationFunctionType.RELU6:
            c(nm + ".lo", F32(0))
            c(nm + ".hi", F32(6))
            nodes.append(helper.make_node("Clip", [y, nm + ".lo", nm + ".hi"], [nm + ".act"]))
            return nm + ".act"
        assert fused == tflite.ActivationFunctionType.NONE, f"activation {fused}"
        return y

    def pads(opt, hw_in, hw_out, k, st):
        if opt.Padding() == tflite.Padding.VALID:
            return [0, 0, 0, 0]
        tot = [max((o - 1) * s + kk - i, 0) for o, s, kk, i in zip(hw_out, st, k, hw_in)]
        return [tot[0] // 2, tot[1] // 2, tot[0] - tot[0] // 2, tot[1] - tot[1] // 2]

    inp = g.Inputs(0)
    ishape = tensor(inp).ShapeAsNumpy()    # N H W C
    s, z = quant(inp)
    dt = np.int8 if tensor(inp).Type() == tflite.TensorType.INT8 else np.uint8
    c("in.s", F32(s[0]))
    c("in.z", np.array(z[0], dt))
    nodes.append(helper.make_node("QuantizeLinear", ["input", "in.s", "in.z"], ["in.q"]))
    nodes.append(helper.make_node("DequantizeLinear", ["in.q", "in.s", "in.z"], [f"t{inp}"]))
    names[inp] = f"t{inp}"
    out_name = None
    for k in range(g.OperatorsLength()):
        op = g.Operators(k)
        code = m.OperatorCodes(op.OpcodeIndex())
        opn = OPS[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
        ins, outs = list(op.InputsAsNumpy()), list(op.OutputsAsNumpy())
        nm = f"{opn.lower()}_{k}"
        tb = op.BuiltinOptions()
        if opn in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            opt = (tflite.Conv2DOptions() if opn == "CONV_2D" else tflite.DepthwiseConv2DOptions())
            opt.Init(tb.Bytes, tb.Pos)
            x = names[ins[0]]
            w = data(ins[1])
            s_in = quant(ins[0])[0][0]
            if opn == "CONV_2D":
                wn = w.transpose(0, 3, 1, 2)                 # OHWI -> OIHW
                groups = 1
            else:
                assert opt.DepthMultiplier() == 1, "depth multiplier"
                wn = w.transpose(3, 0, 1, 2)                 # 1HWC -> C1HW
                groups = wn.shape[0]
            wf, s_w = weights(ins[1], wn, True, s_in, nm)
            bf = bias(ins[2], s_w, s_in, nm, True)
            ih, iw = tensor(ins[0]).ShapeAsNumpy()[1:3]
            oh, ow = tensor(outs[0]).ShapeAsNumpy()[1:3]
            st = [opt.StrideH(), opt.StrideW()]
            p = pads(opt, (ih, iw), (oh, ow), wn.shape[2:], st)
            nodes.append(helper.make_node("Conv", [x, wf, bf], [nm + ".y"], name=nm,
                                          kernel_shape=list(wn.shape[2:]), strides=st, pads=p,
                                          group=groups))
            y = act(nm + ".y", opt.FusedActivationFunction(), outs[0], nm)
            nodes.append(helper.make_node("Identity", [y], [f"t{outs[0]}.f"]))
            act_q(outs[0])
        elif opn == "ADD":
            opt = tflite.AddOptions()
            opt.Init(tb.Bytes, tb.Pos)
            nodes.append(helper.make_node("Add", [names[ins[0]], names[ins[1]]], [nm + ".y"], name=nm))
            y = act(nm + ".y", opt.FusedActivationFunction(), outs[0], nm)
            nodes.append(helper.make_node("Identity", [y], [f"t{outs[0]}.f"]))
            act_q(outs[0])
        elif opn in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
            opt = tflite.Pool2DOptions()
            opt.Init(tb.Bytes, tb.Pos)
            k_ = [opt.FilterHeight(), opt.FilterWidth()]
            st = [opt.StrideH(), opt.StrideW()]
            ih, iw = tensor(ins[0]).ShapeAsNumpy()[1:3]
            oh, ow = tensor(outs[0]).ShapeAsNumpy()[1:3]
            p = pads(opt, (ih, iw), (oh, ow), k_, st)
            kind = "AveragePool" if opn == "AVERAGE_POOL_2D" else "MaxPool"
            nodes.append(helper.make_node(kind, [names[ins[0]]], [f"t{outs[0]}.f"], name=nm,
                                          kernel_shape=k_, strides=st, pads=p))
            act_q(outs[0])
        elif opn == "RESHAPE":
            nodes.append(helper.make_node("Flatten", [names[ins[0]]], [f"t{outs[0]}"], axis=1))
            names[outs[0]] = f"t{outs[0]}"
        elif opn == "FULLY_CONNECTED":
            x = names[ins[0]]
            s_in = quant(ins[0])[0][0]
            w = data(ins[1])                                 # [out, in]
            per = len(quant(ins[1])[0]) > 1
            wf, s_w = weights(ins[1], w, per, s_in, nm)
            bf = bias(ins[2], s_w, s_in, nm, per)
            nodes.append(helper.make_node("Gemm", [x, wf, bf], [f"t{outs[0]}"], name=nm, transB=1))
            names[outs[0]] = out_name = f"t{outs[0]}"
        elif opn == "SOFTMAX":
            names[outs[0]] = names[ins[0]]
        else:
            raise NotImplementedError(opn)
    n_out = tensor(g.Outputs(0)).ShapeAsNumpy()[-1]
    graph = helper.make_graph(
        nodes, "tflite", [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, int(ishape[3]), int(ishape[1]), int(ishape[2])])],
        [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, int(n_out)])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()
    onnx.save(convert(a.src), a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
