"""FOMO-style detector: a MobileNetV2 trunk cut at stride 8 plus a 1x1 conv
head that outputs a per-cell class map (background + objects), the layout of
Edge Impulse's FOMO ("faster objects, more objects").

  python3 -m npe.frontend.fomo out.onnx [--img 96] [--classes 3]

The trunk is the ImageNet MobileNetV2 1.0 from the ONNX model zoo, cut after
the expand ReLU6 of block 6 (192 channels, H/8 x W/8). The head (1x1 conv
to 32 + ReLU, 1x1 conv to `classes`) is randomly initialised with a fixed
seed: the model tests the engine (feature-map output, bit-exactness), it is
not a trained detector. Softmax over classes is left to the host. The result
is a float ONNX model for ptq.py.
"""
import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper, utils

CUT = "368"          # output of Clip_34: block 6 expand ReLU6, stride 8, 192 channels


def build(src, img=96, classes=3, seed=0):
    m = onnx.load(src)
    for d, v in zip(m.graph.input[0].type.tensor_type.shape.dim, (1, 3, img, img)):
        d.ClearField("dim_param")
        d.dim_value = v
    m = onnx.shape_inference.infer_shapes(m)
    m = utils.Extractor(m).extract_model([m.graph.input[0].name], [CUT])
    rng = np.random.default_rng(seed)
    c_in, hid = 192, 32
    w1 = rng.normal(0, np.sqrt(2 / c_in), (hid, c_in, 1, 1)).astype(np.float32)
    b1 = np.full(hid, 0.1, np.float32)
    w2 = rng.normal(0, np.sqrt(1 / hid), (classes, hid, 1, 1)).astype(np.float32)
    b2 = np.zeros(classes, np.float32)
    g = m.graph
    for nm, a in (("head1.weight", w1), ("head1.bias", b1), ("head2.weight", w2), ("head2.bias", b2)):
        g.initializer.append(numpy_helper.from_array(a, nm))
    g.node.extend([
        helper.make_node("Conv", [CUT, "head1.weight", "head1.bias"], ["head1"], name="head1",
                         kernel_shape=[1, 1]),
        helper.make_node("Relu", ["head1"], ["head1.relu"], name="head1.relu"),
        helper.make_node("Conv", ["head1.relu", "head2.weight", "head2.bias"], ["heatmap"],
                         name="head2", kernel_shape=[1, 1]),
    ])
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("heatmap", onnx.TensorProto.FLOAT,
                                                  [1, classes, img // 8, img // 8]))
    onnx.checker.check_model(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--src", default="npe/build/models/src/mobilenetv2-12.onnx")
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--classes", type=int, default=3)
    a = ap.parse_args()
    onnx.save(build(a.src, a.img, a.classes), a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
