"""The target models, fetched and turned into QDQ ONNX the importer takes.

  python3 -m npe.frontend.models [name ...]      # all if none given

  shufflenet_x0.5  the golden ShuffleNetV2 x0.5 (old weights.bin, 3 classes)
  shufflenet_x1.0  ShuffleNetV2 x1.0, ImageNet (ONNX model zoo, float), npe PTQ
  mobilenetv2      MobileNetV2 1.0, ImageNet (ONNX model zoo, float), npe PTQ
  resnet8          ResNet-8, CIFAR-10 (MLPerf Tiny reference, TFLite int8)
  dscnn            DS-CNN, keyword spotting (MLPerf Tiny reference, TFLite int8)

Downloads go to npe/build/models/ (not in git). PTQ calibrates on the old
board_test images (64 of them; the golden images are kept for checks).
"""
import os
import sys
import urllib.request

import numpy as np
import onnx

from npe.compiler.lower_shufflenet import OLD, ROOT

DIR = os.path.join(ROOT, "npe", "build", "models")
ZOO = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification"
TINY = "https://raw.githubusercontent.com/mlcommons/tiny/master/benchmark/training"
MODELS = {
    "shufflenet_x0.5": dict(kind="golden"),
    "shufflenet_x1.0": dict(kind="float", url=f"{ZOO}/shufflenet/model/shufflenet-v2-12.onnx"),
    "mobilenetv2": dict(kind="float", url=f"{ZOO}/mobilenet/model/mobilenetv2-12.onnx"),
    "resnet8": dict(kind="tflite", url=f"{TINY}/image_classification/trained_models/pretrainedResnet_quant.tflite"),
    "dscnn": dict(kind="tflite", url=f"{TINY}/keyword_spotting/trained_models/kws_ref_model.tflite"),
}
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def fetch(name):
    url = MODELS[name]["url"]
    path = os.path.join(DIR, "src", os.path.basename(url))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading", url)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def prepare(name):
    """Path of the model's QDQ ONNX (built on first use)."""
    out = os.path.join(DIR, f"{name}.onnx")
    if os.path.exists(out):
        return out
    os.makedirs(DIR, exist_ok=True)
    kind = MODELS[name]["kind"]
    if kind == "golden":
        from npe.frontend.export_golden import export
        onnx.save(export(), out)
    elif kind == "float":
        from npe.frontend import ptq
        ptq.run(fetch(name), out, os.path.join(OLD, "data", "board_test"), 64, MEAN, STD)
    else:
        from npe.frontend.tflite2onnx import convert
        onnx.save(convert(fetch(name)), out)
    return out


def golden_images(n=0):
    root = os.path.join(OLD, "data", "golden")
    d = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
    return [os.path.join(root, x) for x in (d[:n] if n else d)]


def inputs(name, n, prog):
    """n engine inputs (H x W x C int arrays, what the host streams) for a model:
    golden images (squashed to 32 x 32 for ResNet-8), smooth random feature
    maps for DS-CNN."""
    h, w, c = prog.in_shape
    if c == 3:
        out = []
        for d in golden_images(n):
            x = np.fromfile(os.path.join(d, "input.bin"), np.uint8).reshape(224, 224, 3).astype(np.int64)
            if h != 224:
                f = 224 // h
                x = x.reshape(h, f, w, f, 3).mean((1, 3)).round().astype(np.int64)
            out.append(x)
        return out
    rng = np.random.default_rng(0)
    out = []
    lo, hi = (-128, 127) if prog.in_signed else (0, 255)
    for _ in range(n):
        f = np.cumsum(np.cumsum(rng.normal(0, 1, (h, w, c)), 0), 1)
        f = (f - f.mean()) / (f.std() + 1e-9)
        out.append(np.clip(np.round(f * 40 + rng.uniform(-30, 30)), lo, hi).astype(np.int64))
    return out


def main():
    for n in sys.argv[1:] or list(MODELS):
        print(n, "->", prepare(n))


if __name__ == "__main__":
    main()
