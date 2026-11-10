# INT8 golden model

Bit-exact reference for the accelerator. The hardware must reproduce every
tensor these models produce, byte for byte.

| File | What it is |
|---|---|
| `c_model/golden_model.{h,cpp}` | The C++ golden model (plain C++11, no HLS types). Link it into the HLS testbench later |
| `c_model/golden_main.cpp` | Command-line driver: `golden_model weights.bin image.bin [dump_dir]` |
| `ref_int8.py` | Independent NumPy INT8 model, used to cross-check the C++ one |
| `model_def.py` | Float ShuffleNetV2 x0.5 in NumPy, checked equal to torchvision |
| `quantize.py` | BN folding, INT8 quantization, `weights.bin` writer/reader (format in its docstring) |
| `run_checks.py` | Runs everything and writes golden vectors to `../data/` |
| `img2bin.py` | Image to 224x224x3 uint8 HWC `.bin` |
| `export_npz.py` | For the model team: trained `.pth` to `.npz` (needs PyTorch) |
| `check_torchvision.py` | Proves `model_def.py` matches torchvision (needs PyTorch) |
| `train_head.py` | Stand-in checkpoint: ImageNet ShuffleNetV2 x0.5 + 2-class head trained on a local dataset (needs PyTorch) |
| `verify_pytorch.py` | Folded float and INT8 models vs torchvision on a labelled val set (needs PyTorch) |
| `images/` | Six test images copied from Yolo_sim_hls |

## Run

```
cd new_accelrator/golden
python3 run_checks.py                          # synthetic weights for now
python3 run_checks.py --npz shufflenet_2class.npz   # once the checkpoint arrives

cd c_model
g++ -std=c++11 -O2 -Wall -Wextra -o golden_model golden_main.cpp golden_model.cpp
./golden_model ../../data/weights.bin ../../data/golden/dr/input.bin /tmp/dump
```

Needs Python 3 with NumPy and Pillow, and g++.

## Outputs in `../data/`

* `weights.bin`: all INT8 weights, INT32 biases, requant multipliers and shifts.
* `quant_meta.json`: float scales per tensor, for debugging.
* `golden/<image>/<tensor>.bin`: every intermediate tensor, HWC, raw bytes.
  `*.dw` are int8, `gap` and `logits` are little-endian int32, everything else
  uint8. Names: `input`, `conv1`, `maxpool`, `s2u0.b1.dw`, `s2u0.b1.pw`,
  `s2u0.pw1`, `s2u0.dw`, `s2u0.pw2`, `s2u0.out`, ..., `s4u3.out`, `conv5`,
  `gap`, `logits`.
* `golden/report.txt`: result of the last check.

## Current weights (2026-10-19)

Lightning AI (team checkpoint) is down, so `../data/shufflenet_2class.{pth,npz}`
is a stand-in: ImageNet ShuffleNetV2 x0.5 from torchvision with a 2-class
head trained by `train_head.py` on 200 local train images (backbone frozen).
It is for checking numerical integrity, not accuracy (73% on 100 val images).

```
python3 train_head.py shufflenetv2_x0.5-f707e7126e.pth DATASET ../data/shufflenet_2class
python3 run_checks.py --npz ../data/shufflenet_2class.npz --calib DATASET/train
python3 verify_pytorch.py ../data/shufflenet_2class.pth ../data/weights.bin DATASET/val
```

Results (`../data/pytorch_check.txt`): the BN-folded float model equals
torchvision to 4e-15; INT8 agrees with PyTorch on 100/100 val images (same
73% accuracy), mean logit error 0.05 against a mean margin of 2.5. Per-layer
SQNR bottoms out around 11.5 dB at conv5 (plain max calibration, no QAT).

`run_checks.py` without `--npz` still runs on synthetic weights; random
networks amplify any small noise, so its float-vs-INT8 numbers mean nothing.
