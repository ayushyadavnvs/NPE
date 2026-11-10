"""Post-training INT8 quantization and export of weights.bin.

The scheme is the hardware contract (PLAN.md section 4):
  * BatchNorm folded into conv weights and bias.
  * Weights INT8 symmetric per output channel (fc: one scale for the tensor,
    so the host can compare raw logits directly).
  * Activations: UINT8 after ReLU, INT8 after depthwise. Zero point 0.
  * conv1 eats raw 0..255 RGB: input normalisation is folded into conv1 and
    the stem pads with PAD_VALUE (= round(255 * mean)) instead of 0.
  * All unit outputs of one stage share one scale, so split/concat/shuffle
    are pure wiring.
  * acc (INT32) -> out = clamp((acc * M + 2^(S-1)) >> S), M in [2^30, 2^31).
  * Global average pool is a plain INT32 sum over the 7x7 map; the 1/49 is
    folded into the fc scale. Logits are raw INT32.

When the model team delivers QAT-trained scales, they replace calibrate().

weights.bin layout (little endian):
  header: char magic[4] = "BDQW"; u32 version = 1; u32 num_layers;
          u8 pad_value[3]; u8 reserved; f32 logit_scale
  per layer: char name[32]; u8 kind (0 std, 1 dw, 2 pw, 3 fc); u8 k;
          u8 stride; u8 relu; u16 cin; u16 cout;
          i8 w[cout][cin_per_group][k][k]; i32 bias[cout]; i32 M[cout];
          u8 S[cout]
"""
import json
import struct

import numpy as np

from model_def import BN_EPS, MEAN, STD, FloatModel, layer_specs, units

KIND = {"std": 0, "dw": 1, "pw": 2, "fc": 3}
PAD_VALUE = np.round(255.0 * MEAN).astype(np.uint8)
GAP_PIXELS = 49  # 7x7 map before global pooling at 224 input


def calibrate(params, images):
    """Float activation scales per tensor, from the max seen on images."""
    mx = {}
    m = FloatModel(params)
    for img in images:
        rec = {}
        m.forward(img, record=rec)
        for k, v in rec.items():
            mx[k] = max(mx.get(k, 0.0), float(np.abs(v).max()))
    sc = {"conv1": mx["conv1"] / 255.0}
    sc["maxpool"] = sc["conv1"]
    stage_scale = {}
    for prefix, _ in units():
        st = prefix[:2]
        stage_scale[st] = max(stage_scale.get(st, 0.0), mx[f"{prefix}.out"] / 255.0)
    for prefix, down in units():
        st = prefix[:2]
        sc[f"{prefix}.pw1"] = mx[f"{prefix}.pw1"] / 255.0
        sc[f"{prefix}.dw"] = mx[f"{prefix}.dw"] / 127.0
        sc[f"{prefix}.pw2"] = stage_scale[st]
        sc[f"{prefix}.out"] = stage_scale[st]
        if down:
            sc[f"{prefix}.b1.dw"] = mx[f"{prefix}.b1.dw"] / 127.0
            sc[f"{prefix}.b1.pw"] = stage_scale[st]
    sc["conv5"] = mx["conv5"] / 255.0
    return sc


def input_scale(name, sc):
    """Scale of the tensor feeding layer `name` (conv1 sees raw pixels)."""
    if name == "conv1":
        return 1.0
    if name == "conv5":
        return sc["s4u3.out"]
    if name == "fc":
        return sc["conv5"] / GAP_PIXELS
    prefix, part = name.split(".", 1)
    idx = [p for p, _ in units()].index(prefix)
    unit_in = sc["maxpool"] if idx == 0 else sc[f"{units()[idx - 1][0]}.out"]
    if part in ("pw1", "b1.dw"):
        return unit_in
    return sc[{"dw": f"{prefix}.pw1", "pw2": f"{prefix}.dw", "b1.pw": f"{prefix}.b1.dw"}[part]]


def multiplier(m):
    """Real multiplier m > 0 -> (M, S) with m ~= M * 2^-S, M in [2^30, 2^31)."""
    e = int(np.floor(np.log2(m)))
    S = 30 - e
    M = int(round(m * 2.0 ** S))
    if M == 2 ** 31:
        M //= 2
        S -= 1
    assert 1 <= S <= 63, f"shift {S} out of range for multiplier {m}"
    return M, S


def fold(params, name):
    q = params[name]
    if name == "fc":
        return q["w"].reshape(q["w"].shape[0], -1, 1, 1), q["b"].copy()
    g = q["gamma"] / np.sqrt(q["var"] + BN_EPS)
    w = q["w"] * g[:, None, None, None]
    b = q["beta"] - q["mean"] * g
    if name == "conv1":
        # x_norm = (x/255 - mean)/std  ->  act on raw x, pad with 255*mean.
        b = b - (w * (MEAN / STD)[None, :, None, None]).sum(axis=(1, 2, 3))
        w = w / (255.0 * STD)[None, :, None, None]
    return w, b


def quantize(params, sc):
    """Returns list of layer dicts with integer tensors, plus the logit scale."""
    out = []
    logit_scale = None
    for s in layer_specs():
        name = s["name"]
        w, b = fold(params, name)
        s_in = input_scale(name, sc)
        if s["kind"] == "fc":
            s_w = np.full(w.shape[0], np.abs(w).max() / 127.0)
        else:
            s_w = np.abs(w).reshape(w.shape[0], -1).max(axis=1) / 127.0
            s_w[s_w == 0] = 1.0
            # Near-dead channels (BN gamma ~ 0) would need a requant shift
            # above 63; floor their scale, their output is ~ bias anyway.
            s_w = np.maximum(s_w, 1e-3 * s_w.max())
        wq = np.clip(np.round(w / s_w[:, None, None, None]), -127, 127).astype(np.int8)
        bq = np.round(b / (s_w * s_in)).astype(np.int64)
        assert np.abs(bq).max() < 2 ** 31, f"{name}: bias overflows int32"
        if s["kind"] == "fc":
            M = np.zeros(s["cout"], np.int64)
            S = np.zeros(s["cout"], np.int64)
            logit_scale = float(s_w[0] * s_in)
        else:
            MS = [multiplier(s_w[o] * s_in / sc[name]) for o in range(s["cout"])]
            M = np.array([a for a, _ in MS], np.int64)
            S = np.array([c for _, c in MS], np.int64)
        out.append(dict(s, w=wq, bias=bq, M=M, S=S, s_w=s_w, s_in=s_in,
                        s_out=sc.get(name)))
    return out, logit_scale


def write_bin(path, layers, logit_scale):
    with open(path, "wb") as f:
        f.write(b"BDQW")
        f.write(struct.pack("<II", 1, len(layers)))
        f.write(struct.pack("<3BB", *PAD_VALUE.tolist(), 0))
        f.write(struct.pack("<f", logit_scale))
        for L in layers:
            f.write(L["name"].encode().ljust(32, b"\0"))
            f.write(struct.pack("<BBBBHH", KIND[L["kind"]], L["k"], L["stride"],
                                int(L["relu"]), L["cin"], L["cout"]))
            f.write(L["w"].astype("<i1").tobytes())
            f.write(L["bias"].astype("<i4").tobytes())
            f.write(L["M"].astype("<i4").tobytes())
            f.write(L["S"].astype("<u1").tobytes())


def read_bin(path):
    inv = {v: k for k, v in KIND.items()}
    data = open(path, "rb").read()
    assert data[:4] == b"BDQW"
    ver, n = struct.unpack_from("<II", data, 4)
    assert ver == 1
    pad = np.frombuffer(data, np.uint8, 3, 12).copy()
    (logit_scale,) = struct.unpack_from("<f", data, 16)
    off = 20
    layers = []
    for _ in range(n):
        name = data[off:off + 32].rstrip(b"\0").decode()
        kind, k, stride, relu, cin, cout = struct.unpack_from("<BBBBHH", data, off + 32)
        off += 40
        cin_g = 1 if inv[kind] == "dw" else cin
        nw = cout * cin_g * k * k
        w = np.frombuffer(data, "<i1", nw, off).reshape(cout, cin_g, k, k).copy()
        off += nw
        bias = np.frombuffer(data, "<i4", cout, off).astype(np.int64); off += 4 * cout
        M = np.frombuffer(data, "<i4", cout, off).astype(np.int64); off += 4 * cout
        S = np.frombuffer(data, "<u1", cout, off).astype(np.int64); off += cout
        layers.append(dict(name=name, kind=inv[kind], k=k, stride=stride, relu=bool(relu),
                           cin=cin, cout=cout, w=w, bias=bias, M=M, S=S))
    assert off == len(data), "trailing bytes in weights.bin"
    return layers, pad, logit_scale


def write_meta(path, layers, sc, logit_scale):
    meta = dict(pad_value=PAD_VALUE.tolist(), logit_scale=logit_scale,
                activation_scales=sc,
                layers=[dict(name=L["name"], kind=L["kind"], cin=L["cin"], cout=L["cout"],
                             k=L["k"], stride=L["stride"], relu=bool(L["relu"]),
                             input_scale=L["s_in"], output_scale=L["s_out"],
                             weight_scale_min=float(L["s_w"].min()),
                             weight_scale_max=float(L["s_w"].max()),
                             shift_min=int(L["S"].min()), shift_max=int(L["S"].max()))
                        for L in layers])
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
