"""Float ShuffleNetV2 x0.5 (torchvision layout, 3-class head) in NumPy.

All tensors are HWC (height, width, channels), batch handled by looping.
Layer names match tools/budget.py and the INT8 models:
  conv1, s{stage}u{unit}.{pw1,dw,pw2}, s{stage}u0.b1.{dw,pw}, conv5, fc
For downsample units (u0) the main branch is named s{stage}u0.{pw1,dw,pw2}.

Weights come from one of:
  * a torchvision-style state_dict saved as .npz (tools: export_npz.py), or
  * synthetic_params(): He-initialised convolutions whose BatchNorm statistics
    are measured on real images, so activations have realistic ranges. Used
    until the model team's trained checkpoint arrives.
"""
import numpy as np

STAGE_OUT = (24, 48, 96, 192, 1024)
REPEATS = (4, 8, 4)
NUM_CLASSES = 3  # 0 other, 1 bird, 2 drone (clean_v2 label order)
BN_EPS = 1e-5
# torchvision ImageNet normalisation; replace with the model team's values.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)


def layer_specs():
    """Ordered list of convolution layers: dict(name, kind, cin, cout, k, stride, relu)."""
    L = []

    def add(name, kind, cin, cout, k, stride, relu):
        L.append(dict(name=name, kind=kind, cin=cin, cout=cout, k=k, stride=stride, relu=relu))

    add("conv1", "std", 3, STAGE_OUT[0], 3, 2, True)
    cin = STAGE_OUT[0]
    for st, (c, reps) in enumerate(zip(STAGE_OUT[1:4], REPEATS), start=2):
        half = c // 2
        add(f"s{st}u0.b1.dw", "dw", cin, cin, 3, 2, False)
        add(f"s{st}u0.b1.pw", "pw", cin, half, 1, 1, True)
        add(f"s{st}u0.pw1", "pw", cin, half, 1, 1, True)
        add(f"s{st}u0.dw", "dw", half, half, 3, 2, False)
        add(f"s{st}u0.pw2", "pw", half, half, 1, 1, True)
        for u in range(1, reps):
            add(f"s{st}u{u}.pw1", "pw", half, half, 1, 1, True)
            add(f"s{st}u{u}.dw", "dw", half, half, 3, 1, False)
            add(f"s{st}u{u}.pw2", "pw", half, half, 1, 1, True)
        cin = c
    add("conv5", "pw", STAGE_OUT[3], STAGE_OUT[4], 1, 1, True)
    add("fc", "fc", STAGE_OUT[4], NUM_CLASSES, 1, 1, False)
    return L


def units():
    """(prefix, downsample) for each of the 16 shuffle units, in order."""
    out = []
    for st, reps in zip((2, 3, 4), REPEATS):
        for u in range(reps):
            out.append((f"s{st}u{u}", u == 0))
    return out


# --------------------------------------------------------------------------
# Float primitives (HWC)
# --------------------------------------------------------------------------
def conv2d(x, w, stride, groups, pad_value=None):
    """x: HxWxCin, w: Cout x Cin/groups x k x k (torch layout). Zero padding k//2
    unless pad_value (per input channel) is given."""
    H, W, C = x.shape
    cout, cin_g, k, _ = w.shape
    p = k // 2
    if p:
        xp = np.empty((H + 2 * p, W + 2 * p, C), dtype=x.dtype)
        xp[...] = 0 if pad_value is None else pad_value
        xp[p:p + H, p:p + W] = x
    else:
        xp = x
    Ho = (H + 2 * p - k) // stride + 1
    Wo = (W + 2 * p - k) // stride + 1
    # patches: Ho x Wo x k x k x C
    patches = np.empty((Ho, Wo, k, k, C), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            patches[:, :, i, j, :] = xp[i:i + stride * Ho:stride, j:j + stride * Wo:stride, :]
    if groups == 1:
        wm = w.transpose(2, 3, 1, 0).reshape(k * k * C, cout)  # (k,k,cin) x cout
        return patches.reshape(Ho, Wo, k * k * C) @ wm
    assert groups == C == cout and cin_g == 1
    return np.einsum("hwijc,cij->hwc", patches, w[:, 0])


def maxpool3x3s2(x):
    H, W, C = x.shape
    xp = np.full((H + 2, W + 2, C), -np.inf if x.dtype.kind == "f" else 0, dtype=x.dtype)
    xp[1:H + 1, 1:W + 1] = x
    Ho, Wo = (H - 1) // 2 + 1, (W - 1) // 2 + 1
    out = None
    for i in range(3):
        for j in range(3):
            v = xp[i:i + 2 * Ho:2, j:j + 2 * Wo:2]
            out = v if out is None else np.maximum(out, v)
    return out


def channel_shuffle(x):
    """torchvision channel_shuffle with groups=2 on HWC: out[2i]=x[i], out[2i+1]=x[C/2+i]."""
    H, W, C = x.shape
    return x.reshape(H, W, 2, C // 2).transpose(0, 1, 3, 2).reshape(H, W, C)


def preprocess(img_u8):
    return (img_u8.astype(np.float64) / 255.0 - MEAN) / STD


# --------------------------------------------------------------------------
# Float forward
# --------------------------------------------------------------------------
class FloatModel:
    """params[name] = dict(w=..., gamma, beta, mean, var) for conv layers,
    dict(w=(classes x 1024), b=(classes,)) for fc."""

    def __init__(self, params):
        self.p = params
        self.specs = {s["name"]: s for s in layer_specs()}

    def layer(self, name, x, record=None, init_bn=False):
        s, q = self.specs[name], self.p[name]
        groups = s["cin"] if s["kind"] == "dw" else 1
        y = conv2d(x, q["w"], s["stride"], groups)
        if init_bn and q.get("mean") is None:
            q["mean"] = y.reshape(-1, y.shape[-1]).mean(0)
            q["var"] = y.reshape(-1, y.shape[-1]).var(0)
        y = (y - q["mean"]) / np.sqrt(q["var"] + BN_EPS) * q["gamma"] + q["beta"]
        if s["relu"]:
            y = np.maximum(y, 0)
        if record is not None:
            record[name] = y
        return y

    def unit(self, prefix, down, x, record=None, init_bn=False):
        f = lambda n, t: self.layer(f"{prefix}.{n}", t, record, init_bn)
        if down:
            b1 = f("b1.pw", f("b1.dw", x))
            b2 = f("pw2", f("dw", f("pw1", x)))
        else:
            c = x.shape[-1] // 2
            b1 = x[..., :c]
            b2 = f("pw2", f("dw", f("pw1", x[..., c:])))
        y = channel_shuffle(np.concatenate([b1, b2], axis=-1))
        if record is not None:
            record[f"{prefix}.out"] = y
        return y

    def input(self, img_u8):
        return preprocess(img_u8)

    def forward(self, img_u8, record=None, init_bn=False):
        x = self.input(img_u8)
        x = self.layer("conv1", x, record, init_bn)
        x = maxpool3x3s2(x)
        if record is not None:
            record["maxpool"] = x
        for prefix, down in units():
            x = self.unit(prefix, down, x, record, init_bn)
        x = self.layer("conv5", x, record, init_bn)
        g = x.mean(axis=(0, 1))
        logits = self.p["fc"]["w"] @ g + self.p["fc"]["b"]
        if record is not None:
            record["gap"] = g
            record["logits"] = logits
        return logits


def synthetic_params(calib_images, seed=0):
    """He-initialised weights; BN running stats measured on calib_images
    (layer by layer, using the first image batch) so activations look real."""
    rng = np.random.default_rng(seed)
    params = {}
    for s in layer_specs():
        if s["kind"] == "fc":
            params["fc"] = dict(w=rng.normal(0, 1 / np.sqrt(s["cin"]), (s["cout"], s["cin"])),
                                b=rng.normal(0, 0.1, s["cout"]))
            continue
        cin_g = 1 if s["kind"] == "dw" else s["cin"]
        fan_in = cin_g * s["k"] * s["k"]
        params[s["name"]] = dict(
            w=rng.normal(0, np.sqrt(2.0 / fan_in), (s["cout"], cin_g, s["k"], s["k"])),
            gamma=rng.uniform(0.5, 1.5, s["cout"]),
            beta=rng.normal(0.0, 0.2, s["cout"]),
            mean=None, var=None)
    # Each BN takes the statistics of its own input on the first calibration
    # image, layer by layer, so every layer starts out roughly normalised.
    FloatModel(params).forward(calib_images[0], init_bn=True)
    return params


# torchvision state_dict key prefixes for each of our layer names.
def torchvision_keys(name):
    if name == "conv1":
        return "conv1.0", "conv1.1"
    if name == "conv5":
        return "conv5.0", "conv5.1"
    st, rest = name.split("u", 1)
    u, part = rest.split(".", 1)
    base = f"stage{st[1:]}.{u}"
    table = {"b1.dw": ("branch1.0", "branch1.1"), "b1.pw": ("branch1.2", "branch1.3"),
             "pw1": ("branch2.0", "branch2.1"), "dw": ("branch2.3", "branch2.4"),
             "pw2": ("branch2.5", "branch2.6")}
    c, b = table[part]
    return f"{base}.{c}", f"{base}.{b}"


def params_from_state_dict(sd):
    """sd: mapping torchvision key -> numpy array (NUM_CLASSES-class fc)."""
    params = {}
    for s in layer_specs():
        if s["kind"] == "fc":
            params["fc"] = dict(w=np.asarray(sd["fc.weight"], np.float64),
                                b=np.asarray(sd["fc.bias"], np.float64))
            continue
        ck, bk = torchvision_keys(s["name"])
        params[s["name"]] = dict(
            w=np.asarray(sd[f"{ck}.weight"], np.float64),
            gamma=np.asarray(sd[f"{bk}.weight"], np.float64),
            beta=np.asarray(sd[f"{bk}.bias"], np.float64),
            mean=np.asarray(sd[f"{bk}.running_mean"], np.float64),
            var=np.asarray(sd[f"{bk}.running_var"], np.float64))
    return params
