"""NumPy INT8 reference, written independently of the C++ golden model
(vectorised im2col here, plain loops there) so the two can check each other.

Reads weights.bin only, so it also proves the file round-trips.
"""
import numpy as np

from model_def import channel_shuffle, conv2d, maxpool3x3s2, units


def requant(acc, M, S, relu):
    acc = acc.astype(np.int64)
    assert np.abs(acc).max() < 2 ** 31, "accumulator exceeds int32"
    y = (acc * M + (np.int64(1) << (S - 1))) >> S  # arithmetic shift = floor
    return np.clip(y, 0, 255) if relu else np.clip(y, -128, 127)


class Int8Model:
    def __init__(self, layers, pad_value):
        self.L = {l["name"]: l for l in layers}
        self.pad = pad_value.astype(np.int64)

    def layer(self, name, x, rec):
        l = self.L[name]
        groups = l["cin"] if l["kind"] == "dw" else 1
        pad = self.pad if name == "conv1" else None
        acc = conv2d(x.astype(np.int64), l["w"].astype(np.int64), l["stride"], groups, pad)
        y = requant(acc + l["bias"], l["M"], l["S"], l["relu"])
        rec[name] = y
        return y

    def forward(self, img_u8):
        """img_u8: 224x224x3. Returns (logits int64[NUM_CLASSES], dict of every tensor)."""
        rec = {"input": img_u8.astype(np.int64)}
        x = self.layer("conv1", rec["input"], rec)
        x = maxpool3x3s2(x)
        rec["maxpool"] = x
        for prefix, down in units():
            f = lambda n, t: self.layer(f"{prefix}.{n}", t, rec)
            if down:
                b1 = f("b1.pw", f("b1.dw", x))
                b2 = f("pw2", f("dw", f("pw1", x)))
            else:
                c = x.shape[-1] // 2
                b1 = x[..., :c]
                b2 = f("pw2", f("dw", f("pw1", x[..., c:])))
            x = channel_shuffle(np.concatenate([b1, b2], axis=-1))
            rec[f"{prefix}.out"] = x
        x = self.layer("conv5", x, rec)
        gap = x.sum(axis=(0, 1))
        rec["gap"] = gap
        fc = self.L["fc"]
        logits = fc["w"].reshape(fc["cout"], -1).astype(np.int64) @ gap + fc["bias"]
        assert np.abs(logits).max() < 2 ** 31, "fc accumulator exceeds int32"
        rec["logits"] = logits
        return logits, rec
