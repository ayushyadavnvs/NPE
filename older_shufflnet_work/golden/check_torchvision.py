#!/usr/bin/env python3
"""Proves the NumPy float model (model_def.py) is the same network as
torchvision's shufflenet_v2_x0_5(num_classes=NUM_CLASSES). Needs torch + torchvision.

Loads the synthetic weights into both models and compares logits on images/,
then checks the .npz route (export_npz.py -> params_from_state_dict).
Expected: max difference ~1e-14 (float64 rounding) and no unmapped keys.
"""
import glob
import os
import sys
import tempfile

import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from img2bin import load_rgb  # noqa: E402
from model_def import (NUM_CLASSES, FloatModel, layer_specs, params_from_state_dict,  # noqa: E402
                       preprocess, synthetic_params, torchvision_keys)

BN_FIELDS = (("weight", "gamma"), ("bias", "beta"), ("running_mean", "mean"), ("running_var", "var"))


def main():
    imgs = [load_rgb(p) for p in sorted(glob.glob(os.path.join(HERE, "images", "*")))]
    params = synthetic_params(imgs)

    net = torchvision.models.shufflenet_v2_x0_5(weights=None, num_classes=NUM_CLASSES).eval().double()
    sd = net.state_dict()
    used = {"fc.weight", "fc.bias"}
    for s in layer_specs():
        q = params[s["name"]]
        if s["kind"] == "fc":
            sd["fc.weight"][:] = torch.tensor(q["w"])
            sd["fc.bias"][:] = torch.tensor(q["b"])
            continue
        ck, bk = torchvision_keys(s["name"])
        sd[f"{ck}.weight"][:] = torch.tensor(q["w"])
        used.add(f"{ck}.weight")
        for tk, ok in BN_FIELDS:
            sd[f"{bk}.{tk}"][:] = torch.tensor(q[ok])
            used.add(f"{bk}.{tk}")
        used.add(f"{bk}.num_batches_tracked")
    net.load_state_dict(sd)
    print("unmapped torchvision keys:", sorted(set(sd) - used))

    fm = FloatModel(params)
    worst = 0.0
    for img in imgs:
        x = torch.tensor(preprocess(img)).permute(2, 0, 1)[None]
        with torch.no_grad():
            t = net(x)[0].numpy()
        f = fm.forward(img)
        worst = max(worst, float(np.abs(t - f).max()))
        print(f"torchvision {np.round(t, 6)}  numpy {np.round(f, 6)}")
    print(f"max logit difference: {worst:.2e}")

    tmp = os.path.join(tempfile.mkdtemp(), "sd.npz")
    np.savez(tmp, **{k: v.numpy() for k, v in net.state_dict().items()})
    p2 = params_from_state_dict(dict(np.load(tmp)))
    d = np.abs(FloatModel(p2).forward(imgs[0]) - fm.forward(imgs[0])).max()
    print(f"npz route difference: {d:.2e}")
    return 0 if worst < 1e-9 and d < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
