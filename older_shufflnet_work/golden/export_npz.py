#!/usr/bin/env python3
"""For the model team: save a trained torchvision ShuffleNetV2 x0.5 checkpoint
(2-class fc) as .npz so the golden-model flow can run without PyTorch.

  python3 export_npz.py checkpoint.pth shufflenet_2class.npz

Accepts either a bare state_dict or a dict with a 'state_dict'/'model' entry.
"""
import sys

import numpy as np
import torch


def main():
    ckpt = torch.load(sys.argv[1], map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
    elif hasattr(ckpt, "state_dict"):
        ckpt = ckpt.state_dict()
    sd = {k.replace("module.", "", 1): v.detach().cpu().numpy()
          for k, v in ckpt.items() if hasattr(v, "detach")}
    assert sd["fc.weight"].shape[0] == 2, f"fc has {sd['fc.weight'].shape[0]} outputs, expected 2"
    np.savez(sys.argv[2], **sd)
    print(f"wrote {len(sd)} tensors to {sys.argv[2]}")


if __name__ == "__main__":
    main()
