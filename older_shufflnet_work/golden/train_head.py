#!/usr/bin/env python3
"""Stand-in for the model team's checkpoint: ImageNet ShuffleNetV2 x0.5 with a
new 2-class head trained on a small local dataset (linear probe, backbone
frozen). Needs torch + torchvision.

  python3 train_head.py shufflenetv2_x0.5-f707e7126e.pth DATASET_DIR OUT_PREFIX

DATASET_DIR has train/{drones,non-drones}/ and val/{drones,non-drones}/.
Class 0 = non-drone (bird, plane, ...), class 1 = drone.
Writes OUT_PREFIX.pth (torchvision state_dict) and OUT_PREFIX.npz (for
run_checks.py --npz). Images go through img2bin.load_rgb, i.e. exactly the
224x224 uint8 frame the accelerator sees.

The point is a realistic set of trained weights to check the INT8 pipeline
against PyTorch, not a production classifier.
"""
import glob
import os
import sys

import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from img2bin import load_rgb  # noqa: E402
from model_def import preprocess  # noqa: E402

CLASSES = ("non-drones", "drones")


def list_split(root, split):
    items = []
    for label, cls in enumerate(CLASSES):
        for p in sorted(glob.glob(os.path.join(root, split, cls, "*"))):
            items.append((p, label))
    return items


def to_tensor(imgs):
    return torch.tensor(np.stack([preprocess(i) for i in imgs]), dtype=torch.float32).permute(0, 3, 1, 2)


def features(net, imgs, bs=32):
    """1024-d pooled conv5 features (input of fc)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = to_tensor(imgs[i:i + bs])
            x = net.conv5(net.stage4(net.stage3(net.stage2(net.maxpool(net.conv1(x))))))
            out.append(x.mean([2, 3]))
    return torch.cat(out)


def main():
    pth, root, prefix = sys.argv[1:4]
    torch.manual_seed(0)
    net = torchvision.models.shufflenet_v2_x0_5(weights=None)
    net.load_state_dict(torch.load(pth, map_location="cpu"))
    net.fc = torch.nn.Linear(1024, 2)
    net.eval()

    train, val = list_split(root, "train"), list_split(root, "val")
    print(f"train {len(train)}, val {len(val)} images")
    timgs = [load_rgb(p) for p, _ in train]
    ytr = torch.tensor([l for _, l in train])
    # Horizontal flips double the training set.
    ftr = torch.cat([features(net, timgs), features(net, [i[:, ::-1] for i in timgs])])
    ytr = torch.cat([ytr, ytr])
    fva = features(net, [load_rgb(p) for p, _ in val])
    yva = torch.tensor([l for _, l in val])

    # Class-balanced cross entropy, L2 on the head, full-batch LBFGS.
    w = len(ytr) / (2 * torch.bincount(ytr).float())
    opt = torch.optim.LBFGS(net.fc.parameters(), max_iter=500, line_search_fn="strong_wolfe")
    lossf = torch.nn.CrossEntropyLoss(weight=w)

    def closure():
        opt.zero_grad()
        loss = lossf(net.fc(ftr), ytr) + 1e-3 * net.fc.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc_tr = (net.fc(ftr).argmax(1) == ytr).float().mean().item()
        acc_va = (net.fc(fva).argmax(1) == yva).float().mean().item()
    print(f"train acc {acc_tr:.3f}, val acc {acc_va:.3f}")

    sd = net.state_dict()
    torch.save(sd, prefix + ".pth")
    np.savez(prefix + ".npz", **{k: v.numpy() for k, v in sd.items()})
    print(f"wrote {prefix}.pth and {prefix}.npz")


if __name__ == "__main__":
    main()
