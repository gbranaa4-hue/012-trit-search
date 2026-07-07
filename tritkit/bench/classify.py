#!/usr/bin/env python3
"""tritkit benchmark #2 -- finer classification (CIFAR-10, 10-way).

The companion to edge_detect (bench #1). Same tiny model, harder task: 10 classes
instead of a binary detection. This documents the OTHER end of the ledger --
where ternary's cost GROWS because the task needs finer discrimination. Together
the two entries show the gradient: coarse ~= float (bench #1), finer costs more.

Run:  python tritkit/bench/classify.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import tritkit as tk

EPOCHS, WARMUP, BATCH = 25, 5, 128


def build():
    def blk(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                             nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 96),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 10))


def loaders():
    mu, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(), transforms.Normalize(mu, std)])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=te)
    return (DataLoader(train, BATCH, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test, 256, shuffle=False, num_workers=2, pin_memory=True))


@torch.no_grad()
def eval_acc(model, loader, device):
    tk.set_quant(model, True); model.eval()
    ok = n = 0
    for x, y in loader:
        ok += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return ok / n * 100


def main():
    print("=" * 70)
    print("tritkit bench #2 -- finer classification (CIFAR-10, 10-way)")
    print("=" * 70)
    print("  10 classes (chance 10%)\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tl, el = loaders()
    variants = [
        ("float32", lambda: build()),
        ("ternary", lambda: tk.ternarize(build(), keep_first_last=True)),
        ("binary",  lambda: tk.ternarize(build(), keep_first_last=True, binary=True)),
    ]
    rows = {}
    for name, make in variants:
        torch.manual_seed(0)
        model = make()
        tk.qat_fit(model, tl, epochs=EPOCHS, warmup=WARMUP, device=device, log=False)
        acc = eval_acc(model, el, device)
        prof = tk.profile(model.to("cpu"), input_size=(1, 3, 32, 32), device="cpu", show=False)
        rows[name] = dict(acc=acc, **prof)
        print(f"  {name:<9} acc {acc:5.2f}%  {prof['size_kb']:6.1f} KB  {prof['latency_ms']:.2f} ms/img", flush=True)

    f, t = rows["float32"], rows["ternary"]
    print("\n" + "-" * 70)
    print(f"  {'variant':<10}{'acc':>8}{'size':>10}{'vs float':>11}")
    for name, r in rows.items():
        print(f"  {name:<10}{r['acc']:>7.2f}%{r['size_kb']:>8.1f}KB{f['size_kb']/r['size_kb']:>10.1f}x")
    print("-" * 70)
    print(f"  LEDGER: on 10-way (finer), ternary costs {f['acc']-t['acc']:+.1f}pp vs float "
          f"(compare bench #1's -0.4pp on binary detection) -- the granularity gradient.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_classify.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
