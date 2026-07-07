#!/usr/bin/env python3
"""tritkit benchmark #1 -- coarse edge detection (CIFAR animal/vehicle).

Dogfoods the public tritkit API end-to-end: build a plain tiny CNN, then
  float   = train it as-is
  ternary = tk.ternarize(...) + tk.qat_fit(...)
  binary  = tk.ternarize(..., binary=True) + tk.qat_fit(...)
and report accuracy + size + latency for each. A coarse binary task ("is it a
vehicle?") is ternary's comfort zone -- this entry documents the WIN end of the
ledger. (Fine-identity / PTQ entries document the FAIL end.)

Run:  python tritkit/bench/edge_detect.py
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

VEHICLES = {0, 1, 8, 9}       # airplane, automobile, ship, truck -> 1; rest (animals) -> 0
EPOCHS, WARMUP, BATCH = 20, 4, 128


def to_binary(y):             # module-level -> picklable for Windows DataLoader
    return 1 if y in VEHICLES else 0


def build():
    def blk(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                             nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(3, 24), blk(24, 48), blk(48, 64),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))


def loaders():
    mu, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(), transforms.Normalize(mu, std)])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tr, target_transform=to_binary)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=te, target_transform=to_binary)
    return (DataLoader(train, BATCH, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test, 256, shuffle=False, num_workers=2, pin_memory=True))


@torch.no_grad()
def eval_acc(model, loader, device):
    tk.set_quant(model, True)      # no-op on plain float model; turns quant on for ternarized
    model.eval()
    ok = n = 0
    for x, y in loader:
        ok += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        n += len(y)
    return ok / n * 100


def run():
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
        print(f"  {name:<9} acc {acc:5.2f}%  {prof['size_kb']:6.1f} KB  "
              f"{prof['mmacs']:.1f} MMACs  {prof['latency_ms']:.2f} ms/img", flush=True)
    return rows


def main():
    print("=" * 70)
    print("tritkit bench #1 -- coarse edge detection (CIFAR animal/vehicle)")
    print("=" * 70)
    print("  task: binary 'is it a vehicle?'  (majority-class baseline 60.0%)\n")
    rows = run()

    f, t, b = rows["float32"], rows["ternary"], rows["binary"]
    print("\n" + "-" * 70)
    print(f"  {'variant':<10}{'acc':>8}{'size':>10}{'vs float':>11}{'latency':>12}")
    for name, r in rows.items():
        print(f"  {name:<10}{r['acc']:>7.2f}%{r['size_kb']:>8.1f}KB"
              f"{f['size_kb']/r['size_kb']:>10.1f}x{r['latency_ms']:>10.2f}ms")
    print("-" * 70)
    print(f"  LEDGER: ternary holds {t['acc']:.1f}% vs {f['acc']:.1f}% float "
          f"({f['acc']-t['acc']:+.1f}pp) at {f['size_kb']/t['size_kb']:.1f}x smaller -- "
          f"coarse detection is ternary's WIN regime.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_edge_detect.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
