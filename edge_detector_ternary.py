#!/usr/bin/env python3
"""
LANE B, first build: ternary coarse detection at edge scale.

Question: on a COARSE task (binary "animal vs vehicle" from CIFAR-10 -- a
reliable local stand-in for edge presence/detection), does a TINY ternary
model hold float accuracy while being small enough for microcontroller-class
hardware? Reports the three numbers that decide "cheaper specs":
  accuracy (does ternary keep it) + model size (KB) + CPU latency (ms/img).

Contestants (same tiny architecture, only the weight alphabet changes):
  float32 / ternary(QAT) / binary(QAT).

PRE-REGISTERED (before running): on this coarse task, ternary lands within
~2pp of float (coarse tasks are ternary's comfort zone -- matches CIFAR-10 ~=
float) while being ~20x smaller. Binary a touch behind ternary on accuracy.
If ternary drops >5pp, that would flag even coarse detection as too hard for
this tiny a model -- reported either way.

Reuses the exact ternary/binary QAT from binary_vs_ternary_vs_triadic.py.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from binary_vs_ternary_vs_triadic import QuantConv2d, QuantLinear, tq, bq, set_quant, device

EPOCHS, WARMUP, BATCH, SEED = 20, 4, 128, 0
VEHICLES = {0, 1, 8, 9}   # airplane, automobile, ship, truck -> class 1; rest = animals -> 0
LOG2_3 = np.log2(3)


class TinyDetector(nn.Module):
    """~40k-param CNN -- deliberately microcontroller-scale."""
    def __init__(self, qfn, n_classes=2):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                QuantConv2d(ci, co, 3, padding=1, qfn=qfn, quantize=False),
                nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 24), block(24, 48), block(48, 64))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = QuantLinear(64, n_classes, qfn=qfn, quantize=False)
    def forward(self, x):
        x = self.gap(self.features(x)).flatten(1)
        return self.head(x)


def to_binary(y):     # module-level (picklable for Windows DataLoader workers)
    return 1 if y in VEHICLES else 0


def loaders():
    mu, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = transforms.Compose([transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(), transforms.Normalize(mu, std)])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])
    train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tr, target_transform=to_binary)
    test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=te, target_transform=to_binary)
    return (DataLoader(train, BATCH, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test, 256, shuffle=False, num_workers=2, pin_memory=True))


def train(qfn, quantize, loader, label):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = TinyDetector(qfn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        set_quant(model, quantize and ep >= WARMUP); model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); ce(model(x), y).backward(); opt.step()
        sch.step()
    return model


@torch.no_grad()
def accuracy(model, loader, quantize):
    set_quant(model, quantize); model.eval()
    ok = n = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        ok += (pred == y).sum().item(); n += len(y)
    return ok / n * 100


def params_and_sizes(model):
    total = sum(p.numel() for p in model.parameters())
    quant = sum(m.weight.numel() for m in model.modules() if isinstance(m, (QuantConv2d, QuantLinear)))
    other = total - quant
    float_kb = total * 4 / 1024
    tern_kb = (quant * LOG2_3 + other * 32) / 8 / 1024
    bin_kb = (quant * 1 + other * 32) / 8 / 1024
    return total, float_kb, tern_kb, bin_kb


def flops(model):
    macs = [0]
    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            macs[0] += o.shape[1] * o.shape[2] * o.shape[3] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            macs[0] += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 32).to(next(model.parameters()).device))
    for h in hs:
        h.remove()
    return macs[0]


def latency_cpu(model, iters=50, warmup=10):
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
    return (time.perf_counter() - t0) / iters * 1000


def main():
    print("=" * 74)
    print("LANE B -- ternary coarse detection at edge scale (CIFAR animal/vehicle)")
    print("=" * 74)
    tl, el = loaders()
    # majority-class baseline (6 animal classes vs 4 vehicle -> 60% animals)
    print("  task: binary 'is it a vehicle?'  (majority-class baseline = 60.0%)")

    total, fkb, tkb, bkb = params_and_sizes(TinyDetector(tq))
    fl = flops(TinyDetector(tq).to(device))
    print(f"  model: TinyDetector, {total:,} params, {fl/1e6:.1f} MMACs/img @32x32\n")

    print("  training float / ternary / binary (same architecture)...", flush=True)
    fm = train(tq, False, tl, "float"); f_acc = accuracy(fm, el, False)
    tm = train(tq, True, tl, "ternary"); t_acc = accuracy(tm, el, True)
    bm = train(bq, True, tl, "binary"); b_acc = accuracy(bm, el, True)
    lat = latency_cpu(TinyDetector(tq))

    print("\n" + "-" * 74)
    print(f"  {'model':<12}{'accuracy':>10}{'size':>12}{'vs float size':>16}")
    print("  " + "-" * 56)
    print(f"  {'float32':<12}{f_acc:>9.2f}%{fkb:>10.1f}KB{'1.0x':>16}")
    print(f"  {'ternary':<12}{t_acc:>9.2f}%{tkb:>10.1f}KB{fkb/tkb:>14.1f}x")
    print(f"  {'binary':<12}{b_acc:>9.2f}%{bkb:>10.1f}KB{fkb/bkb:>14.1f}x")
    print("  " + "-" * 56)
    print(f"  CPU latency (1 img, 1 thread): {lat:.2f} ms  ({1000/lat:.0f} img/s)")

    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICT")
    print("=" * 74)
    drop = f_acc - t_acc
    print(f"  ternary vs float: {t_acc:.2f}% vs {f_acc:.2f}%  (drop {drop:+.2f}pp)  at {fkb/tkb:.1f}x smaller")
    if drop <= 2.0:
        print("  => WORKS: ternary holds float accuracy on this coarse task while shrinking")
        print(f"     the model to {tkb:.1f} KB -- microcontroller-scale coarse detection is viable.")
    elif drop <= 5.0:
        print(f"  => MARGINAL: ternary costs {drop:.1f}pp here -- usable but not free; a bit more")
        print("     capacity or training would likely close it.")
    else:
        print(f"  => TOO LOSSY: ternary drops {drop:.1f}pp even on this coarse task at this tiny size.")
        print("     Needs more capacity/epochs before the edge pitch holds.")
    print(f"\n  ternary vs binary: {t_acc:.2f}% vs {b_acc:.2f}%  "
          f"(ternary {'>' if t_acc>b_acc else '<='} binary by {abs(t_acc-b_acc):.2f}pp)")
    print("\n[scope] coarse binary proxy for edge detection; next graduation = a named")
    print("        TinyML benchmark (Visual Wake Words / Speech Commands).")


if __name__ == "__main__":
    main()
