#!/usr/bin/env python3
"""
BINARY vs TERNARY vs TRIADIC -- does the number of quant levels drive rotation
robustness, or is it the architecture?

Three contestants on CIFAR-10, evaluated across rotation angles:
  A. BinaryStandardCNN   -- {-1,+1} weights, standard arch   (no zero level)
  B. TernaryStandardCNN  -- {-1,0,+1} weights, standard arch (adds the zero level)
  C. TritCognition       -- {-1,0,+1} weights, TRIADIC arch  (5x5 Light stream + gate)

The design isolates two things cleanly:
  * A vs B  = the effect of the ZERO LEVEL at fixed architecture. This is the
    "ternary beat binary at rotation?" question, controlled: identical backbone,
    identical STE, identical recipe -- the ONLY difference is whether a zero band
    exists (binary uses sign; ternary zeroes |w| < 0.7*mean|w|).
  * B vs C  = the effect of the ARCHITECTURE at fixed ternary quantization.

Prior results in this repo (experiments.py isolation, DOCS.md/FILES.md):
  "ternary weights alone don't cause robustness -- the triadic (5x5) structure
   does." And zero_trit_findings.md refuted the zero-trit-as-uncertainty story.

PRE-REGISTERED PREDICTIONS (committed before running; judged on NORMALIZED
stability = worst_drop/clean, which controls for binary's likely lower clean
accuracy -- a capacity effect, not a robustness effect):
  P1: |NormStab(binary) - NormStab(ternary)| < 5pp  -> the zero level does NOT
      drive rotation robustness; "ternary beats binary at rotation" is NOT
      supported (levels matter for ENERGY, not rotation). If instead ternary is
      >5pp more robust, that REVIVES a levels-matter story -- reported either way.
  P2: NormStab(triadic) lower (more robust) than BOTH standard models by a real
      margin -> architecture (5x5 receptive field) is the cause, consistent with
      the prior ablation.

CAVEAT committed up front: 30 epochs (not the 50 of the full study) for a
tractable three-way race; one seed each. Judges DIRECTION/ordering, not a
converged benchmark. Same bilinear-rotation eval and modeled-quant caveats as
the parent experiments.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
QUANT_WARMUP = 6
BATCH = 128
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
SEED = 0


# ────────────────────────────────────────────────────────────────────────────
# Quantizers -- ternary (from experiments.py) and a MATCHED binary (sign, same STE)
# ────────────────────────────────────────────────────────────────────────────
class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        t = 0.7 * x.abs().mean()
        ctx.save_for_backward(x)
        return torch.where(x > t, torch.ones_like(x),
               torch.where(x < -t, -torch.ones_like(x), torch.zeros_like(x)))
    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).float()


class BinaryQuantize(torch.autograd.Function):
    """Matched to TernaryQuantize but with NO zero band: sign(x) -> {-1,+1},
    identical straight-through estimator (clip at |x|<=1)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).float()


tq = TernaryQuantize.apply
bq = BinaryQuantize.apply


class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, qfn=tq, quantize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.qfn = qfn; self.do_quantize = quantize
    def forward(self, x):
        w = self.qfn(self.weight) if self.do_quantize else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding)


class QuantLinear(nn.Linear):
    def __init__(self, *args, qfn=tq, quantize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.qfn = qfn; self.do_quantize = quantize
    def forward(self, x):
        w = self.qfn(self.weight) if self.do_quantize else self.weight
        return F.linear(x, w, self.bias)


def set_quant(model, active):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            m.do_quantize = active


def zero_fraction(model):
    """Fraction of quantized weights that are exactly 0 (ternary) -- 0 for binary."""
    tot = z = 0
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            q = m.qfn(m.weight.data)
            tot += q.numel(); z += (q == 0).sum().item()
    return z / tot if tot else 0.0


# ────────────────────────────────────────────────────────────────────────────
# Standard CNN (parametrized by quantizer) -- models A and B
# ────────────────────────────────────────────────────────────────────────────
class StdBlock(nn.Module):
    def __init__(self, in_ch, out_ch, qfn):
        super().__init__()
        self.conv = nn.Sequential(
            QuantConv2d(in_ch, out_ch, 3, padding=1, qfn=qfn, quantize=False),
            nn.BatchNorm2d(out_ch), nn.ReLU())
    def forward(self, x): return self.conv(x)


class QuantStandardCNN(nn.Module):
    """Same backbone as the repo's StandardCNN/TernaryStandardCNN; the quantizer
    (tq or bq) is the only thing that changes between the binary and ternary runs."""
    def __init__(self, qfn, num_classes=10, in_ch=3):
        super().__init__()
        self.b1 = StdBlock(in_ch, 96, qfn)
        self.b2 = StdBlock(96, 192, qfn)
        self.b3 = StdBlock(192, 128, qfn)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls = QuantLinear(128, num_classes, qfn=qfn, quantize=False)
    def forward(self, x):
        x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
        return self.cls(self.gap(x).squeeze(-1).squeeze(-1))


# ────────────────────────────────────────────────────────────────────────────
# Triadic model (verbatim from experiments.py) -- model C
# ────────────────────────────────────────────────────────────────────────────
class PredictiveTritBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.s0 = nn.Sequential(QuantConv2d(in_ch, out_ch, 1, qfn=tq, quantize=False), nn.BatchNorm2d(out_ch))
        self.s1 = nn.Sequential(QuantConv2d(in_ch, out_ch, 3, padding=1, qfn=tq, quantize=False), nn.BatchNorm2d(out_ch))
        self.s2 = nn.Sequential(QuantConv2d(in_ch, out_ch, 5, padding=2, qfn=tq, quantize=False), nn.BatchNorm2d(out_ch))
        self.pred = QuantConv2d(out_ch, in_ch, 1, qfn=tq, quantize=False)
    def forward(self, x):
        s0 = torch.sigmoid(self.s0(x)); s1 = torch.tanh(self.s1(x)); s2 = torch.tanh(self.s2(x))
        out = s1 * (1 - s0) + s2 * s0
        return out, self.pred(out), x


class PredLoss(nn.Module):
    def __init__(self, w=0.01):
        super().__init__(); self.w = w; self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, labels, preds):
        cls = self.ce(logits, labels)
        pred = sum(F.mse_loss(p, a.detach()) for p, a in preds)
        return cls + self.w * pred


class TritCognition(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.b1 = PredictiveTritBlock(in_ch, 32)
        self.b2 = PredictiveTritBlock(32, 64)
        self.b3 = PredictiveTritBlock(64, 128)
        self.pool = nn.MaxPool2d(2); self.gap = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            QuantConv2d(128, 32, 1, qfn=tq, quantize=False), nn.ReLU(),
            QuantConv2d(32, 1, 1, qfn=tq, quantize=False), nn.Sigmoid())
        self.mem = nn.Sequential(QuantLinear(128, 128, qfn=tq, quantize=False), nn.Sigmoid())
        self.cls = QuantLinear(128, num_classes, qfn=tq, quantize=False)
    def forward(self, x):
        preds = []
        o, p, i = self.b1(x); preds.append((p, i)); o = self.pool(o)
        o, p, i = self.b2(o); preds.append((p, i)); o = self.pool(o)
        o, p, i = self.b3(o); preds.append((p, i)); o = self.pool(o)
        o = o * self.attn(o)
        feat = self.gap(o).squeeze(-1).squeeze(-1)
        feat = feat * self.mem(feat)
        return self.cls(feat), preds


# ────────────────────────────────────────────────────────────────────────────
def get_cifar10():
    mu, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(mu, std)])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])
    train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=te)
    return train, test


def train_std(model, loader, label):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        set_quant(model, epoch >= QUANT_WARMUP); model.train(); tot = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(); loss = ce(model(imgs), labels); loss.backward(); opt.step()
            tot += loss.item()
        sch.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            ph = "QUANT" if epoch >= QUANT_WARMUP else "warmup"
            print(f"  [{label}|{ph}] epoch {epoch+1:>2}/{EPOCHS} loss={tot/len(loader):.4f}", flush=True)


def train_trit(model, loader, label):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lf = PredLoss(0.01)
    for epoch in range(EPOCHS):
        set_quant(model, epoch >= QUANT_WARMUP); model.train(); tot = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(); logits, preds = model(imgs)
            loss = lf(logits, labels, preds); loss.backward(); opt.step()
            tot += loss.item()
        sch.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            ph = "QUANT" if epoch >= QUANT_WARMUP else "warmup"
            print(f"  [{label}|{ph}] epoch {epoch+1:>2}/{EPOCHS} loss={tot/len(loader):.4f}", flush=True)


@torch.no_grad()
def evaluate(model, loader, n, triadic=False):
    set_quant(model, True); model.eval(); res = {}
    for angle in ANGLES:
        correct = 0
        for imgs, labels in loader:
            imgs = TF.rotate(imgs.to(device), angle,
                             interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            out = model(imgs)
            if triadic: out = out[0]
            correct += (out.argmax(1) == labels.to(device)).sum().item()
        res[angle] = correct / n * 100
    return res


def norm_stability(acc):
    return (acc[0] - min(acc.values())) / acc[0] * 100


def main():
    print(f"Device: {device}  |  {EPOCHS} epochs, warmup {QUANT_WARMUP}, "
          f"angles {ANGLES}", flush=True)
    train, test = get_cifar10()
    tl = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    el = DataLoader(test, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    n_test = len(test)

    contestants = [
        ("Binary-std", lambda: QuantStandardCNN(bq), train_std, False),
        ("Ternary-std", lambda: QuantStandardCNN(tq), train_std, False),
        ("Triadic", lambda: TritCognition(), train_trit, True),
    ]

    results = {}
    for name, build, trainer, triadic in contestants:
        torch.manual_seed(SEED); np.random.seed(SEED)
        print(f"\n=== training {name} ===", flush=True)
        model = build().to(device)
        trainer(model, tl, name)
        acc = evaluate(model, el, n_test, triadic=triadic)
        zf = zero_fraction(model)
        results[name] = dict(acc=acc, zero_frac=zf,
                             clean=acc[0], floor=min(acc.values()),
                             worst_drop=acc[0] - min(acc.values()),
                             norm_stab=norm_stability(acc))
        print(f"  -> clean {acc[0]:.2f}%  floor {min(acc.values()):.2f}%  "
              f"norm_stab {norm_stability(acc):.2f}%  zero_frac {zf:.3f}", flush=True)

    # ── report ──
    print("\n" + "=" * 74)
    print("RESULTS  (norm_stab = worst_drop/clean; LOWER = more rotation-robust)")
    print("=" * 74)
    print(f"  {'model':<14}{'clean':>8}{'floor':>8}{'worst_drop':>12}{'norm_stab':>11}{'zero%':>8}")
    for name in results:
        r = results[name]
        print(f"  {name:<14}{r['clean']:>7.2f}%{r['floor']:>7.2f}%{r['worst_drop']:>11.2f}pp"
              f"{r['norm_stab']:>10.2f}%{r['zero_frac']*100:>7.1f}%")

    print(f"\n  {'angle':>6}" + "".join(f"{n:>13}" for n in results))
    for a in ANGLES:
        print(f"  {a:>5}°" + "".join(f"{results[n]['acc'][a]:>12.2f}%" for n in results))

    # ── pre-registered verdicts ──
    nb, nt, nc = (results[k]['norm_stab'] for k in ("Binary-std", "Ternary-std", "Triadic"))
    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 74)
    p1_gap = nt - nb
    p1_levels_inert = abs(p1_gap) < 5.0
    print(f"\nP1 (does the ZERO LEVEL drive rotation robustness?):")
    print(f"    binary norm_stab {nb:.2f}%  vs  ternary {nt:.2f}%   (d={p1_gap:+.2f}pp)")
    if p1_levels_inert:
        print("    => NO. Binary ~ ternary in rotation robustness. The zero level does")
        print("       NOT drive it -> 'ternary beats binary at rotation' NOT supported.")
        print("       (Ternary's win over binary is ENERGY: ~1.1pJ vs ~4.6pJ per op.)")
    elif p1_gap < 0:
        print(f"    => YES, and it REVIVES a levels-matter story: ternary is {-p1_gap:.1f}pp")
        print("       more rotation-robust than binary at identical architecture.")
    else:
        print(f"    => Binary is MORE robust than ternary ({p1_gap:.1f}pp) -- unexpected;")
        print("       report and investigate before interpreting.")

    p2_margin = min(nb, nt) - nc
    print(f"\nP2 (does the ARCHITECTURE drive it?):")
    print(f"    triadic norm_stab {nc:.2f}%  vs  best standard {min(nb,nt):.2f}%   "
          f"(triadic more robust by {p2_margin:+.2f}pp)")
    if p2_margin > 2.0:
        print("    => YES. Triadic beats both standard models -> the 5x5 receptive-field")
        print("       architecture is the robustness driver, matching the prior ablation.")
    else:
        print("    => Triadic does NOT clearly beat the standard models here -- weaker than")
        print("       the prior 50-epoch result; note the reduced budget, report as-is.")

    print("\n[scope] 30 epochs, one seed, bilinear rotation, modeled quant. Judges the")
    print("        ordering (levels vs architecture), not a converged benchmark.")

    os.makedirs("results", exist_ok=True)
    with open("results/binary_vs_ternary_vs_triadic.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/binary_vs_ternary_vs_triadic.json")


if __name__ == "__main__":
    main()
