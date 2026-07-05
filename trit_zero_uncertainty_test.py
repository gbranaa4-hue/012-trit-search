#!/usr/bin/env python3
"""
Prediction 3 of the Observer-Shadow-Light theory, tested properly.

paper/observer_shadow_light_theory.md makes three testable predictions.
Predictions 1-2 were tested in trit_stream_ablation_test.py (1 confirmed,
2 weak-with-interaction). Prediction 3 -- "zero-trit activation fraction
increases under rotation, and the increase is larger for the triadic model
than for a gate-free ternary CNN" -- was never run: the earlier study
measured WEIGHT sparsity (fixed at inference, unchanged by rotation, as
expected) and left activation-level measurement as a follow-up. The
ablation script's measure_activation_sparsity() samples only 5 batches
and its output was never reported in findings.

THE THEORY'S CLAIM: the Observer gate output s0 ~ 0.5 means "withholding
judgment"; the blend out = Shadow*(1-s0) + Light*s0 then produces
intermediate values that ternary quantization rounds to 0. Rotation makes
Shadow ambiguous without fully resolving via Light, so genuinely uncertain
spatial locations -- and therefore zero activations -- should increase.
A standard ternary CNN has no Observer gate and no reason to show the
same increase.

SETUP: exact model classes from trit_stream_ablation_test.py (TritModel
full config light_k=5, multiplicative gate, pred loss 0.01; TernaryStdCNN
floor baseline), same training recipe (Adam 1e-3, cosine schedule,
EPOCHS=20, QUANT_WARMUP=4, batch 512, standard CIFAR-10 augmentation).
One training run per model, matching the ablation study's precedent
(disclosed limitation: no model-seed variance estimate).

MEASUREMENT (the part the earlier script didn't do):
  - zero-trit activation fraction = fraction of BLOCK-OUTPUT values that
    ternary quantization (threshold 0.7*mean|a|, per batch) maps to 0.
    For TritModel the hooked tensor is the triadic blend `out` of each
    block (the quantity the theory speaks about); for TernaryStdCNN it is
    each block's post-ReLU output.
  - measured over the FULL 10k test set (not 5 batches), at angles
    0/45/90/135/180 for the curve; the pre-registered contrast is 0 vs 90.
  - paired per-batch statistics: each test batch contributes a
    (sparsity@0, sparsity@90) pair; paired t across batches.
  - mechanism check (TritModel only): Observer-gate uncertainty fraction,
    mean fraction of gate values with |s0 - 0.5| < 0.1, same pairing.

PREDICTIONS, pre-registered before the first run and not edited after:
  P1. TritFull zero-trit activation fraction is higher at 90 deg than at
      0 deg (paired t > 2 across test batches).
  P2. The 90-vs-0 increase is larger for TritFull than for TernaryStdCNN
      (Welch t > 2 on the per-batch paired differences).
  P3. (mechanism) TritFull's Observer-gate uncertainty fraction
      (|s0-0.5| < 0.1) is higher at 90 deg than at 0 deg (paired t > 2).

If P1/P2 hold but P3 fails, the sparsity increase is real but not driven
by the gate -- the theory's mechanism claim would be refuted even if its
surface prediction survives. All three are reported either way.

Run it:
    python trit_zero_uncertainty_test.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n", flush=True)

EPOCHS = 20
QUANT_WARMUP = 4
CURVE_ANGLES = [0, 45, 90, 135, 180]
CONTRAST = (0, 90)   # pre-registered
GATE_BAND = 0.1      # |s0 - 0.5| < GATE_BAND counts as "withholding judgment"

# ── Ternary core (verbatim from trit_stream_ablation_test.py) ────────────────

class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        t = 0.7 * x.abs().mean()
        ctx.save_for_backward(x)
        return torch.where(x > t,  torch.ones_like(x),
               torch.where(x < -t, -torch.ones_like(x),
                           torch.zeros_like(x)))
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).float()

tq = TernaryQuantize.apply

class TernaryConv2d(nn.Conv2d):
    def __init__(self, *a, quantize=False, **k):
        super().__init__(*a, **k)
        self.do_quantize = quantize
    def forward(self, x):
        return F.conv2d(x, tq(self.weight) if self.do_quantize else self.weight,
                        self.bias, self.stride, self.padding)

class TernaryLinear(nn.Linear):
    def __init__(self, *a, quantize=False, **k):
        super().__init__(*a, **k)
        self.do_quantize = quantize
    def forward(self, x):
        return F.linear(x, tq(self.weight) if self.do_quantize else self.weight, self.bias)

def set_quant(model, active):
    for m in model.modules():
        if isinstance(m, (TernaryConv2d, TernaryLinear)):
            m.do_quantize = active

# ── Models (verbatim, plus capture flags for the measurement) ─────────────────

class TritBlock(nn.Module):
    def __init__(self, in_ch, out_ch, light_k=5, additive=False):
        super().__init__()
        self.additive = additive
        self.capture = False
        self.last_s0 = None
        self.last_out = None
        self.s0 = nn.Sequential(
            TernaryConv2d(in_ch, out_ch, 1, quantize=False),
            nn.BatchNorm2d(out_ch))
        self.s1 = nn.Sequential(
            TernaryConv2d(in_ch, out_ch, 3, padding=1, quantize=False),
            nn.BatchNorm2d(out_ch))
        pad2 = light_k // 2
        self.s2 = nn.Sequential(
            TernaryConv2d(in_ch, out_ch, light_k, padding=pad2, quantize=False),
            nn.BatchNorm2d(out_ch))
        self.pred = TernaryConv2d(out_ch, in_ch, 1, quantize=False)

    def forward(self, x):
        s0 = torch.sigmoid(self.s0(x))
        s1 = torch.tanh(self.s1(x))
        s2 = torch.tanh(self.s2(x))
        if self.additive:
            out = s1 + s2
        else:
            out = s1 * (1 - s0) + s2 * s0
        if self.capture:
            self.last_s0 = s0.detach()
            self.last_out = out.detach()
        return out, self.pred(out), x

class TritModel(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, light_k=5, additive=False):
        super().__init__()
        self.b1   = TritBlock(in_ch, 32,  light_k=light_k, additive=additive)
        self.b2   = TritBlock(32,    64,  light_k=light_k, additive=additive)
        self.b3   = TritBlock(64,    128, light_k=light_k, additive=additive)
        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            TernaryConv2d(128, 32, 1, quantize=False), nn.ReLU(),
            TernaryConv2d(32,   1, 1, quantize=False), nn.Sigmoid())
        self.mem  = nn.Sequential(TernaryLinear(128, 128, quantize=False), nn.Sigmoid())
        self.cls  = TernaryLinear(128, num_classes, quantize=False)

    def forward(self, x):
        preds = []
        o, p, i = self.b1(x);  preds.append((p, i)); o = self.pool(o)
        o, p, i = self.b2(o);  preds.append((p, i)); o = self.pool(o)
        o, p, i = self.b3(o);  preds.append((p, i)); o = self.pool(o)
        o = o * self.attn(o)
        feat = self.gap(o).squeeze(-1).squeeze(-1)
        feat = feat * self.mem(feat)
        return self.cls(feat), preds

class TernaryStdCNN(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.capture = False
        self.last_blocks = []
        self.blk1 = nn.Sequential(
            TernaryConv2d(in_ch, 96, 3, padding=1, quantize=False),
            nn.BatchNorm2d(96), nn.ReLU())
        self.blk2 = nn.Sequential(
            TernaryConv2d(96, 192, 3, padding=1, quantize=False),
            nn.BatchNorm2d(192), nn.ReLU())
        self.blk3 = nn.Sequential(
            TernaryConv2d(192, 128, 3, padding=1, quantize=False),
            nn.BatchNorm2d(128), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls = TernaryLinear(128, num_classes, quantize=False)

    def forward(self, x):
        outs = []
        o = self.pool(self.blk1(x)); outs.append(o)
        o = self.pool(self.blk2(o)); outs.append(o)
        o = self.pool(self.blk3(o)); outs.append(o)
        if self.capture:
            self.last_blocks = [t.detach() for t in outs]
        return self.cls(self.gap(o).flatten(1))

# ── Training (same recipe) ────────────────────────────────────────────────────

class PredLoss(nn.Module):
    def __init__(self, w=0.01):
        super().__init__()
        self.w = w
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, labels, preds):
        cls = self.ce(logits, labels)
        pred = sum(F.mse_loss(p, a.detach()) for p, a in preds)
        return cls + self.w * pred

def train(model, loader, label, is_trit=True, pred_weight=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = PredLoss(w=pred_weight) if is_trit else nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(EPOCHS):
        set_quant(model, epoch >= QUANT_WARMUP)
        model.train()
        total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out[0], labels, out[1]) if is_trit else loss_fn(out, labels)
            loss.backward()
            opt.step()
            total += loss.item()
        sch.step()
        phase = "TERNARY" if epoch >= QUANT_WARMUP else "warmup"
        print(f"  [{label}|{phase}] epoch {epoch+1}/{EPOCHS} "
              f"loss={total/len(loader):.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ── Measurement ───────────────────────────────────────────────────────────────

def zero_frac(a):
    t = 0.7 * a.abs().mean()
    return ((a.abs() <= t).float().mean()).item()

@torch.no_grad()
def measure(model, loader, angle, is_trit):
    """Per-batch zero-trit fraction of block outputs (mean over blocks),
    and for the trit model the gate-uncertainty fraction."""
    set_quant(model, True)
    model.eval()
    if is_trit:
        blocks = [model.b1, model.b2, model.b3]
        for b in blocks:
            b.capture = True
    else:
        model.capture = True
    zf_batches, gate_batches, acc_correct, acc_n = [], [], 0, 0
    for imgs, labels in loader:
        imgs = TF.rotate(imgs.to(device), angle,
                         interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        out = model(imgs)
        logits = out[0] if is_trit else out
        acc_correct += (logits.argmax(1) == labels.to(device)).sum().item()
        acc_n += len(labels)
        if is_trit:
            zf = np.mean([zero_frac(b.last_out) for b in blocks])
            gu = np.mean([((b.last_s0 - 0.5).abs() < GATE_BAND).float().mean().item()
                          for b in blocks])
            gate_batches.append(gu)
        else:
            zf = np.mean([zero_frac(t) for t in model.last_blocks])
        zf_batches.append(zf)
    if is_trit:
        for b in blocks:
            b.capture = False
            b.last_s0 = b.last_out = None
    else:
        model.capture = False
        model.last_blocks = []
    return (np.array(zf_batches), np.array(gate_batches) if is_trit else None,
            acc_correct / acc_n * 100)

def paired_t(diffs):
    diffs = np.asarray(diffs)
    return diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))

def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))

# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    mu, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), transforms.Normalize(mu, std)])
    te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])
    train_data = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=tr_tf)
    test_data = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=te_tf)
    tr_loader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=0)
    te_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

    configs = [
        ("TritFull", TritModel(10, 3, light_k=5, additive=False), True, 0.01),
        ("TernaryStdCNN", TernaryStdCNN(10, 3), False, 0.0),
    ]

    curves, batch_zf, batch_gate = {}, {}, {}
    for label, model, is_trit, pred_w in configs:
        model = model.to(device)
        print(f"=== Training {label} "
              f"({sum(p.numel() for p in model.parameters()):,} params) ===", flush=True)
        train(model, tr_loader, label, is_trit=is_trit, pred_weight=pred_w)

        curves[label] = {}
        batch_zf[label] = {}
        batch_gate[label] = {}
        for angle in CURVE_ANGLES:
            zf, gate, acc = measure(model, te_loader, angle, is_trit)
            curves[label][angle] = (zf.mean(), acc)
            batch_zf[label][angle] = zf
            if gate is not None:
                batch_gate[label][angle] = gate
            print(f"  {label} @ {angle:>3} deg: zero-frac {zf.mean():.4f}  "
                  f"acc {acc:.1f}%"
                  + (f"  gate-uncertain {gate.mean():.4f}" if gate is not None else ""),
                  flush=True)

    a0, a90 = CONTRAST
    print("\n===== RESULTS =====")
    print(f"{'Model':<16}{'zf@0':<10}{'zf@90':<10}{'delta':<12}{'paired t':<10}")
    deltas = {}
    for label in curves:
        d = batch_zf[label][a90] - batch_zf[label][a0]
        deltas[label] = d
        print(f"{label:<16}{batch_zf[label][a0].mean():<10.4f}"
              f"{batch_zf[label][a90].mean():<10.4f}{d.mean():<+12.4f}"
              f"{paired_t(d):<+10.2f}")

    print(f"\nP1 (TritFull zero-frac rises at 90 deg): "
          f"delta={deltas['TritFull'].mean():+.4f}, t={paired_t(deltas['TritFull']):+.2f}")
    print(f"P2 (TritFull delta > TernaryStdCNN delta): "
          f"{deltas['TritFull'].mean():+.4f} vs {deltas['TernaryStdCNN'].mean():+.4f}, "
          f"Welch t={welch_t(deltas['TritFull'], deltas['TernaryStdCNN']):+.2f}")
    dg = batch_gate['TritFull'][a90] - batch_gate['TritFull'][a0]
    print(f"P3 (gate-uncertainty rises at 90 deg): "
          f"{batch_gate['TritFull'][a0].mean():.4f} -> {batch_gate['TritFull'][a90].mean():.4f}, "
          f"delta={dg.mean():+.4f}, t={paired_t(dg):+.2f}")

    print("\nFull curves (angle: zero-frac, accuracy):")
    for label in curves:
        row = "  ".join(f"{a}:{curves[label][a][0]:.4f}/{curves[label][a][1]:.0f}%"
                        for a in CURVE_ANGLES)
        print(f"  {label:<16}{row}")

    print("\nPre-registered: P1 t>2, P2 Welch t>2, P3 t>2. Reported as run.")


if __name__ == "__main__":
    main()
