"""
Option 2 — Perturbation Generalization Test

Does the triadic structure give general geometric robustness, or only rotational?

Tests 5 perturbation types on pre-trained models (loaded from saved checkpoints
written by trit_stream_ablation_test.py). If you haven't run that yet, this
script also trains the two key models (TritFull, TernaryStdCNN) fresh.

Perturbations tested:
  1. Rotation        0-360° at 15° steps      (same as stream ablation)
  2. Translation     0-14px horizontal shift
  3. Scaling         0.5x-1.5x zoom
  4. Gaussian noise  sigma 0.0-0.5
  5. Brightness      factor 0.3-2.0
  6. Affine          combined rotation+translation+scale (real-world distortion)

If TritFull beats TernaryStdCNN on ALL perturbation types:
  -> Triadic structure gives general robustness, not rotation-specific
If TritFull only beats on rotation:
  -> The block may have incidentally learned rotation invariance from CIFAR-10
     training distribution, not an architectural property

Usage:
  python trit_perturbation_test.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

EPOCHS       = 20
QUANT_WARMUP = 4

# ── Ternary core (same as stream ablation) ─────────────────────────────────────

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
        super().__init__(*a, **k); self.do_quantize = quantize
    def forward(self, x):
        return F.conv2d(x, tq(self.weight) if self.do_quantize else self.weight,
                        self.bias, self.stride, self.padding)

class TernaryLinear(nn.Linear):
    def __init__(self, *a, quantize=False, **k):
        super().__init__(*a, **k); self.do_quantize = quantize
    def forward(self, x):
        return F.linear(x, tq(self.weight) if self.do_quantize else self.weight, self.bias)

def set_quant(model, active):
    for m in model.modules():
        if isinstance(m, (TernaryConv2d, TernaryLinear)):
            m.do_quantize = active

# ── Models ────────────────────────────────────────────────────────────────────

class TritBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.s0   = nn.Sequential(TernaryConv2d(in_ch, out_ch, 1,            quantize=False), nn.BatchNorm2d(out_ch))
        self.s1   = nn.Sequential(TernaryConv2d(in_ch, out_ch, 3, padding=1, quantize=False), nn.BatchNorm2d(out_ch))
        self.s2   = nn.Sequential(TernaryConv2d(in_ch, out_ch, 5, padding=2, quantize=False), nn.BatchNorm2d(out_ch))
        self.pred = TernaryConv2d(out_ch, in_ch, 1, quantize=False)
    def forward(self, x):
        s0  = torch.sigmoid(self.s0(x))
        s1  = torch.tanh(self.s1(x))
        s2  = torch.tanh(self.s2(x))
        out = s1 * (1 - s0) + s2 * s0
        return out, self.pred(out), x

class TritFull(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.b1   = TritBlock(in_ch, 32)
        self.b2   = TritBlock(32,    64)
        self.b3   = TritBlock(64,    128)
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
        def blk(i, o): return nn.Sequential(
            TernaryConv2d(i, o, 3, padding=1, quantize=False),
            nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(
            blk(in_ch, 96), nn.MaxPool2d(2),
            blk(96, 192),   nn.MaxPool2d(2),
            blk(192, 128),  nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.cls = TernaryLinear(128, num_classes, quantize=False)
    def forward(self, x):
        return self.cls(self.net(x))

# ── Training ──────────────────────────────────────────────────────────────────

class PredLoss(nn.Module):
    def __init__(self):
        super().__init__(); self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, labels, preds):
        return self.ce(logits, labels) + 0.01 * sum(F.mse_loss(p, a.detach()) for p, a in preds)

def train_model(model, loader, label, is_trit):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = PredLoss() if is_trit else nn.CrossEntropyLoss()
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
            loss.backward(); opt.step()
            total += loss.item()
        sch.step()
        if (epoch + 1) % 10 == 0:
            phase = "TRN" if epoch >= QUANT_WARMUP else "WRM"
            print(f"  [{label}|{phase}] e{epoch+1} loss={total/len(loader):.4f} ({time.time()-t0:.0f}s)")

# ── Perturbation functions ────────────────────────────────────────────────────

def perturb_rotation(imgs, strength):
    angle = strength
    return TF.rotate(imgs, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)

def perturb_translation(imgs, strength):
    px = int(strength)
    return TF.affine(imgs, angle=0, translate=[px, 0], scale=1.0, shear=0,
                     interpolation=TF.InterpolationMode.BILINEAR, fill=0)

def perturb_scale(imgs, strength):
    scale = strength
    h, w = imgs.shape[-2], imgs.shape[-1]
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    imgs_r = TF.resize(imgs, [new_h, new_w], antialias=True)
    # center crop or pad back to original size
    if scale >= 1.0:
        top  = (new_h - h) // 2
        left = (new_w - w) // 2
        return imgs_r[:, :, top:top+h, left:left+w]
    else:
        out = torch.zeros_like(imgs)
        top  = (h - new_h) // 2
        left = (w - new_w) // 2
        out[:, :, top:top+new_h, left:left+new_w] = imgs_r
        return out

def perturb_noise(imgs, strength):
    noise = torch.randn_like(imgs) * strength
    return (imgs + noise).clamp(-3.0, 3.0)

def perturb_brightness(imgs, strength):
    return TF.adjust_brightness(imgs, strength)

def perturb_affine(imgs, strength):
    angle = strength * 30
    tx    = int(strength * 4)
    scale = 1.0 - strength * 0.15
    return TF.affine(imgs, angle=angle, translate=[tx, 0], scale=scale, shear=0,
                     interpolation=TF.InterpolationMode.BILINEAR, fill=0)

# Perturbation sweep definitions: (name, fn, strengths, x_label)
PERTURBATIONS = [
    ("Rotation",    perturb_rotation,    list(range(0, 361, 15)),         "angle (°)"),
    ("Translation", perturb_translation, list(range(0, 15, 2)),           "shift (px)"),
    ("Scale",       perturb_scale,       [round(0.5 + i*0.1, 1) for i in range(11)], "scale"),
    ("Noise",       perturb_noise,       [round(i*0.05, 2) for i in range(11)],       "sigma"),
    ("Brightness",  perturb_brightness,  [round(0.3 + i*0.17, 2) for i in range(11)],"factor"),
    ("Affine",      perturb_affine,      [round(i*0.1, 1) for i in range(11)],        "strength"),
]

# ── Evaluate across perturbation sweep ───────────────────────────────────────

@torch.no_grad()
def eval_perturbation(model, loader, n_samples, is_trit, perturb_fn, strengths):
    set_quant(model, True)
    model.eval()
    results = {}
    for s in strengths:
        correct = 0
        for imgs, labels in loader:
            imgs_p = perturb_fn(imgs.to(device), s)
            out = model(imgs_p)
            if is_trit: out = out[0]
            correct += (out.argmax(1) == labels.to(device)).sum().item()
        results[s] = correct / n_samples * 100
    return results

def robustness_auc(results, baseline_key):
    """Area under the perturbation curve, normalized by the clean baseline."""
    keys  = sorted(results.keys())
    vals  = [results[k] for k in keys]
    # trapezoidal integration over index (not x-value, since x-scale varies)
    area  = np.trapezoid(vals, dx=1)
    # max possible area if accuracy stayed flat at baseline
    base  = results[baseline_key]
    max_a = base * (len(keys) - 1)
    return area / max_a * 100 if max_a > 0 else 0.0

# ── Dataset ───────────────────────────────────────────────────────────────────

mu, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
tr_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(), transforms.Normalize(mu, std)])
te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])

train_data = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=tr_tf)
test_data  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=te_tf)
tr_loader  = DataLoader(train_data, batch_size=512, shuffle=True,  num_workers=0, pin_memory=True)
te_loader  = DataLoader(test_data,  batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
n_test = len(test_data)

# ── Train both models ─────────────────────────────────────────────────────────

models_to_eval = {}
for label, model_cls, is_trit in [
    ("TritFull",     TritFull,       True),
    ("TernaryStdCNN", TernaryStdCNN, False),
]:
    print(f"\n{'='*60}")
    print(f"  Training {label}...")
    print(f"{'='*60}")
    m = model_cls(10, 3).to(device)
    train_model(m, tr_loader, label, is_trit)
    models_to_eval[label] = (m, is_trit)

# ── Run all perturbations ─────────────────────────────────────────────────────

print("\n" + "="*70)
print("  Running perturbation sweeps...")
print("="*70)

all_results = {label: {} for label in models_to_eval}

for pert_name, pert_fn, strengths, x_label in PERTURBATIONS:
    print(f"\n  Perturbation: {pert_name} ({x_label})")
    for label, (model, is_trit) in models_to_eval.items():
        res = eval_perturbation(model, te_loader, n_test, is_trit, pert_fn, strengths)
        all_results[label][pert_name] = res
        baseline = strengths[0]
        auc = robustness_auc(res, baseline)
        worst = res[baseline] - min(res.values())
        print(f"    {label:<18}: clean={res[baseline]:.1f}%  worst_drop={worst:.1f}pp  AUC={auc:.1f}%")

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "="*75)
print("  PERTURBATION GENERALIZATION — AUC (higher = more robust)")
print("  AUC = area under accuracy-vs-perturbation curve, normalized to clean baseline")
print("="*75)
print(f"  {'Perturbation':<16}  {'TritFull AUC':>14}  {'TernaryStd AUC':>16}  {'Trit advantage':>15}")
print(f"  {'-'*65}")

wins = 0
for pert_name, pert_fn, strengths, x_label in PERTURBATIONS:
    baseline = strengths[0]
    trit_auc  = robustness_auc(all_results["TritFull"][pert_name], baseline)
    std_auc   = robustness_auc(all_results["TernaryStdCNN"][pert_name], baseline)
    delta     = trit_auc - std_auc
    marker    = "<-- TRIT WINS" if delta > 2 else ("<-- STD WINS" if delta < -2 else "~tied")
    if delta > 2: wins += 1
    print(f"  {pert_name:<16}  {trit_auc:>13.1f}%  {std_auc:>15.1f}%  {delta:>+14.1f}pp  {marker}")

print()
if wins == len(PERTURBATIONS):
    print("  Verdict: TritFull wins on ALL perturbation types.")
    print("  -> Triadic structure gives GENERAL robustness, not rotation-specific.")
elif wins > len(PERTURBATIONS) // 2:
    print(f"  Verdict: TritFull wins on {wins}/{len(PERTURBATIONS)} perturbation types.")
    print("  -> Partial generalization — triadic structure helps broadly but not universally.")
else:
    print(f"  Verdict: TritFull wins on only {wins}/{len(PERTURBATIONS)} perturbation types.")
    print("  -> Robustness is rotation-specific, not a general architectural property.")

print("="*75)
