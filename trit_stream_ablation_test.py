"""
Stream Ablation Study — What inside PredictiveTritBlock causes rotation robustness?

The full triadic block:
  s0 = sigmoid(Conv1x1(x))          Observer gate   — point context
  s1 = tanh(Conv3x3(x))             Shadow features — local 3x3
  s2 = tanh(Conv5x5(x))             Light context   — wide 5x5
  out = s1*(1-s0) + s2*s0           multiplicative blend

Three hypotheses about what causes robustness:
  H1: Multi-scale receptive field — 5x5 captures spatial context that
      survives rotation better than 3x3. Test: replace 5x5 with 3x3.
  H2: Multiplicative gating — the blending logic itself, not the kernel
      sizes. Test: keep 1x1/3x3/5x5 but replace gate with simple addition.
  H3: Both together — removing either one hurts. Tests H1 and H2 independently.

Also tests:
  H4: Predictive loss — how much of robustness comes from the pred coding
      auxiliary loss (lambda=0.01) vs the triadic structure alone.

Ablation ladder (each removes one thing from TritFull):
  TritFull          — 1x1 Observer + 3x3 Shadow + 5x5 Light + mult gate + pred loss
  Trit-3x3Light     — replace 5x5 with 3x3 (removes multi-scale, keeps gate)  [H1]
  Trit-AddGate      — keep 1x1/3x3/5x5 but gate = s1+s2 (removes mult gate)   [H2]
  Trit-3x3+AddGate  — both: 3x3 + additive gate (= structured but flat)        [H1+H2]
  Trit-NoPredLoss   — full triadic, lambda=0 (removes pred coding)              [H4]
  TernaryStdCNN     — ternary weights, no triadic at all (baseline)

Usage:
  python trit_stream_ablation_test.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

EPOCHS       = 20
QUANT_WARMUP = 4
ANGLES       = list(range(0, 360, 15))   # full curve at 15° steps

# ── Ternary core ──────────────────────────────────────────────────────────────

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

# ── Triadic blocks with configurable kernel sizes and gate type ───────────────

class TritBlock(nn.Module):
    """
    Configurable triadic block.
    light_k: kernel size for the Light (context) stream — 5 (full) or 3 (H1 ablation)
    additive: if True, gate = s1+s2 instead of s1*(1-s0)+s2*s0 (H2 ablation)
    """
    def __init__(self, in_ch, out_ch, light_k=5, additive=False):
        super().__init__()
        self.additive = additive
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
            out = s1 + s2            # H2: remove multiplicative gate
        else:
            out = s1 * (1 - s0) + s2 * s0   # full triadic blend
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

# ── Ternary standard CNN (no triadic — floor baseline) ───────────────────────

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
    def __init__(self, w=0.01):
        super().__init__()
        self.w = w
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, labels, preds):
        cls  = self.ce(logits, labels)
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
            if is_trit:
                logits, preds = out
                loss = loss_fn(logits, labels, preds)
            else:
                loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            total += loss.item()
        sch.step()
        if (epoch + 1) % 10 == 0:
            phase = "TERNARY" if epoch >= QUANT_WARMUP else "warmup"
            print(f"  [{label}|{phase}] epoch {epoch+1}/{EPOCHS} "
                  f"loss={total/len(loader):.4f}  ({time.time()-t0:.0f}s)")

# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, n_samples, is_trit=True):
    set_quant(model, True)
    model.eval()
    results = {}
    for angle in ANGLES:
        correct = 0
        for imgs, labels in loader:
            imgs = TF.rotate(imgs.to(device), angle,
                             interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            out = model(imgs)
            if is_trit:
                out = out[0]
            correct += (out.argmax(1) == labels.to(device)).sum().item()
        results[angle] = correct / n_samples * 100
    return results

@torch.no_grad()
def measure_activation_sparsity(model, loader, angles, is_trit=True):
    """
    Measure zero-activation fraction per angle — tests the uncertainty hypothesis:
    if the zero trit encodes genuine epistemic uncertainty, activations should
    become sparser when the input is rotated (more ambiguous between Shadow/Light).
    Returns {angle: sparsity_fraction} for all ternary layers combined.
    """
    set_quant(model, True)
    model.eval()

    sparsity_by_angle = {}
    hooks = []
    activation_zeros = [0]
    activation_total = [0]

    def make_hook(name):
        def hook(module, input, output):
            # Count zeros in the quantized output
            with torch.no_grad():
                t = 0.7 * output.abs().mean()
                q = torch.where(output > t,  torch.ones_like(output),
                    torch.where(output < -t, -torch.ones_like(output),
                                torch.zeros_like(output)))
                activation_zeros[0] += (q == 0).sum().item()
                activation_total[0] += q.numel()
        return hook

    # Hook onto conv layers in the triadic blocks only
    for name, module in model.named_modules():
        if isinstance(module, TernaryConv2d) and 's0' in name or \
           isinstance(module, TernaryConv2d) and 's1' in name or \
           isinstance(module, TernaryConv2d) and 's2' in name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    if not hooks:
        # Fallback: hook all TernaryConv2d
        for name, module in model.named_modules():
            if isinstance(module, TernaryConv2d):
                hooks.append(module.register_forward_hook(make_hook(name)))

    for angle in angles:
        activation_zeros[0] = 0
        activation_total[0] = 0
        batch_count = 0
        for imgs, labels in loader:
            imgs = TF.rotate(imgs.to(device), angle,
                             interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            out = model(imgs)
            batch_count += 1
            if batch_count >= 5:   # sample 5 batches per angle for speed
                break
        sp = activation_zeros[0] / activation_total[0] if activation_total[0] > 0 else 0.0
        sparsity_by_angle[angle] = sp

    for h in hooks:
        h.remove()

    return sparsity_by_angle

def worst_drop(acc):
    return acc[0] - min(acc.values())

def mean_drop(acc):
    other = [v for k, v in acc.items() if k != 0]
    return acc[0] - sum(other) / len(other)

# ── Dataset ───────────────────────────────────────────────────────────────────

mu, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
tr_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mu, std)])
te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])

train_data = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=tr_tf)
test_data  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=te_tf)
tr_loader  = DataLoader(train_data, batch_size=512, shuffle=True,  num_workers=0, pin_memory=True)
te_loader  = DataLoader(test_data,  batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
n_test = len(test_data)

# ── Build and train all variants ──────────────────────────────────────────────

configs = [
    # (label,            model,                                             is_trit, pred_w)
    ("TritFull",         TritModel(10, 3, light_k=5, additive=False),      True,    0.01),
    ("Trit-3x3Light",    TritModel(10, 3, light_k=3, additive=False),      True,    0.01),
    ("Trit-AddGate",     TritModel(10, 3, light_k=5, additive=True),       True,    0.01),
    ("Trit-3x3+AddGate", TritModel(10, 3, light_k=3, additive=True),       True,    0.01),
    ("Trit-NoPredLoss",  TritModel(10, 3, light_k=5, additive=False),      True,    0.0),
    ("TernaryStdCNN",    TernaryStdCNN(10, 3),                             False,   0.0),
]

results = {}
sparsity_results = {}
SPARSITY_ANGLES = [0, 45, 90, 135, 180]

for label, model, is_trit, pred_w in configs:
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*65}")
    print(f"  Training: {label}  ({n_params:,} params)")
    print(f"{'='*65}")
    train(model, tr_loader, label, is_trit=is_trit, pred_weight=pred_w)
    print(f"  Evaluating at {len(ANGLES)} angles...")
    acc = evaluate(model, te_loader, n_test, is_trit=is_trit)
    results[label] = acc
    print(f"  0°={acc[0]:.1f}%  worst_drop={worst_drop(acc):.1f}pp  mean_drop={mean_drop(acc):.1f}pp")
    if is_trit:
        print(f"  Measuring activation sparsity vs rotation...")
        sp = measure_activation_sparsity(model, te_loader, SPARSITY_ANGLES, is_trit=is_trit)
        sparsity_results[label] = sp
        sp_clean = sp[0] * 100
        sp_90    = sp[90] * 100
        print(f"  Sparsity: 0°={sp_clean:.1f}%  90°={sp_90:.1f}%  delta={sp_90-sp_clean:+.1f}pp")

# ── Results ───────────────────────────────────────────────────────────────────

print("\n" + "="*75)
print("  STREAM ABLATION RESULTS — CIFAR-10, full 0-360° rotation curve")
print("="*75)
print(f"  {'Model':<22}  {'0°':>6}  {'Worst drop':>11}  {'Mean drop':>10}  {'Mechanism tested'}")
print(f"  {'-'*73}")

baselines = {
    "TritFull":         "full triadic (1x1 + 3x3 + 5x5 + mult gate + pred)",
    "Trit-3x3Light":    "H1: replace 5x5 Light with 3x3 (remove multi-scale)",
    "Trit-AddGate":     "H2: additive gate s1+s2 (remove multiplicative blend)",
    "Trit-3x3+AddGate": "H1+H2: 3x3 + additive (flat triadic structure)",
    "Trit-NoPredLoss":  "H4: pred loss lambda=0 (remove predictive coding)",
    "TernaryStdCNN":    "floor: ternary weights, no triadic at all",
}

full_worst = worst_drop(results["TritFull"])
full_mean  = mean_drop(results["TritFull"])

for label, desc in baselines.items():
    acc = results[label]
    wd = worst_drop(acc)
    md = mean_drop(acc)
    print(f"  {label:<22}  {acc[0]:>5.1f}%  {wd:>+10.1f}pp  {md:>+9.1f}pp  {desc}")

print()
print("  Mechanism attribution (positive = more robustness degraded vs TritFull):")
for label in ["Trit-3x3Light", "Trit-AddGate", "Trit-3x3+AddGate", "Trit-NoPredLoss"]:
    wd_delta = worst_drop(results[label]) - full_worst
    md_delta = mean_drop(results[label]) - full_mean
    print(f"    Remove {label.replace('Trit-',''):<18}: "
          f"worst_drop {wd_delta:+.1f}pp  mean_drop {md_delta:+.1f}pp")

# Print full rotation curve for TritFull vs TernaryStdCNN
print()
print("  Full rotation curve (TritFull vs TernaryStdCNN):")
print(f"  {'Angle':>6}  {'TritFull':>10}  {'TernaryStd':>12}  {'Gap':>8}")
for angle in ANGLES:
    tf_acc  = results["TritFull"][angle]
    std_acc = results["TernaryStdCNN"][angle]
    print(f"  {angle:>5}°  {tf_acc:>9.1f}%  {std_acc:>11.1f}%  {tf_acc-std_acc:>+7.1f}pp")

print("="*75)

# ── Zero-trit uncertainty test ────────────────────────────────────────────────

if sparsity_results:
    print("\n" + "="*75)
    print("  ZERO-TRIT UNCERTAINTY HYPOTHESIS")
    print("  If zero=uncertainty, sparsity should INCREASE under rotation")
    print("="*75)
    print(f"  {'Model':<22}  {'0deg':>8}  {'45deg':>8}  {'90deg':>8}  {'135deg':>8}  {'180deg':>8}  {'Delta(0->90)':>13}")
    print(f"  {'-'*73}")
    for label, sp in sparsity_results.items():
        vals = [sp.get(a, 0)*100 for a in SPARSITY_ANGLES]
        delta = vals[2] - vals[0]
        marker = "INCREASES (uncertainty)" if delta > 1.0 else ("stable" if abs(delta) <= 1.0 else "DECREASES")
        print(f"  {label:<22}  {vals[0]:>7.1f}%  {vals[1]:>7.1f}%  {vals[2]:>7.1f}%  {vals[3]:>7.1f}%  {vals[4]:>7.1f}%  {delta:>+12.1f}pp  {marker}")
    print("="*75)
