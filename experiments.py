import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
import numpy as np
import json, time, os

os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ══════════════════════════════════════════════════════════════════════════════
# TERNARY CORE
# ══════════════════════════════════════════════════════════════════════════════

class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        t = 0.7 * x.abs().mean()
        ctx.save_for_backward(x, t.unsqueeze(0))
        return torch.where(x >  t,  torch.ones_like(x),
               torch.where(x < -t, -torch.ones_like(x),
               torch.zeros_like(x)))
    @staticmethod
    def backward(ctx, grad):
        x, _ = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).float()

tq = TernaryQuantize.apply

class TernaryConv2d(nn.Conv2d):
    def __init__(self, *args, quantize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_quantize = quantize
    def forward(self, x):
        w = tq(self.weight) if self.do_quantize else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding)

class TernaryLinear(nn.Linear):
    """
    Standard linear layer with ternary weights {-1, 0, +1}.
    Forward: W_trit = sign(W) if |W| > 0.7*E[|W|] else 0
    Backward: STE — gradient passes through where |W| <= 1
    No triadic logic — pure ternary MAC (add/subtract/skip).
    """
    def __init__(self, *args, quantize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_quantize = quantize
    def forward(self, x):
        w = tq(self.weight) if self.do_quantize else self.weight
        return F.linear(x, w, self.bias)

def set_quant(model, active):
    for m in model.modules():
        if isinstance(m, (TernaryConv2d, TernaryLinear)):
            m.do_quantize = active

def trit_dist(model):
    total = neg = zero = pos = 0
    for m in model.modules():
        if isinstance(m, (TernaryConv2d, TernaryLinear)):
            t     = 0.7 * m.weight.data.abs().mean()
            q     = torch.where(m.weight.data >  t,  torch.ones_like(m.weight.data),
                    torch.where(m.weight.data < -t, -torch.ones_like(m.weight.data),
                    torch.zeros_like(m.weight.data)))
            total += q.numel()
            neg   += (q == -1).sum().item()
            zero  += (q ==  0).sum().item()
            pos   += (q ==  1).sum().item()
    if total == 0: return 0, 0, 0
    return neg/total*100, zero/total*100, pos/total*100

# ══════════════════════════════════════════════════════════════════════════════
# MODEL B: StandardCNN (float32) — ablation: architecture effect only
# ══════════════════════════════════════════════════════════════════════════════

class StandardCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, x): return self.conv(x)

class StandardCNN(nn.Module):
    """Same depth/width as TritCognition, float32 weights, no triadic structure."""
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.b1   = StandardCNNBlock(in_ch, 96)
        self.b2   = StandardCNNBlock(96, 192)
        self.b3   = StandardCNNBlock(192, 128)
        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.cls  = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.pool(self.b3(x))
        return self.cls(self.gap(x).squeeze(-1).squeeze(-1))

# ══════════════════════════════════════════════════════════════════════════════
# MODEL C: TernaryStandardCNN — ablation: ternary weights without triadic
# Isolates whether ternary quantization alone causes robustness
# ══════════════════════════════════════════════════════════════════════════════

class TernaryStandardCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            TernaryConv2d(in_ch, out_ch, 3, padding=1, quantize=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, x): return self.conv(x)

class TernaryStandardCNN(nn.Module):
    """
    Same as StandardCNN but ternary weights {-1,0,+1}.
    If this matches TritCognition robustness → quantization is the cause.
    If TritCognition is better → triadic structure is the cause.
    """
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.b1   = TernaryStandardCNNBlock(in_ch, 96)
        self.b2   = TernaryStandardCNNBlock(96, 192)
        self.b3   = TernaryStandardCNNBlock(192, 128)
        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.cls  = TernaryLinear(128, num_classes, quantize=False)
    def forward(self, x):
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.pool(self.b3(x))
        return self.cls(self.gap(x).squeeze(-1).squeeze(-1))

# ══════════════════════════════════════════════════════════════════════════════
# MODEL D: TritCognition — full 012 stack + ablation variants
# ══════════════════════════════════════════════════════════════════════════════

class PredictiveTritBlock(nn.Module):
    """
    Three parallel streams, each with BatchNorm:
      stream_0: Conv1x1 + BN → sigmoid  [0,1]   (Observer gate)
      stream_1: Conv3x3 + BN → tanh    [-1,1]   (Shadow features)
      stream_2: Conv5x5 + BN → tanh    [-1,1]   (Light context)
    Triadic interaction: out = s1*(1-s0) + s2*s0
    Hardware: consensus(trit(s0), trit(s1), trit(s2)) = sign(s0+s1+s2)
    pred head: Conv1x1 predicts this block's input x (predictive coding)
    """
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

class PredLoss(nn.Module):
    def __init__(self, w=0.01):
        super().__init__()
        self.w  = w
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, labels, preds):
        cls  = self.ce(logits, labels)
        pred = sum(F.mse_loss(p, a.detach()) for p, a in preds)
        return cls + self.w * pred, cls.item(), pred.item()

class TritCognition(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, use_attention=True, use_memory=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_memory    = use_memory

        self.b1   = PredictiveTritBlock(in_ch, 32)
        self.b2   = PredictiveTritBlock(32, 64)
        self.b3   = PredictiveTritBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d(1)

        # Spatial attention: 128→32→1 sigmoid map (binding)
        self.attn = nn.Sequential(
            TernaryConv2d(128, 32, 1, quantize=False), nn.ReLU(),
            TernaryConv2d(32,   1, 1, quantize=False), nn.Sigmoid()
        ) if use_attention else None

        # MemoryGate: gate = σ(W·feat);  out = feat ⊙ gate
        # Learned feature selection — gates which of 128 dims reach classifier
        self.mem  = nn.Sequential(
            TernaryLinear(128, 128, quantize=False), nn.Sigmoid()
        ) if use_memory else None

        self.cls  = TernaryLinear(128, num_classes, quantize=False)

    def forward(self, x):
        preds = []
        o, p, i = self.b1(x);  preds.append((p,i)); o = self.pool(o)
        o, p, i = self.b2(o);  preds.append((p,i)); o = self.pool(o)
        o, p, i = self.b3(o);  preds.append((p,i)); o = self.pool(o)
        if self.use_attention and self.attn is not None:
            o = o * self.attn(o)
        feat = self.gap(o).squeeze(-1).squeeze(-1)
        if self.use_memory and self.mem is not None:
            feat = feat * self.mem(feat)
        return self.cls(feat), preds

# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def get_cifar10():
    mu, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mu, std)
    ])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])
    train = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=tr)
    test  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=te)
    return train, test, 3, 10

def get_stl10():
    mu, std = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)
    tr = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mu, std)
    ])
    te = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mu, std)
    ])
    # Full 5000 labeled training samples
    train = torchvision.datasets.STL10('./data', split='train', download=True, transform=tr)
    test  = torchvision.datasets.STL10('./data', split='test',  download=True, transform=te)
    return train, test, 3, 10

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

EPOCHS       = 50
QUANT_WARMUP = 8

def train_resnet(model, loader, label):
    """SGD+momentum — standard CIFAR recipe, targets 93-95%"""
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce  = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = ce(model(imgs), labels)
            loss.backward()
            opt.step()
            total += loss.item()
        sch.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] epoch {epoch+1:>2}/{EPOCHS}  loss={total/len(loader):.4f}")

def train_standard(model, loader, label):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce  = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = ce(model(imgs), labels)
            loss.backward()
            opt.step()
            total += loss.item()
        sch.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] epoch {epoch+1:>2}/{EPOCHS}  loss={total/len(loader):.4f}")

def train_ternary_std(model, loader, label):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce  = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        set_quant(model, epoch >= QUANT_WARMUP)
        model.train()
        total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = ce(model(imgs), labels)
            loss.backward()
            opt.step()
            total += loss.item()
        sch.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            phase = "TERNARY" if epoch >= QUANT_WARMUP else "warmup"
            print(f"  [{label}|{phase}] epoch {epoch+1:>2}/{EPOCHS}  loss={total/len(loader):.4f}")

def train_trit(model, loader, label, pred_weight=0.01):
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = PredLoss(w=pred_weight)
    for epoch in range(EPOCHS):
        set_quant(model, epoch >= QUANT_WARMUP)
        model.train()
        total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits, preds = model(imgs)
            loss, _, _    = loss_fn(logits, labels, preds)
            loss.backward()
            opt.step()
            total += loss.item()
        sch.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            phase = "TERNARY" if epoch >= QUANT_WARMUP else "warmup"
            print(f"  [{label}|{phase}] epoch {epoch+1:>2}/{EPOCHS}  loss={total/len(loader):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# Rotation: bilinear interpolation, fill=0 (black padding for corners)
# Same interpolation method applied identically to all models
# ══════════════════════════════════════════════════════════════════════════════

ANGLES = [0, 45, 90, 135, 180, 270]

def evaluate(model, loader, n_samples, use_pred=False, quant=False):
    if quant: set_quant(model, True)
    model.eval()
    results = {}
    with torch.no_grad():
        for angle in ANGLES:
            correct = 0
            for imgs, labels in loader:
                imgs  = TF.rotate(imgs.to(device), angle,
                                  interpolation=TF.InterpolationMode.BILINEAR,
                                  fill=0)
                out   = model(imgs)
                if use_pred: out = out[0]
                correct += (out.argmax(1) == labels.to(device)).sum().item()
            results[angle] = correct / n_samples * 100
    return results

def stability(acc):     return acc[0] - min(acc.values())
def norm_stability(acc): return (acc[0] - min(acc.values())) / acc[0] * 100

# ══════════════════════════════════════════════════════════════════════════════
# RUN EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

all_results = {}

for ds_name, get_ds in [("CIFAR-10", get_cifar10), ("STL-10", get_stl10)]:
    print(f"\n{'═'*65}\n  DATASET: {ds_name}\n{'═'*65}")

    train_data, test_data, in_ch, n_cls = get_ds()
    tr_loader = DataLoader(train_data, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
    te_loader = DataLoader(test_data,  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    n_test    = len(test_data)

    resnet      = torchvision.models.resnet18(weights=None); resnet.fc = nn.Linear(512, n_cls); resnet = resnet.to(device)
    std_cnn     = StandardCNN(n_cls, in_ch).to(device)
    tern_std    = TernaryStandardCNN(n_cls, in_ch).to(device)
    trit_full   = TritCognition(n_cls, in_ch, use_attention=True,  use_memory=True).to(device)
    trit_nopred = TritCognition(n_cls, in_ch, use_attention=True,  use_memory=True).to(device)
    trit_noattn = TritCognition(n_cls, in_ch, use_attention=False, use_memory=True).to(device)
    trit_nomem  = TritCognition(n_cls, in_ch, use_attention=True,  use_memory=False).to(device)

    print(f"\n  Parameters:")
    print(f"    ResNet18           : {sum(p.numel() for p in resnet.parameters()):>10,}")
    print(f"    StandardCNN        : {sum(p.numel() for p in std_cnn.parameters()):>10,}")
    print(f"    TernaryStandardCNN : {sum(p.numel() for p in tern_std.parameters()):>10,}")
    print(f"    TritCognition      : {sum(p.numel() for p in trit_full.parameters()):>10,}")

    print(f"\n  Training ResNet18 (SGD+momentum)...")
    train_resnet(resnet, tr_loader, "ResNet18")

    print(f"\n  Training StandardCNN (float32)...")
    train_standard(std_cnn, tr_loader, "StandardCNN")

    print(f"\n  Training TernaryStandardCNN (ternary, no triadic)...")
    train_ternary_std(tern_std, tr_loader, "TernaryStdCNN")

    print(f"\n  Training TritCognition full (012)...")
    train_trit(trit_full, tr_loader, "TritFull", pred_weight=0.01)

    print(f"\n  Training TritCognition no pred loss (λ=0 ablation)...")
    train_trit(trit_nopred, tr_loader, "TritNoPred", pred_weight=0.0)

    print(f"\n  Training TritCognition no attention (ablation)...")
    train_trit(trit_noattn, tr_loader, "TritNoAttn", pred_weight=0.01)

    print(f"\n  Training TritCognition no memory gate (ablation)...")
    train_trit(trit_nomem, tr_loader, "TritNoMem", pred_weight=0.01)

    # Evaluate
    print(f"\n  Evaluating all models...")
    models_eval = {
        "ResNet18"          : (resnet,      False, False),
        "StandardCNN"       : (std_cnn,     False, False),
        "TernaryStdCNN"     : (tern_std,    False, True),
        "TritFull"          : (trit_full,   True,  True),
        "Trit-NoPredLoss"   : (trit_nopred, True,  True),
        "Trit-NoAttention"  : (trit_noattn, True,  True),
        "Trit-NoMemoryGate" : (trit_nomem,  True,  True),
    }

    res = {}
    for name, (model, use_pred, quant) in models_eval.items():
        acc = evaluate(model, te_loader, n_test, use_pred=use_pred, quant=quant)
        res[name] = acc
        print(f"    {name:<22}: 0°={acc[0]:.2f}%  stab=-{stability(acc):.2f}%  norm=-{norm_stability(acc):.2f}%")

    all_results[ds_name] = res

    neg, zero, pos = trit_dist(trit_full)
    print(f"\n  TritFull weight dist: -1:{neg:.1f}%  0:{zero:.1f}%  +1:{pos:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("  FULL RESULTS")
print("═"*80)

for ds_name, res in all_results.items():
    print(f"\n── {ds_name}")
    print(f"{'Model':<22} | {'0°':>7} | {'45°':>7} | {'90°':>7} | {'180°':>7} | {'Stab↓':>7} | {'NormStab↓':>10}")
    print("-" * 80)
    for name, acc in res.items():
        print(f"{name:<22} | {acc[0]:>6.2f}% | {acc[45]:>6.2f}% | {acc[90]:>6.2f}% | "
              f"{acc[180]:>6.2f}% | {stability(acc):>6.2f}% | {norm_stability(acc):>9.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# ABLATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("  ABLATION SUMMARY")
print("═"*80)

for ds_name, res in all_results.items():
    print(f"\n── {ds_name}")
    full_ns = norm_stability(res["TritFull"])

    print(f"  Component ablations (normalized stability — lower = more robust):")
    for name in ["TritFull", "Trit-NoPredLoss", "Trit-NoAttention", "Trit-NoMemoryGate"]:
        ns   = norm_stability(res[name])
        diff = ns - full_ns
        sign = "+" if diff >= 0 else ""
        tag  = "(full)" if name == "TritFull" else f"({sign}{diff:.2f}pp worse without)"
        print(f"    {name:<22}: {ns:.2f}%  {tag}")

    print(f"\n  Architecture vs quantization isolation:")
    for name in ["StandardCNN", "TernaryStdCNN", "TritFull"]:
        acc = res[name]
        print(f"    {name:<22}: 0°={acc[0]:.2f}%  norm_stab={norm_stability(acc):.2f}%")
    tern_ns = norm_stability(res["TernaryStdCNN"])
    std_ns  = norm_stability(res["StandardCNN"])
    if abs(tern_ns - std_ns) < 3:
        print(f"  → TernaryStdCNN ≈ StandardCNN: ternary weights alone do NOT cause robustness")
    if full_ns < tern_ns:
        print(f"  → TritFull > TernaryStdCNN: triadic structure IS the cause")

# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE + ENERGY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("  HARDWARE EFFICIENCY")
print("═"*80)

trit_p  = sum(p.numel() for p in trit_full.parameters())
res18_p = 11_181_642
E_BIN   = 4.6   # pJ binary MAC (Whatmough et al. 2017, 28nm)
E_TRIT  = 1.1   # pJ ternary op  (Andri et al. 2018 ENVISION, 28nm)
N       = 1_000_000
zero_avg = 0.465
active   = 1 - zero_avg
bin_e    = N * E_BIN
trit_e   = N * active * E_TRIT
saving   = (1 - trit_e / bin_e) * 100

print(f"\n  Energy (Andri et al. 2018 ENVISION, 28nm CMOS):")
print(f"    Binary MAC  : {E_BIN} pJ  (mul 3.7pJ + add 0.9pJ)")
print(f"    Ternary op  : {E_TRIT} pJ  (add/subtract only, no multiply)")
print(f"    Active frac : {active*100:.1f}% (avg CIFAR-10/STL-10 non-zero weights)")
print(f"    Saving      : {saving:.1f}% per {N:,} MACs")
print(f"\n  Model size:")
print(f"    ResNet18 float32   : {res18_p*4/1e6:.1f} MB")
print(f"    TritCognition f32  : {trit_p*4/1e6:.2f} MB")
print(f"    TritCognition trit : {trit_p*1.585/8/1e6:.3f} MB  (log2(3) bits/weight)")
print(f"    Compression        : {trit_p*32/(trit_p*1.585):.1f}x vs float32")
print(f"    Param reduction    : {res18_p/trit_p:.1f}x vs ResNet18")

with open("results/012_v2_benchmark.json", "w") as f:
    json.dump({ds: {m: {str(k): v for k,v in acc.items()} for m, acc in res.items()}
               for ds, res in all_results.items()}, f, indent=2)
print("\nSaved: results/012_v2_benchmark.json")
