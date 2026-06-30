"""
012 Tunable Symmetry Test — Does a Learned Mode-Mixing Weight Beat the
Fixed Triadic Formula on Rotation Robustness?

Tests the concrete, falsifiable claim from the "shaped resonant cavity"
proposal: that giving the Observer/Shadow/Light mixing weights as a
TUNABLE parameter (fixed-symmetric preset, fixed-asymmetric preset,
learned/adaptive, or input-driven) beats the ORIGINAL fixed triadic
formula (out = s1*(1-s0) + s2*s0) on CIFAR-10 rotation robustness.

This does NOT implement the speculative optical-cavity physics (Fano
resonance, "bat-shaped" mode weights, complex-valued mode phases) from
the proposal — that machinery has unverifiable citations and some
questionable math, and isn't needed to test the actual claim, which is
just: "does tunable mode-mixing beat fixed mode-mixing?"

5 variants trained, all same architecture otherwise:
  baseline_original — out = s1*(1-s0) + s2*s0 (the existing 012 formula)
  fixed_symmetric   — out = 0.45*s0 + 0.1*s1 + 0.45*s2 (preset, no learning)
  fixed_asymmetric  — out = 0.1*s0  + 0.8*s1 + 0.1*s2 (preset, no learning)
  adaptive          — out = softmax(learned 3-param)·[s0,s1,s2]
  input_driven      — out = softmax(small predictor(x))·[s0,s1,s2], per-sample

This is an honest test — not engineered for any side to win.

Usage:
  python trit_symmetry_cavity_test.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ══════════════════════════════════════════════════════════════════════════════
# SYMMETRY-MODE TRIADIC BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class SymmetryBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mode="adaptive"):
        super().__init__()
        self.mode = mode
        self.s0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, padding=0), nn.BatchNorm2d(out_ch))
        self.s1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch))
        self.s2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 5, padding=2), nn.BatchNorm2d(out_ch))
        self.pred = nn.Conv2d(out_ch, in_ch, 1)

        if mode == "fixed_symmetric":
            self.register_buffer("weights", torch.tensor([0.45, 0.10, 0.45]))
        elif mode == "fixed_asymmetric":
            self.register_buffer("weights", torch.tensor([0.10, 0.80, 0.10]))
        elif mode == "adaptive":
            self.weights_param = nn.Parameter(torch.ones(3))
        elif mode == "input_driven":
            self.predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(in_ch, 3))
        # mode == "baseline_original" needs no extra params

    def forward(self, x):
        s0 = torch.sigmoid(self.s0(x))
        s1 = torch.tanh(self.s1(x))
        s2 = torch.tanh(self.s2(x))

        if self.mode == "baseline_original":
            out = s1 * (1 - s0) + s2 * s0
        elif self.mode in ("fixed_symmetric", "fixed_asymmetric"):
            w = self.weights
            out = w[0]*s0 + w[1]*s1 + w[2]*s2
        elif self.mode == "adaptive":
            w = torch.softmax(self.weights_param, dim=0)
            out = w[0]*s0 + w[1]*s1 + w[2]*s2
        elif self.mode == "input_driven":
            w = torch.softmax(self.predictor(x), dim=1)
            w0, w1, w2 = w[:,0].view(-1,1,1,1), w[:,1].view(-1,1,1,1), w[:,2].view(-1,1,1,1)
            out = w0*s0 + w1*s1 + w2*s2

        return out, self.pred(out), x

class SymmetryNet(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, mode="adaptive"):
        super().__init__()
        self.b1 = SymmetryBlock(in_ch, 32, mode)
        self.b2 = SymmetryBlock(32, 64, mode)
        self.b3 = SymmetryBlock(64, 128, mode)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(128, num_classes)

    def forward(self, x):
        preds = []
        o, p, i = self.b1(x); preds.append((p, i)); o = self.pool(o)
        o, p, i = self.b2(o); preds.append((p, i)); o = self.pool(o)
        o, p, i = self.b3(o); preds.append((p, i)); o = self.pool(o)
        feat = self.gap(o).squeeze(-1).squeeze(-1)
        return self.cls(feat), preds

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

mu, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
tr = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(), transforms.Normalize(mu, std)])
te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])
train_data = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=tr)
test_data  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=te)
tr_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
te_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
n_test = len(test_data)

EPOCHS = 15
PRED_WEIGHT = 0.01
ANGLES = [0, 45, 90, 135, 180, 270]

def train(model, label):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits, preds = model(imgs)
            cls_loss = ce(logits, labels)
            pred_loss = sum(F.mse_loss(p, a.detach()) for p, a in preds)
            loss = cls_loss + PRED_WEIGHT * pred_loss
            loss.backward(); opt.step(); total += loss.item()
        sch.step()
        print(f"  [{label}] epoch {epoch+1:>2}/{EPOCHS} loss={total/len(tr_loader):.4f} ({time.time()-t0:.0f}s)")

@torch.no_grad()
def evaluate(model):
    model.eval()
    results = {}
    for angle in ANGLES:
        correct = 0
        for imgs, labels in te_loader:
            imgs = TF.rotate(imgs.to(device), angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            logits, _ = model(imgs)
            correct += (logits.argmax(1) == labels.to(device)).sum().item()
        results[angle] = correct / n_test * 100
    return results

def stability(acc): return acc[0] - min(acc.values())
def norm_stability(acc): return (acc[0] - min(acc.values())) / acc[0] * 100

if __name__ == "__main__":
    modes = ["baseline_original", "fixed_symmetric", "fixed_asymmetric", "adaptive", "input_driven"]
    all_results = {}

    for mode in modes:
        print(f"\n{'='*60}\n  Training: {mode}\n{'='*60}")
        model = SymmetryNet(10, 3, mode=mode).to(device)
        train(model, mode)
        acc = evaluate(model)
        all_results[mode] = acc
        print(f"  Result: 0°={acc[0]:.2f}%  norm_stability={norm_stability(acc):.2f}%")

    print("\n" + "="*75)
    print("  RESULTS — Tunable Symmetry vs Fixed Original Triadic Formula")
    print("="*75)
    print(f"{'Mode':<20} | {'0°':>6} | {'45°':>6} | {'90°':>6} | {'180°':>6} | {'Stab↓':>6} | {'NormStab↓':>10}")
    print("-"*75)
    for mode, acc in all_results.items():
        print(f"{mode:<20} | {acc[0]:>5.2f}% | {acc[45]:>5.2f}% | {acc[90]:>5.2f}% | "
              f"{acc[180]:>5.2f}% | {stability(acc):>5.2f}% | {norm_stability(acc):>9.2f}%")

    baseline_ns = norm_stability(all_results["baseline_original"])
    print(f"\n  Baseline (original fixed formula) normalized stability: {baseline_ns:.2f}%")
    best_mode = min(modes, key=lambda m: norm_stability(all_results[m]))
    best_ns = norm_stability(all_results[best_mode])
    if best_mode != "baseline_original" and best_ns < baseline_ns:
        print(f"  Best tunable variant: {best_mode} ({best_ns:.2f}%, "
              f"{baseline_ns-best_ns:.2f}pp better than baseline)")
        print(f"  → Tunable mode-mixing DOES beat the fixed original formula.")
    else:
        print(f"  → No tunable variant beat the original fixed formula.")
