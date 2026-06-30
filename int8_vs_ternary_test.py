"""
012 Ternary vs INT8 Quantization — Real Head-to-Head

INT8 is the industry-standard quantization format with native hardware
support (CPU SIMD, Tensor Cores, ONNX Runtime, TensorRT) — unlike ternary,
which has no native runtime support on consumer hardware today. This
compares both against the same float32 baseline on the same architecture,
same data, same training budget, to see which actually wins on accuracy
per bit of storage.

Float32 : 32 bits/weight,  1x compression (baseline)
Ternary  : 1.585 bits/weight (packed), ~20.2x compression
INT8     : 8 bits/weight, 4x compression

Usage:
  python int8_vs_ternary_test.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ══════════════════════════════════════════════════════════════════════════════
# TERNARY QUANTIZATION (same as precision_loss_test.py)
# ══════════════════════════════════════════════════════════════════════════════

class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        t = 0.7 * x.abs().mean()
        ctx.save_for_backward(x)
        return torch.where(x > t, torch.ones_like(x),
               torch.where(x < -t, -torch.ones_like(x), torch.zeros_like(x)))
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).float()
tq = TernaryQuantize.apply

# ══════════════════════════════════════════════════════════════════════════════
# INT8 QUANTIZATION (symmetric, per-tensor, fake-quant with STE — same
# training-time simulation style as ternary, so it's a fair comparison)
# ══════════════════════════════════════════════════════════════════════════════

class Int8Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        scale = x.abs().max() / 127.0 + 1e-8
        q = torch.round(x / scale).clamp(-127, 127)
        return q * scale
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).float()
iq = Int8Quantize.apply

class QuantConv2d(nn.Conv2d):
    def __init__(self, *a, mode="float32", **k):
        super().__init__(*a, **k)
        self.mode = mode  # "float32" | "ternary" | "int8"
        self.active = False
    def forward(self, x):
        if self.active and self.mode == "ternary":
            w = tq(self.weight)
        elif self.active and self.mode == "int8":
            w = iq(self.weight)
        else:
            w = self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding)

class QuantLinear(nn.Linear):
    def __init__(self, *a, mode="float32", **k):
        super().__init__(*a, **k)
        self.mode = mode
        self.active = False
    def forward(self, x):
        if self.active and self.mode == "ternary":
            w = tq(self.weight)
        elif self.active and self.mode == "int8":
            w = iq(self.weight)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)

def set_active(model, active):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            m.active = active

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mode):
        super().__init__()
        self.conv = nn.Sequential(
            QuantConv2d(in_ch, out_ch, 3, padding=1, mode=mode),
            nn.BatchNorm2d(out_ch), nn.ReLU())
    def forward(self, x): return self.conv(x)

class CNN(nn.Module):
    def __init__(self, num_classes=10, mode="float32"):
        super().__init__()
        self.b1 = CNNBlock(3, 96, mode)
        self.b2 = CNNBlock(96, 192, mode)
        self.b3 = CNNBlock(192, 128, mode)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls = QuantLinear(128, num_classes, mode=mode)
    def forward(self, x):
        x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
        return self.cls(self.gap(x).squeeze(-1).squeeze(-1))

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
QUANT_WARMUP = 4

def train(model, label, is_quant):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(EPOCHS):
        if is_quant: set_active(model, epoch >= QUANT_WARMUP)
        model.train()
        total = 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = ce(model(imgs), labels)
            loss.backward(); opt.step(); total += loss.item()
        sch.step()
        print(f"  [{label}] epoch {epoch+1:>2}/{EPOCHS} loss={total/len(tr_loader):.4f} ({time.time()-t0:.0f}s)")

@torch.no_grad()
def evaluate(model, is_quant):
    if is_quant: set_active(model, True)
    model.eval()
    correct = 0
    for imgs, labels in te_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
    return correct / n_test * 100

results = {}
for mode, label in [("float32", "Float32"), ("ternary", "Ternary"), ("int8", "INT8")]:
    print(f"\n{'='*60}\n  Training {label}...\n{'='*60}")
    model = CNN(10, mode=mode).to(device)
    train(model, label, is_quant=(mode != "float32"))
    acc = evaluate(model, is_quant=(mode != "float32"))
    results[mode] = acc

print("\n" + "="*60)
print("  RESULT — Ternary vs INT8 vs Float32")
print("="*60)
print(f"  Float32 accuracy : {results['float32']:.2f}%")
print(f"  Ternary accuracy : {results['ternary']:.2f}%  (lost {results['float32']-results['ternary']:.2f}pp, ~20.2x compression, 1.585 bits/weight)")
print(f"  INT8 accuracy    : {results['int8']:.2f}%  (lost {results['float32']-results['int8']:.2f}pp, 4x compression, 8 bits/weight)")
print(f"\n  Accuracy-per-compression tradeoff:")
print(f"  Ternary loses {results['float32']-results['ternary']:.2f}pp for 20.2x smaller")
print(f"  INT8 loses    {results['float32']-results['int8']:.2f}pp for 4x smaller")
if results['int8'] > results['ternary']:
    print(f"\n  INT8 wins on accuracy ({results['int8']-results['ternary']:.2f}pp higher) "
          f"AND has native hardware support — the practical industry choice.")
else:
    print(f"\n  Ternary wins on accuracy despite far higher compression — notable result.")
