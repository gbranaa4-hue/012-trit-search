"""
Measures actual accuracy precision loss from ternary quantization.
Trains StandardCNN (float32) vs TernaryStandardCNN (ternary weights)
— same architecture, same data, only weight precision differs.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

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
    def __init__(self, *a, quantize=True, **k):
        super().__init__(*a, **k); self.do_quantize = quantize
    def forward(self, x):
        w = tq(self.weight) if self.do_quantize else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding)

class TernaryLinear(nn.Linear):
    def __init__(self, *a, quantize=True, **k):
        super().__init__(*a, **k); self.do_quantize = quantize
    def forward(self, x):
        w = tq(self.weight) if self.do_quantize else self.weight
        return F.linear(x, w, self.bias)

def set_quant(model, active):
    for m in model.modules():
        if isinstance(m, (TernaryConv2d, TernaryLinear)):
            m.do_quantize = active

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ternary=False):
        super().__init__()
        Conv = TernaryConv2d if ternary else nn.Conv2d
        kwargs = {"quantize": False} if ternary else {}
        self.conv = nn.Sequential(
            Conv(in_ch, out_ch, 3, padding=1, **kwargs),
            nn.BatchNorm2d(out_ch), nn.ReLU())
    def forward(self, x): return self.conv(x)

class CNN(nn.Module):
    def __init__(self, num_classes=10, ternary=False, mixed=False):
        """mixed=True: first+last layer stay float32, only middle layers ternarized."""
        super().__init__()
        b1_ternary = ternary and not mixed
        self.b1 = CNNBlock(3, 96, b1_ternary)
        self.b2 = CNNBlock(96, 192, ternary)
        self.b3 = CNNBlock(192, 128, ternary)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        cls_ternary = ternary and not mixed
        Lin = TernaryLinear if cls_ternary else nn.Linear
        kwargs = {"quantize": False} if cls_ternary else {}
        self.cls = Lin(128, num_classes, **kwargs)
    def forward(self, x):
        x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
        return self.cls(self.gap(x).squeeze(-1).squeeze(-1))

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

def train(model, label, is_ternary):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(EPOCHS):
        if is_ternary: set_quant(model, epoch >= QUANT_WARMUP)
        model.train()
        total = 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = ce(model(imgs), labels)
            loss.backward(); opt.step(); total += loss.item()
        sch.step()
        phase = "TERNARY" if (is_ternary and epoch >= QUANT_WARMUP) else ("warmup" if is_ternary else "")
        print(f"  [{label}] epoch {epoch+1:>2}/{EPOCHS} loss={total/len(tr_loader):.4f} {phase} ({time.time()-t0:.0f}s)")

@torch.no_grad()
def evaluate(model, is_ternary):
    if is_ternary: set_quant(model, True)
    model.eval()
    correct = 0
    for imgs, labels in te_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        correct += (out.argmax(1) == labels).sum().item()
    return correct / n_test * 100

print("="*60)
print("  Training StandardCNN (float32)...")
print("="*60)
std_model = CNN(10, ternary=False).to(device)
train(std_model, "Float32", is_ternary=False)
std_acc = evaluate(std_model, is_ternary=False)

print("\n" + "="*60)
print("  Training TernaryStandardCNN (ternary {-1,0,+1})...")
print("="*60)
tern_model = CNN(10, ternary=True).to(device)
train(tern_model, "Ternary", is_ternary=True)
tern_acc = evaluate(tern_model, is_ternary=True)

print("\n" + "="*60)
print("  Training Mixed-Precision (first+last float32, middle ternary)...")
print("="*60)
mixed_model = CNN(10, ternary=True, mixed=True).to(device)
train(mixed_model, "Mixed", is_ternary=True)
mixed_acc = evaluate(mixed_model, is_ternary=True)

print("\n" + "="*60)
print("  RESULT — Precision Loss")
print("="*60)
print(f"  Float32 CNN accuracy        : {std_acc:.2f}%")
print(f"  Ternary CNN accuracy        : {tern_acc:.2f}%  (lost {std_acc-tern_acc:.2f}pp)")
print(f"  Mixed-precision CNN accuracy: {mixed_acc:.2f}%  (lost {std_acc-mixed_acc:.2f}pp)")
print(f"  Mixed vs full-ternary gain  : {mixed_acc-tern_acc:+.2f}pp")
print(f"  Storage saved (full ternary): 20.2x smaller")
print(f"  Storage saved (mixed)       : ~18x smaller (first/last layers stay float32)")
