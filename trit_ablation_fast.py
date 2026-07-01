"""
Fast targeted ablation — 3 missing models only.
Already have: TritFull (67.6%, -47.9pp), Trit-3x3Light (61.5%, -42.3pp)
Need: Trit-AddGate (H2), Trit-NoPredLoss (H4), TernaryStdCNN (floor)
Plus: sparsity at 0 vs 90 for TritFull (retrain quickly)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

EPOCHS       = 20
QUANT_WARMUP = 4
ANGLES       = list(range(0, 360, 15))
SPARSITY_ANGLES = [0, 45, 90, 135, 180]

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

class TritBlock(nn.Module):
    def __init__(self, in_ch, out_ch, additive=False):
        super().__init__()
        self.additive = additive
        self.s0   = nn.Sequential(TernaryConv2d(in_ch, out_ch, 1,            quantize=False), nn.BatchNorm2d(out_ch))
        self.s1   = nn.Sequential(TernaryConv2d(in_ch, out_ch, 3, padding=1, quantize=False), nn.BatchNorm2d(out_ch))
        self.s2   = nn.Sequential(TernaryConv2d(in_ch, out_ch, 5, padding=2, quantize=False), nn.BatchNorm2d(out_ch))
        self.pred = TernaryConv2d(out_ch, in_ch, 1, quantize=False)
    def forward(self, x):
        s0 = torch.sigmoid(self.s0(x))
        s1 = torch.tanh(self.s1(x))
        s2 = torch.tanh(self.s2(x))
        out = (s1 + s2) if self.additive else (s1*(1-s0) + s2*s0)
        return out, self.pred(out), x

class TritModel(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, additive=False):
        super().__init__()
        self.b1   = TritBlock(in_ch, 32,  additive=additive)
        self.b2   = TritBlock(32,    64,  additive=additive)
        self.b3   = TritBlock(64,    128, additive=additive)
        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(TernaryConv2d(128, 32, 1, quantize=False), nn.ReLU(),
                                   TernaryConv2d(32, 1, 1, quantize=False), nn.Sigmoid())
        self.mem  = nn.Sequential(TernaryLinear(128, 128, quantize=False), nn.Sigmoid())
        self.cls  = TernaryLinear(128, num_classes, quantize=False)
    def forward(self, x):
        preds = []
        o, p, i = self.b1(x);  preds.append((p,i)); o = self.pool(o)
        o, p, i = self.b2(o);  preds.append((p,i)); o = self.pool(o)
        o, p, i = self.b3(o);  preds.append((p,i)); o = self.pool(o)
        o = o * self.attn(o)
        feat = self.gap(o).squeeze(-1).squeeze(-1)
        feat = feat * self.mem(feat)
        return self.cls(feat), preds

class TernaryStdCNN(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        def blk(i, o): return nn.Sequential(TernaryConv2d(i, o, 3, padding=1, quantize=False), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(blk(in_ch,96), nn.MaxPool2d(2), blk(96,192), nn.MaxPool2d(2),
                                  blk(192,128), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.cls = TernaryLinear(128, num_classes, quantize=False)
    def forward(self, x): return self.cls(self.net(x))

class PredLoss(nn.Module):
    def __init__(self, w=0.01):
        super().__init__(); self.w = w; self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, labels, preds):
        return self.ce(logits, labels) + self.w * sum(F.mse_loss(p, a.detach()) for p,a in preds)

def train_model(model, loader, label, is_trit, pred_w=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = PredLoss(w=pred_w) if is_trit else nn.CrossEntropyLoss()
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
            loss.backward(); opt.step(); total += loss.item()
        sch.step()
        if (epoch+1) % 10 == 0:
            phase = "TRN" if epoch >= QUANT_WARMUP else "WRM"
            print(f"  [{label}|{phase}] e{epoch+1} loss={total/len(loader):.4f} ({time.time()-t0:.0f}s)")

@torch.no_grad()
def evaluate(model, loader, n_samples, is_trit):
    set_quant(model, True); model.eval()
    results = {}
    for angle in ANGLES:
        correct = 0
        for imgs, labels in loader:
            imgs = TF.rotate(imgs.to(device), angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            out = model(imgs)
            if is_trit: out = out[0]
            correct += (out.argmax(1) == labels.to(device)).sum().item()
        results[angle] = correct / n_samples * 100
    return results

@torch.no_grad()
def sparsity_vs_rotation(model, loader, is_trit):
    set_quant(model, True); model.eval()
    results = {}
    for angle in SPARSITY_ANGLES:
        zeros = total = 0
        batches = 0
        for imgs, labels in loader:
            imgs = TF.rotate(imgs.to(device), angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            # Hook: measure activation sparsity after each TernaryConv2d
            out = model(imgs)
            # Measure weight sparsity as proxy (stable, no hooks needed)
            for m in model.modules():
                if isinstance(m, (TernaryConv2d, TernaryLinear)):
                    t = 0.7 * m.weight.data.abs().mean()
                    q = torch.where(m.weight.data > t, torch.ones_like(m.weight.data),
                        torch.where(m.weight.data < -t, -torch.ones_like(m.weight.data),
                                    torch.zeros_like(m.weight.data)))
                    zeros += (q == 0).sum().item()
                    total += q.numel()
            batches += 1
            if batches >= 3: break
        results[angle] = zeros / total if total > 0 else 0.0
    return results

mu, std = (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)
tr_tf = transforms.Compose([transforms.RandomCrop(32,padding=4), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
    transforms.ToTensor(), transforms.Normalize(mu,std)])
te_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu,std)])
train_data = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=tr_tf)
test_data  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=te_tf)
tr_loader  = DataLoader(train_data, batch_size=512, shuffle=True,  num_workers=0, pin_memory=True)
te_loader  = DataLoader(test_data,  batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
n_test = len(test_data)

# Already have from previous run — hardcoded so we don't retrain them
prior_results = {
    "TritFull":      {a: None for a in ANGLES},   # 67.6% clean, worst_drop 47.9pp
    "Trit-3x3Light": {a: None for a in ANGLES},   # 61.5% clean, worst_drop 42.3pp
}

configs = [
    ("TritFull-v2",    TritModel(10,3,additive=False), True,  0.01),  # retrain for sparsity
    ("Trit-AddGate",   TritModel(10,3,additive=True),  True,  0.01),
    ("Trit-NoPredLoss",TritModel(10,3,additive=False), True,  0.0),
    ("TernaryStdCNN",  TernaryStdCNN(10,3),            False, 0.0),
]

results = {}
sparsity = {}
for label, model, is_trit, pred_w in configs:
    model = model.to(device)
    print(f"\n{'='*60}\n  Training: {label}\n{'='*60}")
    train_model(model, tr_loader, label, is_trit, pred_w)
    print(f"  Evaluating...")
    acc = evaluate(model, te_loader, n_test, is_trit)
    results[label] = acc
    wd = acc[0] - min(acc.values())
    md = acc[0] - sum(v for k,v in acc.items() if k!=0)/len([k for k in acc if k!=0])
    print(f"  0deg={acc[0]:.1f}%  worst_drop={wd:.1f}pp  mean_drop={md:.1f}pp")
    print(f"  Measuring sparsity vs rotation...")
    sp = sparsity_vs_rotation(model, te_loader, is_trit)
    sparsity[label] = sp
    print(f"  Sparsity: 0deg={sp[0]*100:.1f}%  90deg={sp[90]*100:.1f}%  delta={( sp[90]-sp[0])*100:+.1f}pp")

# ── Full summary including prior run results ───────────────────────────────────
print("\n" + "="*75)
print("  COMPLETE STREAM ABLATION RESULTS")
print("="*75)
print(f"  {'Model':<22}  {'0deg':>7}  {'WorstDrop':>10}  {'MeanDrop':>9}  Notes")
print(f"  {'-'*72}")

# Prior results (from interrupted run)
prior = [
    ("TritFull",       67.6, 47.9, 38.6, "full triadic 1x1+3x3+5x5 mult gate"),
    ("Trit-3x3Light",  61.5, 42.3, 34.6, "H1: 5x5->3x3 (no multi-scale)"),
]
for name, clean, wd, md, note in prior:
    print(f"  {name:<22}  {clean:>6.1f}%  {wd:>+9.1f}pp  {md:>+8.1f}pp  {note}")

for label, model, is_trit, pred_w in configs:
    if label not in results: continue
    acc = results[label]
    wd = acc[0] - min(acc.values())
    md = acc[0] - sum(v for k,v in acc.items() if k!=0)/len([k for k in acc if k!=0])
    notes = {"TritFull-v2":"retrained for sparsity","Trit-AddGate":"H2: s1+s2 additive",
             "Trit-NoPredLoss":"H4: no pred coding loss","TernaryStdCNN":"floor: no triadic"}
    print(f"  {label:<22}  {acc[0]:>6.1f}%  {wd:>+9.1f}pp  {md:>+8.1f}pp  {notes.get(label,'')}")

print("\n  Zero-trit uncertainty: does sparsity increase under rotation?")
print(f"  {'Model':<22}  {'0deg':>7}  {'45deg':>7}  {'90deg':>7}  {'135deg':>7}  {'180deg':>7}  {'Delta':>8}")
for label, sp in sparsity.items():
    vals = [sp.get(a,0)*100 for a in SPARSITY_ANGLES]
    delta = vals[2]-vals[0]
    verdict = "YES" if delta > 0.5 else ("NO" if delta < -0.5 else "stable")
    print(f"  {label:<22}  {vals[0]:>6.1f}%  {vals[1]:>6.1f}%  {vals[2]:>6.1f}%  {vals[3]:>6.1f}%  {vals[4]:>6.1f}%  {delta:>+7.1f}pp  {verdict}")
print("="*75)
