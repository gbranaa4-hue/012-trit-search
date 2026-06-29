"""
012 Ternary Architecture — with Persistent Memory Encoding
Extends TritCognition with a TritMemoryCell that persists state
across a sequence of inputs (video frames, sensor streams, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

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

# ══════════════════════════════════════════════════════════════════════════════
# TRIT MEMORY CELL
#
# Three gates, all ternary weighted:
#
#   forget_gate f = σ(W_f · x)          what fraction of memory to erase
#   write_gate  w = tanh(W_w · x)       what new content to write
#   read_gate   r = σ(W_r · x)          how much memory to mix into output
#
# Memory update:
#   M_t = M_{t-1} * (1 - f) + w * f    erase old, write new
#
# Output:
#   out = x + r * M_t                   residual: input + memory readout
#
# Trit interpretation:
#   M = -1  this feature was suppressed in past observations  (Observer)
#   M =  0  this feature is unknown / not yet seen            (Shadow)
#   M = +1  this feature was activated in past observations   (Light)
#
# Hardware: memory cell is a ternary register bank
#           gates are ternary MAC units
#           update is add/subtract/skip — no multipliers
# ══════════════════════════════════════════════════════════════════════════════

class TritMemoryCell(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size   = size
        self.forget = TernaryLinear(size, size, quantize=False)  # erase gate
        self.write  = TernaryLinear(size, size, quantize=False)  # write gate
        self.read   = TernaryLinear(size, size, quantize=False)  # read gate
        self.norm   = nn.LayerNorm(size)

        # Persistent memory buffer — survives across forward calls
        # Initialized to 0 (neutral — nothing observed yet)
        self.register_buffer('memory', torch.zeros(1, size))

    def forward(self, x):
        """
        x     : (batch, size) — current input features
        memory: (1,     size) — persistent trit state from previous observations
        """
        # Expand memory to match batch size
        mem = self.memory.expand(x.size(0), -1)

        # Gates
        f = torch.sigmoid(self.forget(x))        # forget: what to erase  [0,1]
        w = torch.tanh(self.write(x))             # write:  new content   [-1,1]
        r = torch.sigmoid(self.read(x))           # read:   how much to use [0,1]

        # Memory update: erase fraction f of old memory, write new content
        new_mem = mem * (1 - f) + w * f           # M_t

        # Store mean across batch (so memory persists meaningfully)
        self.memory = new_mem.mean(dim=0, keepdim=True).detach()

        # Output: residual connection — input + scaled memory readout
        out = x + r * new_mem

        return self.norm(out), new_mem

    def reset(self):
        """Reset memory to neutral state (start of new sequence)"""
        self.memory.zero_()

    def trit_state(self):
        """
        Snap current memory to nearest trit for hardware readout.
        Returns distribution: fraction of {-1, 0, +1} cells.
        """
        t    = 0.7 * self.memory.abs().mean()
        snap = torch.where(self.memory >  t,  torch.ones_like(self.memory),
               torch.where(self.memory < -t, -torch.ones_like(self.memory),
               torch.zeros_like(self.memory)))
        total = snap.numel()
        neg   = (snap == -1).sum().item() / total * 100
        zero  = (snap ==  0).sum().item() / total * 100
        pos   = (snap ==  1).sum().item() / total * 100
        return neg, zero, pos


# ══════════════════════════════════════════════════════════════════════════════
# TRIADIC BLOCK (same as before)
# ══════════════════════════════════════════════════════════════════════════════

class PredictiveTritBlock(nn.Module):
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
        return cls + self.w * pred


# ══════════════════════════════════════════════════════════════════════════════
# TRIT COGNITION WITH MEMORY
#
# Architecture:
#   Input
#   → TriadicBlock(3→32)   + MaxPool
#   → TriadicBlock(32→64)  + MaxPool
#   → TriadicBlock(64→128) + MaxPool
#   → SpatialAttention
#   → GlobalAvgPool          (128-dim feature vector)
#   → TritMemoryCell         ← NEW: persistent memory across observations
#   → TernaryLinear(128→N)
#
# The memory cell sits between the feature extractor and classifier.
# It accumulates a ternary state across a sequence of images,
# allowing the network to recognize patterns that span multiple observations.
# ══════════════════════════════════════════════════════════════════════════════

class TritCognitionWithMemory(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, memory_size=128):
        super().__init__()
        self.b1   = PredictiveTritBlock(in_ch, 32)
        self.b2   = PredictiveTritBlock(32, 64)
        self.b3   = PredictiveTritBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            TernaryConv2d(128, 32, 1, quantize=False), nn.ReLU(),
            TernaryConv2d(32,   1, 1, quantize=False), nn.Sigmoid()
        )
        # Memory cell — persists trit state across forward passes
        self.memory = TritMemoryCell(memory_size)
        self.cls    = TernaryLinear(memory_size, num_classes, quantize=False)

    def forward(self, x, use_memory=True):
        preds = []
        o, p, i = self.b1(x);  preds.append((p,i)); o = self.pool(o)
        o, p, i = self.b2(o);  preds.append((p,i)); o = self.pool(o)
        o, p, i = self.b3(o);  preds.append((p,i)); o = self.pool(o)
        o    = o * self.attn(o)
        feat = self.gap(o).squeeze(-1).squeeze(-1)   # (B, 128)

        if use_memory:
            feat, mem_state = self.memory(feat)      # (B, 128) enriched with memory
        else:
            mem_state = None

        return self.cls(feat), preds, mem_state

    def reset_memory(self):
        self.memory.reset()


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE DATASET
# Wraps CIFAR-10 into sequences of N images from same class
# Simulates a sensor stream where the same object appears repeatedly
# ══════════════════════════════════════════════════════════════════════════════

class SequenceDataset(torch.utils.data.Dataset):
    """
    Groups CIFAR-10 images into sequences of length seq_len.
    Each sequence contains images from the same class.
    The memory cell should accumulate evidence across the sequence
    and improve classification by the final frame.
    """
    def __init__(self, dataset, seq_len=4):
        self.seq_len = seq_len
        # Group indices by class
        self.by_class = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.by_class:
                self.by_class[label] = []
            self.by_class[label].append(idx)
        self.dataset  = dataset
        self.sequences = self._build_sequences()

    def _build_sequences(self):
        seqs = []
        for label, indices in self.by_class.items():
            for i in range(0, len(indices) - self.seq_len, self.seq_len):
                seqs.append((indices[i:i+self.seq_len], label))
        return seqs

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        indices, label = self.sequences[idx]
        imgs = torch.stack([self.dataset[i][0] for i in indices])
        return imgs, label   # (seq_len, C, H, W), label


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

mu, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mu, std)
])
test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])

cifar_train = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
cifar_test  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

# Standard single-image loaders (compare memory vs no-memory)
train_loader = DataLoader(Subset(cifar_train, range(10000)), batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(cifar_test,                        batch_size=128, shuffle=False, num_workers=0)

# Sequence loaders (4 images per sequence, same class)
SEQ_LEN = 4
seq_train = SequenceDataset(Subset(cifar_train, range(10000)), seq_len=SEQ_LEN)
seq_test  = SequenceDataset(cifar_test,                        seq_len=SEQ_LEN)
seq_train_loader = DataLoader(seq_train, batch_size=32,  shuffle=True,  num_workers=0)
seq_test_loader  = DataLoader(seq_test,  batch_size=64,  shuffle=False, num_workers=0)

print(f"Standard train : {len(train_loader.dataset)} images")
print(f"Sequence train : {len(seq_train)} sequences of {SEQ_LEN}")
print(f"Standard test  : {len(test_loader.dataset)} images")
print(f"Sequence test  : {len(seq_test)} sequences\n")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

EPOCHS       = 30
QUANT_WARMUP = 6

def train_with_memory(model, loader, seq_loader, label):
    """
    Two-phase training:
    Phase 1 (single images): learn basic feature extraction
    Phase 2 (sequences):     learn to accumulate memory across frames
    """
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = PredLoss(w=0.01)

    for epoch in range(EPOCHS):
        set_quant(model, epoch >= QUANT_WARMUP)
        model.train()
        total = 0

        if epoch < EPOCHS // 2:
            # Phase 1: single images, no memory accumulation
            model.reset_memory()
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                logits, preds, _ = model(imgs, use_memory=False)
                loss = loss_fn(logits, labels, preds)
                loss.backward()
                opt.step()
                total += loss.item()
        else:
            # Phase 2: sequences, memory accumulates across frames
            for seqs, labels in seq_loader:
                # seqs: (B, seq_len, C, H, W)
                seqs, labels = seqs.to(device), labels.to(device)
                model.reset_memory()
                opt.zero_grad()

                seq_loss = 0
                for t in range(SEQ_LEN):
                    frame             = seqs[:, t]              # (B, C, H, W)
                    logits, preds, _  = model(frame, use_memory=True)
                    # Weight later frames more — memory should help over time
                    weight    = (t + 1) / SEQ_LEN
                    seq_loss += weight * loss_fn(logits, labels, preds)

                seq_loss.backward()
                opt.step()
                total += seq_loss.item()

        sch.step()
        phase = "TERNARY" if epoch >= QUANT_WARMUP else "warmup"
        train_phase = "single" if epoch < EPOCHS // 2 else "sequence"
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  [{label}|{phase}|{train_phase}] epoch {epoch+1:>2}/{EPOCHS}  loss={total/len(loader):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

ANGLES = [0, 45, 90, 135, 180, 270]

def evaluate_single(model, loader, use_memory=False):
    """Standard single-image evaluation with rotation test."""
    set_quant(model, True)
    model.eval()
    results = {}
    with torch.no_grad():
        for angle in ANGLES:
            correct = 0
            model.reset_memory()
            for imgs, labels in loader:
                imgs  = TF.rotate(imgs.to(device), angle,
                                  interpolation=TF.InterpolationMode.BILINEAR,
                                  fill=0)
                out, _, _ = model(imgs, use_memory=use_memory)
                correct  += (out.argmax(1) == labels.to(device)).sum().item()
            results[angle] = correct / len(loader.dataset) * 100
    return results

def evaluate_sequence(model, loader):
    """
    Sequence evaluation: measure accuracy at each frame position.
    Frame 1: memory empty   — no prior context
    Frame 2: memory has 1 prior observation
    Frame 3: memory has 2 prior observations
    Frame 4: memory has 3 prior observations
    Should improve across frames as memory accumulates.
    """
    set_quant(model, True)
    model.eval()
    frame_acc = {t: 0 for t in range(SEQ_LEN)}
    total     = 0

    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            model.reset_memory()
            B = seqs.size(0)
            total += B

            for t in range(SEQ_LEN):
                frame            = seqs[:, t]
                out, _, mem_state = model(frame, use_memory=True)
                frame_acc[t]     += (out.argmax(1) == labels).sum().item()

    return {t: frame_acc[t] / total * 100 for t in range(SEQ_LEN)}

def evaluate_rotation_sequence(model, loader, angle):
    """
    Sequence evaluation under rotation.
    Tests if memory helps recover accuracy when images are rotated.
    """
    set_quant(model, True)
    model.eval()
    frame_acc = {t: 0 for t in range(SEQ_LEN)}
    total     = 0

    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            model.reset_memory()
            B = seqs.size(0)
            total += B

            for t in range(SEQ_LEN):
                frame = TF.rotate(seqs[:, t], angle,
                                  interpolation=TF.InterpolationMode.BILINEAR,
                                  fill=0)
                out, _, _ = model(frame, use_memory=True)
                frame_acc[t] += (out.argmax(1) == labels).sum().item()

    return {t: frame_acc[t] / total * 100 for t in range(SEQ_LEN)}


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

model = TritCognitionWithMemory(num_classes=10, in_ch=3, memory_size=128).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Base TritCognition : 396,174")
print(f"  TritMemoryCell adds: {sum(p.numel() for p in model.memory.parameters()):,}\n")

print("Training TritCognition with Memory...")
train_with_memory(model, train_loader, seq_train_loader, "TritMem")

# ── Standard rotation benchmark ───────────────────────────────────────────────
print("\n── Single image evaluation (no memory)")
acc_nomem = evaluate_single(model, test_loader, use_memory=False)
print("\n── Single image evaluation (with memory)")
acc_mem   = evaluate_single(model, test_loader, use_memory=True)

print(f"\n{'Angle':<8} | {'No Memory':>10} | {'With Memory':>12} | {'Delta':>7}")
print("-" * 45)
for a in ANGLES:
    d = acc_mem[a] - acc_nomem[a]
    print(f"{a:<7}° | {acc_nomem[a]:>9.2f}% | {acc_mem[a]:>11.2f}% | {'+' if d>=0 else ''}{d:.2f}%")

stab_no  = acc_nomem[0] - min(acc_nomem.values())
stab_mem = acc_mem[0]   - min(acc_mem.values())
print(f"\nStability (no memory)  : -{stab_no:.2f}%")
print(f"Stability (with memory): -{stab_mem:.2f}%")

# ── Sequence evaluation ────────────────────────────────────────────────────────
print("\n── Sequence evaluation (memory accumulates across frames)")
seq_acc = evaluate_sequence(model, seq_test_loader)
print(f"\n{'Frame':<8} | {'Accuracy':>10} | {'Memory state'}")
print("-" * 50)
for t in range(SEQ_LEN):
    neg, zero, pos = model.memory.trit_state()
    tag = "← memory empty" if t == 0 else f"← -{neg:.0f}% 0:{zero:.0f}% +{pos:.0f}%"
    print(f"Frame {t+1:<3} | {seq_acc[t]:>9.2f}% | {tag}")

improvement = seq_acc[SEQ_LEN-1] - seq_acc[0]
print(f"\nAccuracy gain frame 1→{SEQ_LEN}: {'+' if improvement>=0 else ''}{improvement:.2f}%")
print("(positive = memory helps over time)")

# ── Sequence under rotation ────────────────────────────────────────────────────
print("\n── Sequence evaluation under 90° rotation")
seq_rot = evaluate_rotation_sequence(model, seq_test_loader, angle=90)
print(f"\n{'Frame':<8} | {'90° Accuracy':>13}")
print("-" * 25)
for t in range(SEQ_LEN):
    print(f"Frame {t+1:<3} | {seq_rot[t]:>12.2f}%")
rot_improvement = seq_rot[SEQ_LEN-1] - seq_rot[0]
print(f"\nGain frame 1→{SEQ_LEN} under 90° rotation: {'+' if rot_improvement>=0 else ''}{rot_improvement:.2f}%")

# ── Memory trit state after full test set ────────────────────────────────────
neg, zero, pos = model.memory.trit_state()
print(f"\n── Final memory trit state (after seeing test set):")
print(f"  -1 (suppressed) : {neg:.1f}%")
print(f"   0 (neutral)    : {zero:.1f}%")
print(f"  +1 (activated)  : {pos:.1f}%")
print(f"  Active cells    : {neg+pos:.1f}% — these encode learned patterns")

# Save
results = {
    "single_no_memory" : {str(k): v for k,v in acc_nomem.items()},
    "single_with_memory": {str(k): v for k,v in acc_mem.items()},
    "sequence_accuracy" : {str(k): v for k,v in seq_acc.items()},
    "sequence_90deg"    : {str(k): v for k,v in seq_rot.items()},
    "memory_trit_state" : {"neg": neg, "zero": zero, "pos": pos},
}
import json
with open("results/012_memory_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/012_memory_benchmark.json")

# ── Hardware memory estimate ───────────────────────────────────────────────────
mem_params = sum(p.numel() for p in model.memory.parameters())
print(f"\n── Memory cell hardware footprint:")
print(f"  Parameters      : {mem_params:,}")
print(f"  Ternary storage : {mem_params*1.585/8/1024:.2f} KB")
print(f"  State vector    : 128 trits = {128*1.585/8:.1f} bytes")
print(f"  On FPGA         : ~{mem_params*2} flip-flops (2 bits per trit)")
print(f"  Update per step : {128*3} ternary MACs (3 gates × 128 weights)")
print(f"  Compare LSTM    : {128*4*128*2:,} ops per step (4 gates, float32 weights)")
print(f"  Speedup vs LSTM : ~{128*4*128*2 / (128*3):.0f}x fewer operations")
