#!/usr/bin/env python3
"""
Does ternary hurt LESS on less-specific recognition? (granularity sweep)

Hypothesis (from the session): ternary quantization destroys FINE discrimination
(exact identity/class) more than COARSE recognition (category/superclass). Face
identity (LFW) collapsed under ternary-PTQ; CIFAR-10 (10 coarse classes) was
ternary-QAT ~ float. This isolates the axis cleanly: ONE model on CIFAR-100,
read out at FINE (100 classes) vs COARSE (20 superclasses) -- same images, same
weights, only the decision granularity changes.

PRE-REGISTERED (before running): ternary's accuracy gap to float is SMALLER at
the 20-way superclass readout than at the 100-way fine readout -> ternary keeps
'it's a vehicle' even when it loses 'car vs truck'. Confirm if
  (float_fine - tern_fine)  >  (float_coarse - tern_coarse) + 1pp.
Disconfirm if the gaps are equal (granularity irrelevant) or fine<coarse.

Same architecture / QAT recipe as binary_vs_ternary_vs_triadic.py (imported).
30 epochs, one seed -- judges the gap ORDERING, not a converged CIFAR-100 SOTA.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import datasets as hfds

from binary_vs_ternary_vs_triadic import QuantStandardCNN, tq, set_quant, device

EPOCHS, WARMUP, BATCH, SEED = 30, 6, 128, 0


class HFWrap(Dataset):
    """Wrap a HuggingFace cifar100 split -> (tensor img, fine_label)."""
    def __init__(self, split, tfm):
        self.imgs = split["img"]          # list of PIL
        self.fine = split["fine_label"]
        self.tfm = tfm
    def __len__(self): return len(self.fine)
    def __getitem__(self, i): return self.tfm(self.imgs[i].convert("RGB")), self.fine[i]


def loaders():
    mu, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    tr = transforms.Compose([transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(), transforms.Normalize(mu, std)])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, std)])
    ds = hfds.load_dataset("uoft-cs/cifar100")
    # derive the fine(100)->coarse(20) map straight from the dataset labels
    fl, cl = ds["train"]["fine_label"], ds["train"]["coarse_label"]
    m = {}
    for f, c in zip(fl, cl):
        m[f] = c
    coarse = np.array([m[i] for i in range(100)])
    tl = DataLoader(HFWrap(ds["train"], tr), BATCH, shuffle=True, num_workers=0, pin_memory=True)
    el = DataLoader(HFWrap(ds["test"], te), 256, shuffle=False, num_workers=0, pin_memory=True)
    return tl, el, coarse


def train(loader, quantize, label):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = QuantStandardCNN(tq, num_classes=100).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        set_quant(model, quantize and ep >= WARMUP); model.train(); tot = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = ce(model(x), y); loss.backward(); opt.step()
            tot += loss.item()
        sch.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [{label}] ep {ep+1}/{EPOCHS} loss={tot/len(loader):.3f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, loader, quantize, coarse):
    set_quant(model, quantize); model.eval()
    fine_ok = coarse_ok = n = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        yf = y.numpy()
        fine_ok += (pred == yf).sum()
        coarse_ok += (coarse[pred] == coarse[yf]).sum()
        n += len(yf)
    return fine_ok / n * 100, coarse_ok / n * 100


def main():
    print("=" * 72)
    print("GRANULARITY: does ternary hurt less on coarse recognition? (CIFAR-100)")
    print("=" * 72)
    tl, el, coarse = loaders()
    print(f"  CIFAR-100 loaded (HF): {len(set(coarse.tolist()))} superclasses over 100 classes")

    print("\n=== float ===")
    fm = train(tl, quantize=False, label="float")
    f_fine, f_coarse = evaluate(fm, el, quantize=False, coarse=coarse)
    print("\n=== ternary (QAT) ===")
    tm = train(tl, quantize=True, label="ternary")
    t_fine, t_coarse = evaluate(tm, el, quantize=True, coarse=coarse)

    fine_gap = f_fine - t_fine
    coarse_gap = f_coarse - t_coarse
    print("\n" + "=" * 72)
    print(f"  {'readout':<26}{'float':>9}{'ternary':>10}{'gap (float-tern)':>18}")
    print("  " + "-" * 62)
    print(f"  {'FINE  (100 classes)':<26}{f_fine:>8.2f}%{t_fine:>9.2f}%{fine_gap:>16.2f}pp")
    print(f"  {'COARSE(20 superclass)':<26}{f_coarse:>8.2f}%{t_coarse:>9.2f}%{coarse_gap:>16.2f}pp")

    print("\n" + "=" * 72)
    print("PRE-REGISTERED VERDICT")
    print("=" * 72)
    print(f"  ternary's cost is {fine_gap:.2f}pp at FINE vs {coarse_gap:.2f}pp at COARSE")
    if fine_gap > coarse_gap + 1.0:
        print(f"  => CONFIRMED: ternary hurts LESS on less-specific recognition")
        print(f"     (gap shrinks {fine_gap:.2f}pp -> {coarse_gap:.2f}pp when the task only needs")
        print(f"     the superclass). A ternary model too lossy for identity can still do")
        print(f"     detection / category / attribute recognition -- pick the task to the")
        print(f"     precision ternary leaves you.")
    elif abs(fine_gap - coarse_gap) <= 1.0:
        print(f"  => NULL: ternary's cost is the same at both granularities -- the")
        print(f"     'less-specific is easier for ternary' idea is NOT supported here.")
    else:
        print(f"  => REVERSED: ternary hurts MORE at coarse readout -- unexpected; report")
        print(f"     and investigate before interpreting.")
    print("\n[scope] 30 epochs, one seed, CIFAR-100 superclass as the coarse proxy. Judges")
    print("        the gap ordering (fine vs coarse), not a converged benchmark.")


if __name__ == "__main__":
    main()
