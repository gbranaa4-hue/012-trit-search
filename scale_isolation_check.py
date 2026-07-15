"""
Follow-up: isolate whether the 250x MSE gap in instrument_check.py is caused by
BCQ's multi-basis mechanism, or simply by BCQ having a per-row scale factor
that gbranaa4-hue's raw ternary_ptq() lacks (the "boring explanation" check).

Adds a scaled-ternary variant: same {-1,0,+1} sign pattern, but multiplied by
alpha = mean(|w| where sign!=0) per row -- i.e. the same alpha BCQ computes,
just applied to a 3-level (with true zero) code instead of BCQ's 2-level code.
"""
import sys
sys.path.insert(0, "/home/claude/option_d")
import torch
import numpy as np
from bcq_cpu import quantize as bcq_quantize

def ternary_ptq_raw(w):
    t = 0.7 * w.abs().mean()
    return torch.where(w > t, torch.ones_like(w),
           torch.where(w < -t, -torch.ones_like(w), torch.zeros_like(w)))

def ternary_ptq_scaled(w):
    t = 0.7 * w.abs().mean()
    sign = torch.where(w > t, torch.ones_like(w),
           torch.where(w < -t, -torch.ones_like(w), torch.zeros_like(w)))
    # per-row alpha, computed only over the nonzero (kept) entries -- same spirit as BCQ's alpha
    alpha = torch.zeros(w.shape[0], 1)
    for i in range(w.shape[0]):
        nz = sign[i] != 0
        alpha[i] = w[i][nz].abs().mean() if nz.any() else 0.0
    return sign * alpha

def mse(a, b):
    return ((a - b) ** 2).mean().item()

torch.manual_seed(0)
shape = (384, 384)
std = (2.0 / shape[1]) ** 0.5

raw_errs, scaled_errs, bcq1_errs = [], [], []
for seed in range(5):
    torch.manual_seed(seed)
    w = torch.randn(shape) * std
    raw_errs.append(mse(w, ternary_ptq_raw(w)))
    scaled_errs.append(mse(w, ternary_ptq_scaled(w)))
    ret, B, alpha, mask = bcq_quantize(w, qbits=1, rounds=15, group_size=-1)
    bcq1_errs.append(mse(w, ret.cpu()))

print(f"{'Method':<28} {'Mean MSE':>12} {'Worst seed':>12}")
print("-" * 54)
print(f"{'Ternary PTQ (raw, no scale)':<28} {np.mean(raw_errs):>12.5f} {max(raw_errs):>12.5f}")
print(f"{'Ternary, scaled (added alpha)':<28} {np.mean(scaled_errs):>12.5f} {max(scaled_errs):>12.5f}")
print(f"{'BCQ 1-bit (real ShiftAddLLM)':<28} {np.mean(bcq1_errs):>12.5f} {max(bcq1_errs):>12.5f}")
print()
gap_before = np.mean(raw_errs) / np.mean(bcq1_errs)
gap_after = np.mean(scaled_errs) / np.mean(bcq1_errs)
print(f"Raw ternary vs BCQ 1-bit gap:    {gap_before:.1f}x")
print(f"Scaled ternary vs BCQ 1-bit gap: {gap_after:.1f}x")
