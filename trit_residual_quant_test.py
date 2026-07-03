"""
Residual ("keep the shadow") ternary quantization -- reconstruction test.

The idea: a single trit throws away everything that didn't fit into
{-1,0,+1}. That discarded remainder is "the shadow the collapse casts off."
Residual quantization keeps it by encoding the leftover in a SECOND trit at
a smaller scale:  w  ≈  a1*t1  +  a2*t2,  with t1,t2 ∈ {-1,0,+1}, a2 < a1.

This measures the CHEAP, foundational question directly -- does keeping the
shadow actually reduce reconstruction error? -- without training a whole
model. If 2-level residual doesn't beat 1-level here, it won't help
downstream and there's no reason to spend a multi-hour training A/B on it.
Honest tradeoff being measured: error reduction vs. 2x storage (two trits
+ two scales per weight instead of one).

For a fair comparison BOTH levels use the optimal least-squares scale
(a = <w,t>/<t,t>), not a bare {-1,0,+1} with no magnitude -- otherwise the
single-level baseline is handicapped and the residual looks better than it
is for the wrong reason.
"""
import torch


def ternary(x, thresh_frac=0.7):
    """Same deadzone rule as TernaryQuantize in trit_triadic_encoder.py:
    threshold = 0.7 * mean(|x|), then sign with a dead zone toward 0."""
    t = thresh_frac * x.abs().mean()
    return torch.where(x > t, torch.ones_like(x),
           torch.where(x < -t, -torch.ones_like(x), torch.zeros_like(x)))


def optimal_scale(w, t):
    """Least-squares scale a minimizing ||w - a*t||: a = <w,t>/<t,t>."""
    denom = (t * t).sum()
    if denom == 0:
        return torch.tensor(0.0)
    return (w * t).sum() / denom


def quantize_1level(w):
    t1 = ternary(w)
    a1 = optimal_scale(w, t1)
    return a1 * t1, {"trits": 1}


def quantize_2level_residual(w):
    t1 = ternary(w)
    a1 = optimal_scale(w, t1)
    recon1 = a1 * t1
    resid = w - recon1                # <-- the shadow: what the first trit discarded
    t2 = ternary(resid)
    a2 = optimal_scale(resid, t2)
    recon2 = recon1 + a2 * t2
    return recon2, {"trits": 2}


def rel_error(w, recon):
    return (w - recon).pow(2).sum().sqrt() / w.pow(2).sum().sqrt()


def sparsity(t):
    return (t == 0).float().mean().item()


def run_case(name, w):
    r1, _ = quantize_1level(w)
    r2, _ = quantize_2level_residual(w)
    e1 = rel_error(w, r1).item()
    e2 = rel_error(w, r2).item()
    reduction = (e1 - e2) / e1 * 100 if e1 > 0 else 0.0
    print(f"{name:28s}  1-level rel_err={e1:.4f}   "
          f"2-level rel_err={e2:.4f}   error reduced {reduction:5.1f}%")
    return e1, e2


def main():
    torch.manual_seed(0)
    print("Reconstruction error: 1 trit (+scale) vs 2 trits (residual/'shadow')")
    print("Lower rel_err = closer to the real weights. 2-level costs 2x storage.\n")

    # Weight distributions representative of real neural net weights
    cases = [
        ("gaussian (typical init)", torch.randn(4096)),
        ("gaussian narrow",         torch.randn(4096) * 0.1),
        ("heavy-tailed (t-dist)",   torch.randn(4096) / torch.randn(4096).abs().clamp(min=0.3)),
        ("uniform",                 (torch.rand(4096) - 0.5) * 2),
        ("sparse (80% near zero)",  torch.randn(4096) * (torch.rand(4096) > 0.8).float()),
        ("2D weight matrix",        torch.randn(512, 512).flatten()),
    ]

    e1s, e2s = [], []
    for name, w in cases:
        e1, e2 = run_case(name, w)
        e1s.append(e1); e2s.append(e2)

    import statistics
    print(f"\nMean 1-level rel_err: {statistics.mean(e1s):.4f}")
    print(f"Mean 2-level rel_err: {statistics.mean(e2s):.4f}")
    avg_red = statistics.mean([(a-b)/a*100 for a, b in zip(e1s, e2s)])
    print(f"Mean error reduction from keeping the shadow: {avg_red:.1f}%  (at 2x storage cost)")

    # Honest reference: what does a plain float16 (also ~half precision-ish
    # storage relative to float32) reconstruction error look like? ~0. The
    # point of ternary isn't beating float -- it's how close you get at
    # ~1.6 bits/weight. This just frames whether the 2nd trit is worth it.
    print("\nInterpretation guide:")
    print("  >30% reduction -> shadow clearly worth a training A/B")
    print("  10-30%        -> marginal; depends if the task is accuracy-starved")
    print("  <10%          -> not worth 2x storage; skip it, save the training run")


if __name__ == "__main__":
    main()
