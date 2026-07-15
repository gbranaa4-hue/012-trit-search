"""
Option D (bigger-N version) -- same quantization methods (raw ternary, scaled
ternary, BCQ 1/2/3-bit), but evaluated against a larger pair set instead of
the 20-triplet quick test, to get enough resolution to actually trust a
result like "BCQ 3-bit >= float32."

Reuses trit_benchmark.py's own run_benchmark() function directly (imported,
not reimplemented) so the accuracy computation is identical to your existing
numbers -- only the pair source and the model's weights change.

Pair source, in order of preference:
  1. --stream mode's live GitHub extraction (~100 pairs, 1pp resolution)
     -- requires `pip install datasets` and network access; pairs are
     pulled fresh each run, so treat this as "does the ranking replicate
     on an independent sample," not as a frozen benchmark number.
  2. Falls back to the full BENCHMARK list (~25 pairs, ~4pp resolution)
     if datasets isn't installed or the stream fails -- still better
     than the 20-triplet subset, but flag this in any write-up: it's a
     modest improvement in resolution, not a large one.

IMPORTANT: whichever pair set is used, it is extracted ONCE per run and
reused across all quantization variants below, so comparisons *within*
one run are apples-to-apples even if the live-pulled pairs differ
between separate runs on different days.

Pre-registered before running:
  CONFIRM   the ranking from the 20-triplet run replicates: raw ternary
            worst, BCQ-1bit < scaled ternary, BCQ-2bit and BCQ-3bit
            recovering most/all of the gap to INT8.
  DISCONFIRM  BCQ-3bit's apparent tie-or-beat vs float32 washes out into
            a real, visible deficit once N is large enough to resolve it
            (expected and fine -- that's the ceiling-effect explanation
            being confirmed, not a failure of this test).
"""
import copy, sys
sys.path.insert(0, ".")
import torch
import torch.nn as nn

import trit_benchmark as tb   # reuse their real run_benchmark() and BENCHMARK
from bcq_cpu import quantize as bcq_quantize

MODEL_PATH = "models/code-minilm"


def ternary_ptq_raw(w):
    t = 0.7 * w.abs().mean()
    return torch.where(w > t, torch.ones_like(w),
           torch.where(w < -t, -torch.ones_like(w), torch.zeros_like(w)))

def ternary_ptq_scaled(w):
    sign = ternary_ptq_raw(w)
    alpha = torch.zeros(w.shape[0], 1, device=w.device, dtype=w.dtype)
    for i in range(w.shape[0]):
        nz = sign[i] != 0
        alpha[i] = w[i][nz].abs().mean() if nz.any() else 0.0
    return sign * alpha

def apply_fn(model, fn):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.data.copy_(fn(module.weight.data))

def apply_bcq(model, qbits):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                w = module.weight.data.float()
                ret, B, alpha, mask = bcq_quantize(w, qbits=qbits, rounds=15, group_size=-1)
                module.weight.data.copy_(ret.to(module.weight.dtype))


def get_pairs(model):
    """Try live GitHub stream first (~100 pairs), fall back to full BENCHMARK (~25)."""
    try:
        from datasets import load_dataset
        from trit_embed_train import extract_pairs_from_code
        import random
        pairs = []
        ds = load_dataset("codeparrot/github-code", streaming=True, split="train")
        for i, item in enumerate(ds):
            if i >= 1000:
                break
            code = item.get("content", "")
            if len(code) < 300:
                continue
            extracted = extract_pairs_from_code(code, "unknown")
            if extracted:
                anchor, positive = extracted[0]
                lines = positive.splitlines()
                random.shuffle(lines)
                wrong = "\n".join(lines[:5])
                if len(wrong) > 20:
                    pairs.append((anchor, positive, wrong))
            if len(pairs) >= 100:
                break
        if len(pairs) >= 50:
            print(f"Using {len(pairs)} live GitHub pairs (1 pair ~= {100/len(pairs):.1f}pp resolution).\n")
            return pairs
        print(f"Only got {len(pairs)} live pairs, falling back to full BENCHMARK.\n")
    except Exception as e:
        print(f"Live stream unavailable ({e}), falling back to full BENCHMARK.\n")

    pairs = tb.BENCHMARK
    print(f"Using full BENCHMARK: {len(pairs)} pairs (1 pair ~= {100/len(pairs):.1f}pp resolution).\n")
    return pairs


def main():
    base = tb.load_model(MODEL_PATH)
    base.eval()

    pairs = get_pairs(base)  # frozen for this run, reused for every variant below

    base_report = tb.run_benchmark(base, "float32", pairs)
    tb.print_report(base_report)

    variants = [
        ("ternary_raw",    lambda m: apply_fn(m, ternary_ptq_raw)),
        ("ternary_scaled", lambda m: apply_fn(m, ternary_ptq_scaled)),
        ("bcq_1bit",       lambda m: apply_bcq(m, 1)),
        ("bcq_2bit",       lambda m: apply_bcq(m, 2)),
        ("bcq_3bit",       lambda m: apply_bcq(m, 3)),
    ]

    print("\nSummary (n =", base_report["n"], "pairs):")
    print(f"  {'Method':<18} {'Accuracy':>10} {'Drop vs float32':>18}")
    print(f"  {'float32':<18} {base_report['accuracy']:>9.1f}% {'--':>18}")

    for name, fn in variants:
        m = copy.deepcopy(base)
        m.eval()
        fn(m)
        report = tb.run_benchmark(m, name, pairs)
        drop = base_report["accuracy"] - report["accuracy"]
        print(f"  {name:<18} {report['accuracy']:>9.1f}% {drop:>+17.1f}pp")


if __name__ == "__main__":
    main()
