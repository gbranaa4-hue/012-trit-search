"""
Option D (real out-of-sample version) -- generates triplets from your OWN
local codebase instead of live GitHub or the handwritten BENCHMARK list,
using extract_pairs_from_code() from trit_embed_train.py (the same function
that built your 80k-pair QAT training set -- not reimplemented here).

Why this instead of chasing the datasets/codeparrot fix: it's offline, has
no dependency on an external library's script-loading format, and -- more
importantly -- it's guaranteed to be genuinely out-of-sample, since none of
this repo's own trit_*.py source was in the handwritten 20/25-pair
BENCHMARK (those are short hand-written game/ML snippets, this is real
source code).

How triplets are built: for each (anchor, positive) pair extracted from
your source files, the "wrong" answer is the positive from a DIFFERENT,
randomly chosen pair -- i.e. a real, different function, not a shuffled
version of the same one. This is a harder, more faithful negative than
stream_github_test's shuffled-lines approach, and closer in spirit to the
hand-crafted BENCHMARK's design (same general domain, different behavior).

Pre-registered before running:
  CONFIRM   the ranking (raw < bcq_1bit < scaled < bcq_2bit <= bcq_3bit)
            replicates on this genuinely independent, much larger set.
  DISCONFIRM  bcq_3bit's "beats float32" result washes out to a small
            real deficit (expected -- this is the ceiling-effect
            explanation being confirmed, not a failed test) OR the
            whole ranking reshuffles (would mean the earlier finding
            doesn't generalize beyond hand-picked game-code snippets).
"""
import copy, glob, random, re, sys
sys.path.insert(0, ".")
import torch
import torch.nn as nn

import trit_benchmark as tb
from trit_embed_train import extract_pairs_from_code
from bcq_cpu import quantize as bcq_quantize

MODEL_PATH = "models/code-minilm"
N_TRIPLETS = 150
random.seed(0)


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


def build_local_triplets(n_target=N_TRIPLETS):
    """Extract (anchor, positive) pairs from this repo's own .py source,
    then build (query, correct, wrong) triplets with real different-function
    negatives."""
    all_pairs = []
    seen_anchors = set()
    py_files = glob.glob("*.py") + glob.glob("hardware/**/*.py", recursive=True)
    for path in py_files:
        if path == __file__ or "test" in path.lower():
            # skip test files (would mostly extract test-assertion boilerplate,
            # not representative "what does this function do" pairs) and skip self
            continue
        try:
            code = open(path, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for anchor, positive in extract_pairs_from_code(code, "python"):
            if len(positive) < 60 or len(positive) > 800:
                continue
            key = (anchor, positive[:40])
            if key in seen_anchors:
                continue
            seen_anchors.add(key)
            all_pairs.append((anchor, positive))

    random.shuffle(all_pairs)
    if len(all_pairs) < 20:
        raise RuntimeError(f"Only extracted {len(all_pairs)} pairs -- not enough .py source "
                            f"to build a reliable triplet set. Point py_files at a bigger "
                            f"local codebase if you have one checked out alongside this repo.")

    n = min(n_target, len(all_pairs))
    chosen = all_pairs[:n]
    triplets = []
    for i, (anchor, positive) in enumerate(chosen):
        # pick a wrong answer from a different, randomly chosen pair
        j = random.choice([k for k in range(len(all_pairs)) if k != i])
        wrong = all_pairs[j][1]
        if wrong == positive:
            continue
        triplets.append((anchor, positive, wrong))
    return triplets


def main():
    base = tb.load_model(MODEL_PATH)
    base.eval()

    print("Extracting triplets from local .py source (not the handwritten BENCHMARK)...")
    triplets = build_local_triplets()
    print(f"Built {len(triplets)} genuinely out-of-sample triplets "
          f"(1 triplet ~= {100/len(triplets):.2f}pp resolution).\n")

    base_report = tb.run_benchmark(base, "float32", triplets)
    tb.print_report(base_report)

    variants = [
        ("ternary_raw",    lambda m: apply_fn(m, ternary_ptq_raw)),
        ("ternary_scaled", lambda m: apply_fn(m, ternary_ptq_scaled)),
        ("bcq_1bit",       lambda m: apply_bcq(m, 1)),
        ("bcq_2bit",       lambda m: apply_bcq(m, 2)),
        ("bcq_3bit",       lambda m: apply_bcq(m, 3)),
    ]

    print(f"\nSummary (n = {base_report['n']} local out-of-sample triplets):")
    print(f"  {'Method':<18} {'Accuracy':>10} {'Drop vs float32':>18}")
    print(f"  {'float32':<18} {base_report['accuracy']:>9.1f}% {'--':>18}")

    for name, fn in variants:
        m = copy.deepcopy(base)
        m.eval()
        fn(m)
        report = tb.run_benchmark(m, name, triplets)
        drop = base_report["accuracy"] - report["accuracy"]
        print(f"  {name:<18} {report['accuracy']:>9.1f}% {drop:>+17.1f}pp")


if __name__ == "__main__":
    main()
