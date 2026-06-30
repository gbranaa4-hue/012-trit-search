"""
012 Resonant Memory Retrieval Test

Tests the hypothesis: "A resonating cell front-end, integrating multiple
noisy observations of the same fact over time, should produce cleaner
input to the Hopfield memory than using a single noisy observation or a
simple instant average."

Uses the real TernaryHopfield class from trit_memory_store.py. Skips the
neural FactEncoder (500-epoch training) and uses synthetic random trit
patterns as stand-in "facts" — tests the actual retrieval mechanism
without the extra training time.

Setup: store 30 random trit patterns. Pick one target fact. Simulate K
noisy "glimpses" of it (independent corruptions at a given noise rate,
like K independent corrupted readings of the same underlying signal).
Compare three ways of using those K glimpses before Hopfield retrieval:

  A) Single-shot   — use only the last glimpse, ignore the rest
  B) Instant vote  — simple elementwise majority vote across all K glimpses
  C) Resonant      — low-pass filter (EMA) each dimension across the K
                     glimpses, arriving sequentially, then quantize

Metric: does retrieval converge to the correct stored pattern (checked
via nearest_label) more often under each strategy, across noise levels?

This is an honest test — not engineered for any side to win.

Usage:
  python trit_resonant_memory.py
"""
import random
import torch
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# TernaryHopfield — copied directly from trit_memory_store.py (importing that
# file executes its full module-level demo script, which has no __main__
# guard — a real bug there, worth fixing separately).
# ══════════════════════════════════════════════════════════════════════════════

class TernaryHopfield:
    def __init__(self, trit_dim=256):
        self.trit_dim = trit_dim
        self.W = torch.zeros(trit_dim, trit_dim, device=device)
        self.patterns = []
        self.labels = []
        self.n_stored = 0

    def store(self, pattern, label=""):
        p = pattern.float().to(device)
        self.W += torch.outer(p, p)
        self.W.fill_diagonal_(0)
        self.patterns.append(p.clone())
        self.labels.append(label)
        self.n_stored += 1

    def store_batch(self, patterns, labels):
        for p, l in zip(patterns, labels):
            self.store(p, l)
        self.W /= max(self.n_stored, 1)

    def retrieve(self, query, steps=10):
        x = query.float().to(device)
        for _ in range(steps):
            x_new = torch.sign(self.W @ x)
            if torch.allclose(x_new, x): break
            x = x_new
        return x

    def nearest_label(self, retrieved):
        if not self.patterns:
            return "empty memory", 0.0
        retrieved = retrieved.to(device)
        best_sim, best_idx = -float('inf'), 0
        for i, p in enumerate(self.patterns):
            sim = F.cosine_similarity(retrieved.unsqueeze(0), p.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return self.labels[best_idx], best_sim
TRIT_DIM = 256
N_FACTS  = 30
K_GLIMPSES = 6   # number of noisy observations of the target fact

# ══════════════════════════════════════════════════════════════════════════════
# BUILD MEMORY — random sparse trit patterns as stand-in facts
# ══════════════════════════════════════════════════════════════════════════════

def random_trit_pattern(dim, sparsity=0.5):
    """Random pattern with `sparsity` fraction zeros, rest split -1/+1."""
    p = torch.zeros(dim)
    n_active = int(dim * (1 - sparsity))
    active_idx = torch.randperm(dim)[:n_active]
    signs = torch.randint(0, 2, (n_active,)) * 2 - 1
    p[active_idx] = signs.float()
    return p

memory = TernaryHopfield(TRIT_DIM)
patterns, labels = [], []
for i in range(N_FACTS):
    p = random_trit_pattern(TRIT_DIM)
    patterns.append(p)
    labels.append(f"fact_{i}")
memory.store_batch(patterns, labels)

TARGET_IDX = 5
target_pattern = patterns[TARGET_IDX]
target_label = labels[TARGET_IDX]

def corrupt(pattern, rate):
    mask = torch.rand(pattern.shape) < rate
    noise = (torch.randint(0, 3, pattern.shape).float() - 1)  # random -1/0/+1
    return torch.where(mask, noise, pattern)

# ══════════════════════════════════════════════════════════════════════════════
# THREE STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def strategy_single_shot(glimpses):
    return glimpses[-1]

def strategy_instant_vote(glimpses):
    stacked = torch.stack(glimpses)  # (K, dim)
    summed = stacked.sum(dim=0)
    return torch.sign(summed)

def strategy_resonant(glimpses, tau=2.5):
    lp = torch.zeros(TRIT_DIM)
    for g in glimpses:
        lp += (g - lp) / tau
    t = 0.3 * lp.abs().mean()
    return torch.where(lp > t, torch.ones_like(lp),
           torch.where(lp < -t, -torch.ones_like(lp), torch.zeros_like(lp)))

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print(f"  Resonant Memory Retrieval Test — target: '{target_label}'")
    print(f"  {N_FACTS} stored facts, {K_GLIMPSES} noisy glimpses per trial")
    print("="*70)

    noise_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    n_trials = 50

    results = {"single": {}, "vote": {}, "resonant": {}}

    for rate in noise_rates:
        correct = {"single": 0, "vote": 0, "resonant": 0}
        for trial in range(n_trials):
            glimpses = [corrupt(target_pattern, rate) for _ in range(K_GLIMPSES)]

            for name, strat in [("single", strategy_single_shot),
                                 ("vote", strategy_instant_vote),
                                 ("resonant", strategy_resonant)]:
                cleaned = strat(glimpses)
                retrieved = memory.retrieve(cleaned)
                key, sim = memory.nearest_label(retrieved)
                if key == target_label:
                    correct[name] += 1

        for name in results:
            results[name][rate] = correct[name] / n_trials * 100

    print(f"\n  {'Noise':>6} | {'Single-shot':>12} | {'Instant-vote':>13} | {'Resonant':>9}")
    print(f"  {'-'*50}")
    for rate in noise_rates:
        print(f"  {rate*100:>5.0f}% | {results['single'][rate]:>11.1f}% | "
              f"{results['vote'][rate]:>12.1f}% | {results['resonant'][rate]:>8.1f}%")

    print("\n" + "="*70)
    print("  Honest result — testing whether resonance improves noisy-glimpse")
    print("  memory retrieval vs single-shot or instant-vote, not engineered to win.")
    print("="*70)
