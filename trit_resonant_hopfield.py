"""
012 Resonant Hopfield Memory Test (Corrected Hypothesis)

Previous test (trit_resonant_memory.py) found instant-vote beat resonance
because it averaged all glimpses upfront, then did one Hopfield retrieve —
the resonator was just a worse averaging function for a static signal.

Corrected hypothesis being tested here: resonance should be INSIDE the
memory's settling process, not a pre-filter on the input. At each step,
blend a new noisy glimpse into persistent internal state via slow decay,
THEN apply one Hopfield energy-minimization update — interleaved over
multiple steps, not averaged once upfront.

Three approaches compared, all given the same noisy glimpses:
  A) Standard       — one glimpse, standard multi-step Hopfield retrieve()
  B) Instant-vote    — average all glimpses upfront, then one retrieve()
                       (carried over from trit_resonant_memory.py — the
                       previous winner, the bar to beat)
  C) Resonant-settle — interleave EMA blending with Hopfield update,
                       one glimpse injected per external step

This is an honest test — not engineered for any side to win.

Usage:
  python trit_resonant_hopfield.py
"""
import random
import torch
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRIT_DIM = 256
N_FACTS = 30
K_GLIMPSES = 6

# ══════════════════════════════════════════════════════════════════════════════
# TernaryHopfield (same as trit_resonant_memory.py)
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

    def settle_step(self, x):
        """One Hopfield energy-minimization step, no convergence loop."""
        return torch.sign(self.W @ x)

    def retrieve(self, query, steps=10):
        x = query.float().to(device)
        for _ in range(steps):
            x_new = self.settle_step(x)
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

# ══════════════════════════════════════════════════════════════════════════════
# SETUP — same random pattern store as before, same target
# ══════════════════════════════════════════════════════════════════════════════

def random_trit_pattern(dim, sparsity=0.5):
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
    noise = (torch.randint(0, 3, pattern.shape).float() - 1)
    return torch.where(mask, noise, pattern)

# ══════════════════════════════════════════════════════════════════════════════
# THREE APPROACHES
# ══════════════════════════════════════════════════════════════════════════════

def approach_standard(glimpses):
    """A) One glimpse (the last), standard multi-step Hopfield retrieve."""
    return memory.retrieve(glimpses[-1])

def approach_instant_vote(glimpses):
    """B) Average all glimpses upfront, then one retrieve — previous winner."""
    stacked = torch.stack(glimpses)
    voted = torch.sign(stacked.sum(dim=0))
    return memory.retrieve(voted)

def approach_resonant_settle(glimpses, decay=0.7):
    """C) Interleave EMA blending with Hopfield settling, one glimpse per step."""
    state = torch.zeros(TRIT_DIM, device=device)
    for g in glimpses:
        state = decay * state + (1 - decay) * g.to(device)
        state = memory.settle_step(state)
    # A few extra pure-settling steps after all glimpses are in
    for _ in range(5):
        new_state = memory.settle_step(state)
        if torch.allclose(new_state, state): break
        state = new_state
    return state

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print(f"  Resonant-Settle Hopfield Test — target: '{target_label}'")
    print(f"  {N_FACTS} stored facts, {K_GLIMPSES} noisy glimpses per trial")
    print("="*70)

    noise_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    n_trials = 50
    results = {"standard": {}, "vote": {}, "resonant_settle": {}}

    for rate in noise_rates:
        correct = {"standard": 0, "vote": 0, "resonant_settle": 0}
        for trial in range(n_trials):
            glimpses = [corrupt(target_pattern, rate) for _ in range(K_GLIMPSES)]

            for name, fn in [("standard", approach_standard),
                              ("vote", approach_instant_vote),
                              ("resonant_settle", approach_resonant_settle)]:
                retrieved = fn(glimpses)
                key, sim = memory.nearest_label(retrieved)
                if key == target_label:
                    correct[name] += 1

        for name in results:
            results[name][rate] = correct[name] / n_trials * 100

    print(f"\n  {'Noise':>6} | {'Standard':>9} | {'Instant-vote':>13} | {'Resonant-settle':>16}")
    print(f"  {'-'*55}")
    for rate in noise_rates:
        print(f"  {rate*100:>5.0f}% | {results['standard'][rate]:>8.1f}% | "
              f"{results['vote'][rate]:>12.1f}% | {results['resonant_settle'][rate]:>15.1f}%")

    print("\n" + "="*70)
    print("  Honest result — testing the corrected hypothesis (resonance as")
    print("  internal settling dynamics, not a pre-filter), not engineered to win.")
    print("="*70)
