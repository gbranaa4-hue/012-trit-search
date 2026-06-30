"""
012 Resonant Consensus-Gate Cache vs LRU — Noisy Access Pattern Test

Tests the specific hypothesis from the resonating-cell proposal:
"LRU degrades under noisy access patterns because noise corrupts its
recency tracking; a resonating-cell low-pass filter in front of the
consensus gate should be more robust because it smooths out one-off
noise accesses instead of treating every access as equally meaningful."

Noise model: interleave the real Zipfian trace with random "decoy"
accesses to items outside the normal working set — single, isolated
touches that pollute LRU's raw recency ordering but should mostly
average out under a low-pass filter.

This is an honest test — built to find out if the hypothesis is true,
not to make either side win.

Usage:
  python trit_resonant_cache.py
"""
import random
from collections import OrderedDict

random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# WORKLOAD — Zipfian core + injected noise accesses
# ══════════════════════════════════════════════════════════════════════════════

def zipfian_trace(n_items=200, n_accesses=20000, skew=1.2):
    weights = [1.0 / (rank ** skew) for rank in range(1, n_items + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(list(range(n_items)), weights=probs, k=n_accesses)

def inject_noise(trace, noise_rate, n_items, noise_pool_offset=10000):
    """Replace `noise_rate` fraction of accesses with one-off decoy items
    (IDs outside the normal working set) — simulates random/transient
    access spikes that don't reflect real future utility."""
    noisy = []
    decoy_id = noise_pool_offset
    for x in trace:
        if random.random() < noise_rate:
            noisy.append(decoy_id)
            decoy_id += 1  # every decoy is unique — never reaccessed
        else:
            noisy.append(x)
    return noisy

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE: standard LRU
# ══════════════════════════════════════════════════════════════════════════════

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = True

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total * 100 if total else 0

# ══════════════════════════════════════════════════════════════════════════════
# CONSENSUS-GATE (no resonance — same as trit_cache_eviction.py, for
# 3-way comparison: LRU vs plain-consensus vs resonant-consensus)
# ══════════════════════════════════════════════════════════════════════════════

def consensus(v0, v1, v2):
    total = v0 + v1 + v2
    return 1 if total > 0 else (-1 if total < 0 else 0)

class ConsensusCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.tick = 0
        self.hits = 0
        self.misses = 0

    def _score(self, meta):
        gap = self.tick - meta["last_access"]
        v_recency = -1 if gap > self.capacity else (1 if gap <= 1 else 0)
        v_freq    = -1 if meta["count"] <= 1 else (1 if meta["count"] >= 4 else 0)
        v_rand    = random.choice([-1, 0, 1])
        return consensus(v_recency, v_freq, v_rand)

    def access(self, key):
        self.tick += 1
        if key in self.cache:
            self.cache[key]["last_access"] = self.tick
            self.cache[key]["count"] += 1
            self.hits += 1
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                worst = min(self.cache, key=lambda k: self._score(self.cache[k]))
                del self.cache[worst]
            self.cache[key] = {"last_access": self.tick, "count": 1}

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total * 100 if total else 0

# ══════════════════════════════════════════════════════════════════════════════
# RESONANT CONSENSUS-GATE
# Per-item low-pass filtered "freshness" replaces raw last-access-tick.
# A single isolated (noise) access bumps the filter only slightly;
# sustained real access builds the filtered value up over multiple hits.
# ══════════════════════════════════════════════════════════════════════════════

class ResonantConsensusCache:
    def __init__(self, capacity, tau=4.0):
        self.capacity = capacity
        self.tau = tau
        self.cache = {}  # key -> {"lp": float, "count": int}
        self.tick = 0
        self.hits = 0
        self.misses = 0

    def _decay_all(self):
        # Every cached item's filtered freshness decays one tick toward 0,
        # whether accessed or not (the resonator's low-pass behavior).
        for meta in self.cache.values():
            meta["lp"] += (0.0 - meta["lp"]) / self.tau

    def _score(self, meta):
        v_recency = 1 if meta["lp"] > 0.6 else (-1 if meta["lp"] < 0.15 else 0)
        v_freq    = -1 if meta["count"] <= 1 else (1 if meta["count"] >= 4 else 0)
        v_rand    = random.choice([-1, 0, 1])
        return consensus(v_recency, v_freq, v_rand)

    def access(self, key):
        self.tick += 1
        self._decay_all()
        if key in self.cache:
            meta = self.cache[key]
            meta["lp"] += (1.0 - meta["lp"]) / self.tau  # low-pass step toward 1 on hit
            meta["count"] += 1
            self.hits += 1
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                worst = min(self.cache, key=lambda k: self._score(self.cache[k]))
                del self.cache[worst]
            self.cache[key] = {"lp": 1.0 / self.tau, "count": 1}

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total * 100 if total else 0

# ══════════════════════════════════════════════════════════════════════════════
# RUN COMPARISON ACROSS NOISE LEVELS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("  LRU vs Consensus-Gate vs Resonant-Consensus — Noisy Trace Test")
    print("="*70)

    capacity = 20
    for noise_rate in [0.0, 0.1, 0.25, 0.4]:
        random.seed(42)
        base_trace = zipfian_trace(n_items=200, n_accesses=20000, skew=1.2)
        trace = inject_noise(base_trace, noise_rate, n_items=200) if noise_rate > 0 else base_trace

        random.seed(123)
        lru = LRUCache(capacity)
        for k in trace: lru.access(k)

        random.seed(123)
        cg = ConsensusCache(capacity)
        for k in trace: cg.access(k)

        random.seed(123)
        rc = ResonantConsensusCache(capacity)
        for k in trace: rc.access(k)

        print(f"\n  Noise rate = {noise_rate*100:.0f}%  (cache capacity={capacity})")
        print(f"    LRU                 : {lru.hit_rate():.2f}%")
        print(f"    Consensus-gate      : {cg.hit_rate():.2f}%  ({cg.hit_rate()-lru.hit_rate():+.2f}pp vs LRU)")
        print(f"    Resonant-consensus  : {rc.hit_rate():.2f}%  ({rc.hit_rate()-lru.hit_rate():+.2f}pp vs LRU)")

    print("\n" + "="*70)
    print("  Honest result — testing whether resonance helps under noise,")
    print("  not engineered for any side to win.")
    print("="*70)
