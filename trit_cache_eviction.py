"""
012 Consensus-Gate Cache Eviction vs Standard LRU

A real head-to-head test of the consensus-gate primitive (sign(v0+v1+v2),
the same math as hardware/consensus_gate.sv and trit_os_sim.py's scheduler)
against the established LRU cache eviction algorithm.

This is NOT engineered to make ternary win — it's an honest comparison
on a realistic access pattern (Zipfian distribution, the standard way
to simulate real-world cache workloads where some items are accessed
far more often than others).

Metric: hit rate % over N accesses, fixed cache capacity.

Usage:
  python trit_cache_eviction.py
"""

import random
from collections import OrderedDict

random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# WORKLOAD — Zipfian access pattern (realistic: 20% of items get 80% of accesses)
# ══════════════════════════════════════════════════════════════════════════════

def zipfian_trace(n_items=200, n_accesses=20000, skew=1.2):
    weights = [1.0 / (rank ** skew) for rank in range(1, n_items + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]
    items = list(range(n_items))
    return random.choices(items, weights=probs, k=n_accesses)

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
                self.cache.popitem(last=False)  # evict least recently used
            self.cache[key] = True

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total * 100 if total else 0

# ══════════════════════════════════════════════════════════════════════════════
# CONSENSUS-GATE EVICTION
# Each cached item is scored by 3 trit voters on every eviction:
#   recency  : -1 stale (not used recently), +1 fresh, 0 mid
#   frequency: -1 rarely used, +1 frequently used, 0 mid
#   random   : small tiebreak noise trit, prevents deterministic pathology
# Evict the item with the LOWEST consensus score (most "vote to evict").
# ══════════════════════════════════════════════════════════════════════════════

def consensus(v0, v1, v2):
    total = v0 + v1 + v2
    return 1 if total > 0 else (-1 if total < 0 else 0)

class ConsensusCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}       # key -> dict(last_access, count)
        self.tick = 0
        self.hits = 0
        self.misses = 0

    def _score(self, key, meta):
        recency_gap = self.tick - meta["last_access"]
        v_recency = -1 if recency_gap > self.capacity else (1 if recency_gap <= 1 else 0)
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
                # Evict the lowest-scoring item (most votes to evict)
                worst_key = min(self.cache, key=lambda k: self._score(k, self.cache[k]))
                del self.cache[worst_key]
            self.cache[key] = {"last_access": self.tick, "count": 1}

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total * 100 if total else 0

# ══════════════════════════════════════════════════════════════════════════════
# RUN COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*65)
    print("  Consensus-Gate Eviction vs LRU — Real Head-to-Head Test")
    print("="*65)

    for capacity in [10, 20, 50]:
        trace = zipfian_trace(n_items=200, n_accesses=20000, skew=1.2)

        lru = LRUCache(capacity)
        for k in trace:
            lru.access(k)

        cg = ConsensusCache(capacity)
        for k in trace:
            cg.access(k)

        lru_rate = lru.hit_rate()
        cg_rate  = cg.hit_rate()
        diff = cg_rate - lru_rate

        print(f"\n  Cache capacity = {capacity} (of 200 possible items, Zipfian access)")
        print(f"    LRU hit rate            : {lru_rate:.2f}%")
        print(f"    Consensus-gate hit rate : {cg_rate:.2f}%")
        print(f"    Difference              : {diff:+.2f}pp "
              f"({'consensus-gate wins' if diff > 0 else 'LRU wins' if diff < 0 else 'tie'})")

    print("\n" + "="*65)
    print("  This is an honest result — not engineered for either side to win.")
    print("="*65)
