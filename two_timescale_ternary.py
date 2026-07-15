#!/usr/bin/env python3
"""Two-timescale ternary: is a SLOW structural gate + FAST sign layer non-parasitic?

Ternary weight  W = alpha * G * S   with   G in {0,1} (TOPOLOGY, which edges live)
                                           S in {-1,+1} (POLARITY, excite/inhibit).
  S updates every step (fast).  G updates every K steps (slow, the timescale knob).
  A gated-off edge (G_i=0) gets NO sign gradient -> its sign shadow is FROZEN.
  So the gate can act as a STRUCTURAL MEMORY that protects the polarity of edges
  that are currently disconnected, ready for when they are rewired back in.

Environment: R fixed regimes cycle every T_REGIME steps. Each regime uses a sparse
SIGNED support drawn from a shared feature pool, so features RECUR across regimes
-> whether a protected sign SURVIVES a rewire is directly measurable. Off-support
features are pure noise the gate must NOT connect (the confound). The target needs
the SIGNS right (y = sum of s*_i x_i over the live support), so polarity matters.

The timescale knob K controls the gate's sample size AND cadence:
  every K steps the gate connects the top-K_SUPPORT features by |sum_{last K} x_i*y|.
  small K  -> gate decides from few noisy samples, churns the topology, signs never
              settle before being gated off  (PARASITIC / aliased; K=1 = no separation)
  medium K -> clean support, signs converge on a held-still topology, freeze correct
              (NON-PARASITIC)
  big  K   -> the K-window spans several regimes and blurs the support (degrades)

FALSIFICATION control -- mode:
  protect : keep the sign shadow when an edge is disconnected (structural memory)
  erase   : zero the sign shadow on disconnect (no memory)
  If 'protect' does NOT beat 'erase', the "gate protects polarity" story is FALSE.

PRE-REGISTERED before running:
  CONFIRM  a WINDOW in K exists where transfer >> 0.5 and tail-error is low; the
           lower edge is the timescale-separation boundary K*. protect < erase inside
           the window. Both beat K=1 (joint / no separation).
  DISCONFIRM  transfer ~0.5 for all K, OR error is flat/monotonic in K, OR
              protect == erase (memory irrelevant).
"""
import numpy as np

N          = 40      # feature pool
K_SUPPORT  = 8       # live edges per regime
R          = 5       # distinct recurring regimes
T_REGIME   = 200     # steps a regime stays active
STEPS      = 16000
LR_S       = 0.05    # fast sign-shadow rate
ALPHA      = 1.0
SEEDS      = 6
K_GRID     = [1, 2, 5, 10, 20, 50, 100, 200, 400, 800]

# teacher: fixed polarity per feature, R fixed supports (shared pool -> recurrence)
_tr = np.random.default_rng(0)
S_TRUE   = _tr.choice([-1.0, 1.0], size=N)
SUPPORTS = [_tr.choice(N, size=K_SUPPORT, replace=False) for _ in range(R)]

def run(K, mode, seed):
    r = np.random.default_rng(1000 + seed)
    u = np.zeros(N)                                  # sign shadow, S = sign(u)
    G = np.zeros(N); G[r.choice(N, K_SUPPORT, replace=False)] = 1.0
    seen = G > 0                                     # has been live at least once
    acc = np.zeros(N)                                # gate evidence over last K steps
    transfers, tail_sq, tail_n = [], 0.0, 0
    tail_start = STEPS - STEPS // 4
    for t in range(1, STEPS + 1):
        A = SUPPORTS[(t // T_REGIME) % R]
        x = r.standard_normal(N)
        y = float(np.sum(S_TRUE[A] * x[A]))
        S = np.where(u >= 0.0, 1.0, -1.0)
        e = y - ALPHA * float(np.dot(G * S, x))
        u += LR_S * e * x * G                        # fast; *G freezes gated-off edges
        acc += x * y                                 # structural sensing
        if t % K == 0:
            newidx = np.argsort(-np.abs(acc))[:K_SUPPORT]
            cand = [i for i in newidx if seen[i]]    # reconnected & seen before
            if cand:
                transfers.append(np.mean([(u[i] >= 0) == (S_TRUE[i] > 0) for i in cand]))
            newG = np.zeros(N); newG[newidx] = 1.0
            if mode == "erase":
                u[(G > 0) & (newG == 0)] = 0.0        # wipe memory of edges dropped now
            G = newG; seen |= G > 0; acc[:] = 0.0
        if t > tail_start:
            tail_sq += e * e; tail_n += 1
    return tail_sq / tail_n, (np.mean(transfers) if transfers else np.nan)

def main():
    var_y = K_SUPPORT                                # Var(y) = number of unit signed terms
    print(f"teacher: N={N}, support={K_SUPPORT}, regimes={R}, T_REGIME={T_REGIME}, Var(y)={var_y}")
    print(f"{'K':>5} | {'protect NMSE':>12} {'transfer':>9} | {'erase NMSE':>11} | {'gain':>6}")
    print("-" * 56)
    rows = []
    for K in K_GRID:
        pm = np.mean([run(K, "protect", s) for s in range(SEEDS)], axis=0)   # (nmse, transfer)
        em = np.mean([run(K, "erase",   s)[0] for s in range(SEEDS)])
        p_nmse, tr = pm[0] / var_y, pm[1]
        e_nmse = em / var_y
        gain = (e_nmse - p_nmse) / e_nmse * 100                              # % error protect saves
        rows.append((K, p_nmse, tr, e_nmse, gain))
        print(f"{K:>5} | {p_nmse:>12.3f} {tr:>9.2f} | {e_nmse:>11.3f} | {gain:>5.1f}%")

    # verdict
    good = [row for row in rows if row[2] > 0.75 and row[1] < 0.5]           # high transfer, low err
    joint = rows[0]                                                          # K=1, no separation
    print("\n--- verdict (against pre-registration) ---")
    print(f"K=1 (no timescale separation):  NMSE={joint[1]:.3f}, transfer={joint[2]:.2f}")
    if good:
        ks = [g[0] for g in good]
        best = min(good, key=lambda g: g[1])
        print(f"NON-PARASITIC window found at K in {ks}")
        print(f"  boundary K* (smallest good K) = {ks[0]}  <- signs need >= this many held-still steps")
        print(f"  best K={best[0]}: NMSE {best[1]:.3f} (vs joint {joint[1]:.3f}), transfer {best[2]:.2f}, "
              f"protect saves {best[4]:.1f}% error vs erase")
        print("CONFIRMED: slow gate + fast sign is non-parasitic in a K-window; memory (protect) helps.")
    else:
        print("DISCONFIRMED: no clean window (transfer never>0.75 with low error). Report as negative.")

if __name__ == "__main__":
    main()
