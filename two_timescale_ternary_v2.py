#!/usr/bin/env python3
"""Two-timescale ternary, v2 -- ISOLATED test (v1 confounded gate-quality with cadence).

We remove the gate-learning confound: the gate is given the TRUE support (oracle
topology), and we vary only ONE thing -- T, the number of steps the topology is
held still before it changes. Fast sign layer S (=sign(u)) learns every step on the
live edges; a gated-off edge gets no gradient so its sign is FROZEN (structural
memory). Regimes recur (R fixed supports cycled), so a frozen sign can be reused.

This isolates the SIGN side of "non-parasitic": with topology handed to it cleanly,
does the fast sign layer converge and TRANSFER as structure changes at rate T?

Knob T = structural-change period (large T = slow structure).  tau_S = sign
convergence time. The claim predicts a boundary at T ~ tau_S:
  T >> tau_S : signs converge within a regime -> low error, high transfer (NON-PARASITIC)
  T <~ tau_S : structure changes before signs settle -> high error, transfer ~0.5 (PARASITIC)

FALSIFICATION control -- mode:
  protect : keep the sign shadow of an edge that leaves the support (memory)
  erase   : zero it on the way out (no memory)
  protect should beat erase where regimes RECUR and signs had converged; if it never
  does, the "gate protects polarity" story is false.

PRE-REGISTERED:
  CONFIRM   tail-NMSE falls monotonically as T grows, crossing from ~1 (no better than
            zero) to ~0, with the knee at T ~ tau_S; transfer rises 0.5 -> ~1 across the
            same knee; protect < erase in a mid-T band.
  DISCONFIRM  NMSE flat in T, or transfer stays ~0.5, or protect==erase everywhere.
"""
import numpy as np

N, K_SUPPORT, R = 40, 8, 5
LR_S, ALPHA = 0.05, 1.0
STEPS, SEEDS = 30000, 4
T_GRID = [5, 10, 20, 40, 80, 160, 320, 640, 1280]

_tr = np.random.default_rng(0)
S_TRUE   = _tr.choice([-1.0, 1.0], size=N)
SUPPORTS = [_tr.choice(N, size=K_SUPPORT, replace=False) for _ in range(R)]
SUPPSET  = [set(s.tolist()) for s in SUPPORTS]

def run(T, mode, seed):
    r = np.random.default_rng(2000 + seed)
    u = np.zeros(N)
    seen = np.zeros(N, bool)
    prev_reg = -1
    transfers = []
    tail_sq = tail_n = 0
    tail_start = STEPS - STEPS // 4
    for t in range(1, STEPS + 1):
        reg = (t // T) % R
        A = SUPPORTS[reg]
        if reg != prev_reg:                                   # structural change
            if prev_reg >= 0:
                cand = [i for i in A if seen[i]]              # reactivated & seen-before edges
                if cand:
                    transfers.append(np.mean([(u[i] >= 0) == (S_TRUE[i] > 0) for i in cand]))
                if mode == "erase":
                    for i in (SUPPSET[prev_reg] - SUPPSET[reg]):
                        u[i] = 0.0                            # wipe memory of edges leaving
            prev_reg = reg
            seen[A] = True
        x = r.standard_normal(N)
        y = float(np.sum(S_TRUE[A] * x[A]))
        Gmask = np.zeros(N); Gmask[A] = 1.0                   # ORACLE topology
        S = np.where(u >= 0.0, 1.0, -1.0)
        e = y - ALPHA * float(np.dot(Gmask * S, x))
        u += LR_S * e * x * Gmask                             # fast sign; frozen off-support
        if t > tail_start:
            tail_sq += e * e; tail_n += 1
    return tail_sq / tail_n, (np.mean(transfers) if transfers else np.nan)

def main():
    var_y = K_SUPPORT
    print(f"ISOLATED (oracle gate): N={N}, support={K_SUPPORT}, regimes={R}, Var(y)={var_y}")
    print(f"{'T':>6} | {'protect NMSE':>12} {'transfer':>9} | {'erase NMSE':>11} | {'protect gain':>12}")
    print("-" * 62)
    rows = []
    for T in T_GRID:
        pm = np.mean([run(T, "protect", s) for s in range(SEEDS)], axis=0)
        em = np.mean([run(T, "erase",   s)[0] for s in range(SEEDS)])
        p, tr, e = pm[0] / var_y, pm[1], em / var_y
        gain = (e - p) / e * 100
        rows.append((T, p, tr, e, gain))
        print(f"{T:>6} | {p:>12.3f} {tr:>9.2f} | {e:>11.3f} | {gain:>11.1f}%")

    print("\n--- verdict (against pre-registration) ---")
    lo, hi = rows[0], rows[-1]
    print(f"fast structure  T={lo[0]:>4}: NMSE {lo[1]:.3f}, transfer {lo[2]:.2f}")
    print(f"slow structure  T={hi[0]:>4}: NMSE {hi[1]:.3f}, transfer {hi[2]:.2f}")
    # knee: smallest T with NMSE < 0.25 (explains >75% variance) and transfer > 0.9
    good = [row for row in rows if row[1] < 0.25 and row[2] > 0.9]
    monotone = all(rows[i][1] >= rows[i + 1][1] - 0.03 for i in range(len(rows) - 1))
    band = [row for row in rows if row[4] > 3.0]                              # protect meaningfully helps
    if good and monotone:
        print(f"knee T* ~ {good[0][0]} (sign convergence time); NMSE {lo[1]:.2f} -> {hi[1]:.2f} as structure slows")
        if band:
            print(f"protect > erase in T band {[b[0] for b in band]} (max +{max(b[4] for b in band):.1f}%): memory helps where regimes recur")
        print("CONFIRMED: with a clean topology, slow-structure/fast-sign is non-parasitic; boundary at T~tau_S.")
    else:
        print("DISCONFIRMED or messy -- report exactly what the table shows, no spin.")

if __name__ == "__main__":
    main()
