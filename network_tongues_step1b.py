#!/usr/bin/env python3
"""STEP 1b -- network tongues with gradient noise CONTROLLED (batch-averaged gradient).

Step 1a finding (network_tongues_step1.py, honest read overriding its own verdict
printer): with single-sample gradients, capture exists ONLY at omega_d ~ omega0 at
K=0.25 (NMSE 0.0) and is DESTROYED at K>=0.5 -- noise kick scales with K exactly like
the capture torque, so raising K buys no SNR, only diffusion. The rotor's "tongues
widen with K" did NOT scale naively. Its "grid hits" were the omega0 capture point
mis-matched by tolerance, and K=0.9 "dips" were noise-floor minima. Retracted.

This run batch-averages the gradient over B samples per step (kick /sqrt(B)) --
the discriminating test between "noise effect" and "network effect".

PRE-REGISTERED:
 A'  with B=16, always_on capture band (NMSE<0.5) around omega0 WIDENS with K
     (rotor physics restored under noise control). If still no widening -> genuine
     network limit, report as the boundary.
 B'  freeze shows deep dips (NMSE<0.8) on grid A AWAY from omega0 (0.273, 0.820,
     1.093, 1.366); drift on shifted grid B away (0.327, 0.873, 1.146, 1.419).
     Two-grid signature = re-entry commensurability physics.
 C'  no away-grid dips in either -> re-entry cost negligible; golden rule unrefined.
"""
import numpy as np

N, R, T_S = 12, 3, 23
OMEGA0, RHO = 0.60, 0.5
STEPS, BURN, SEEDS, B = 9000, 4500, 2, 16
OD = np.linspace(0.20, 1.60, 141)
DW = OD[1] - OD[0]

rng0 = np.random.default_rng(0)
DELTA = rng0.uniform(0, 2 * np.pi, N)
OFF_REGIME = np.repeat(np.arange(R), N // R)
INIT = rng0.uniform(0, 2 * np.pi, N)

GRID_A = [2 * np.pi * k / T_S for k in range(1, 6)]
GRID_B = [OMEGA0 + 2 * np.pi * k / T_S for k in (-1, 1, 2, 3)]   # k=0 (=omega0) excluded: unresolvable
AWAY = 0.07                                                       # exclusion zone around omega0

def run(K, mode, seed):
    r = np.random.default_rng(4000 + seed)
    M = len(OD)
    theta = np.tile(INIT, (M, 1))
    psi = np.zeros(M)
    acc = np.zeros(M); sq = np.zeros(M); sy = np.zeros(M); n = 0
    for t in range(STEPS):
        live = np.ones(N, bool) if mode == "always_on" else (OFF_REGIME != (t // T_S) % R)
        x = np.sqrt(1 - RHO**2) * r.standard_normal((B, N)) + RHO * r.standard_normal((B, 1))
        xl = x[:, live]                                   # (B,L)
        c = np.cos(psi[:, None] + DELTA[None, :])         # (M,N)
        y = c[:, live] @ xl.T                             # (M,B)
        st = np.sin(theta[:, live])
        yh = np.cos(theta[:, live]) @ xl.T                # (M,B)
        e = y - yh
        g = (e @ xl) / B * st                             # (M,L) batch-mean grad of .5e^2
        dth = np.zeros((M, N))
        dth[:, live] = OMEGA0 - K * g                     # true descent
        if mode == "drift":
            dth[:, ~live] = OMEGA0
        theta = theta + dth
        psi = psi + OD
        if t >= BURN:
            acc += dth[:, live].mean(1)
            sq += (e * e).mean(1); sy += (y * y).mean(1); n += 1
    return (acc / n) / OD, sq / sy

def deep_dips(nmse, thresh=0.8, top=8):
    idx = [i for i in range(1, len(OD) - 1)
           if nmse[i] < nmse[i - 1] and nmse[i] <= nmse[i + 1] and nmse[i] < thresh]
    idx.sort(key=lambda i: nmse[i])
    return [(float(OD[i]), float(nmse[i])) for i in idx[:top]]

def away_hits(dd, grid, tol=0.035):
    return [(w, v) for w, v in dd if abs(w - OMEGA0) > AWAY and any(abs(w - g) < tol for g in grid)]

def main():
    print(f"batched gradients: B={B} (kick /4), N={N}, T_S={T_S}, omega0={OMEGA0}, seeds={SEEDS}")
    print(f"grid A away-points: {[round(g,3) for g in GRID_A if abs(g-OMEGA0)>AWAY]}")
    print(f"grid B away-points: {[round(g,3) for g in GRID_B]}\n")
    res = {}
    print(f"{'mode':>10} {'K':>5} | {'capture width (NMSE<0.5)':>24} | {'min NMSE':>8} | away-grid dips")
    print("-" * 92)
    for mode in ["always_on", "freeze", "drift"]:
        for K in [0.25, 0.5, 0.9]:
            Wm = np.zeros(len(OD)); Em = np.zeros(len(OD))
            for s in range(SEEDS):
                Wv, Ev = run(K, mode, s)
                Wm += Wv; Em += Ev
            Wm /= SEEDS; Em /= SEEDS
            res[(mode, K)] = (Wm, Em)
            width = (Em < 0.5).sum() * DW
            dd = deep_dips(Em)
            grid = GRID_A if mode == "freeze" else (GRID_B if mode == "drift" else [])
            hits = away_hits(dd, grid) if grid else []
            print(f"{mode:>10} {K:>5.2f} | {width:>24.2f} | {Em.min():>8.3f} | "
                  f"{[(round(w,3), round(v,2)) for w, v in hits] if hits else '-'}"
                  f"   dips: {[(round(w,2), round(v,2)) for w, v in dd[:4]]}")

    print("\n--- verdicts (pre-registered) ---")
    w = {K: (res[('always_on', K)][1] < 0.5).sum() * DW for K in [0.25, 0.5, 0.9]}
    Aprime = w[0.25] > 0 and w[0.5] >= w[0.25] and w[0.9] >= w[0.5]
    print(f"A' widening: capture width {w[0.25]:.2f} -> {w[0.5]:.2f} -> {w[0.9]:.2f}  => "
          f"{'CONFIRMED (noise was the culprit; rotor physics scales under noise control)' if Aprime else 'NOT confirmed (genuine network limit)'}")
    fz = away_hits(deep_dips(res[('freeze', 0.9)][1]), GRID_A) + away_hits(deep_dips(res[('freeze', 0.5)][1]), GRID_A)
    dr = away_hits(deep_dips(res[('drift', 0.9)][1]), GRID_B) + away_hits(deep_dips(res[('drift', 0.5)][1]), GRID_B)
    print(f"B' re-entry: freeze away-hits {[(round(w_,3)) for w_, _ in fz]} | drift away-hits {[(round(w_,3)) for w_, _ in dr]}")
    if fz and dr:
        print("  => CONFIRMED two-grid signature: re-entry commensurability is real physics.")
    elif fz or dr:
        print("  => PARTIAL (one grid): report exactly that; do not claim the mechanism.")
    else:
        print("  => C' holds: no away-grid dips; re-entry cost negligible at these settings.")

if __name__ == "__main__":
    main()
