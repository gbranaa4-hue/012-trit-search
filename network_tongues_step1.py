#!/usr/bin/env python3
"""STEP 1 -- does the torus/tongue physics survive the FULL gated network?

Upgrades over the single rotor (arnold_tongues_gated_rotor.py):
  * N=12 phases coupled through ONE shared scalar error (the real network mechanism)
  * correlated inputs (one-factor rho=0.5) -> genuine cross-phase coupling in expectation
  * a real cycling gate: each feature is OFF one regime in three (duty 2/3, period 3*T_S)
  * CORRECT gradient-descent sign (the rotor used ascent; tongues are K-sign-symmetric so
    its staircase stands, but its tracking-error reading was anti-phase -- retracted)

Modes for the gated-off phases:
  always_on : no gating (network control -- should reproduce rotor physics)
  freeze    : theta frozen while off (the protect/structural-memory design)
  drift     : theta free-runs at omega0 while off (no memory)

PRE-REGISTERED:
 A (scaling)   always_on shows a demand-lock tongue (W~1 plateau) near omega0=0.6 that
               widens with K, absent at K=0; NMSE LOW inside it (descent sign fixed).
 B (re-entry)  freeze shows extra NMSE dips on grid  omega_d = k*2pi/T_S
               (0.273,0.546,0.820,1.093,1.366);  drift's dips sit on the SHIFTED grid
               omega_d = omega0 + k*2pi/T_S (0.327,0.600,0.873,1.146,1.419).
               If B holds, commensurate ratios HELP memory reuse under rotating demands
               (a refinement pulling against the naive golden-ratio rule).
 C             if freeze shows no grid dips but the tongue persists, re-entry cost is
               negligible and the incommensurate/golden rule survives unrefined.
 DISCONFIRM    no tongue in always_on -> the reduced model does NOT scale; report null.
"""
import numpy as np

N, R, T_S = 12, 3, 23
OMEGA0, RHO = 0.60, 0.5
STEPS, BURN, SEEDS = 9000, 4500, 3
OD = np.linspace(0.20, 1.60, 141)
LOCK_TOL, SLOPE_TOL = 0.03, 0.05

rng0 = np.random.default_rng(0)
DELTA = rng0.uniform(0, 2 * np.pi, N)
OFF_REGIME = np.repeat(np.arange(R), N // R)      # feature i sleeps during regime OFF_REGIME[i]
INIT = rng0.uniform(0, 2 * np.pi, N)

GRID_A = [2 * np.pi * k / T_S for k in range(1, 6)]                 # freeze re-entry grid
GRID_B = [OMEGA0 + 2 * np.pi * k / T_S for k in range(-1, 4)]       # drift re-entry grid

def run(K, mode, seed):
    r = np.random.default_rng(4000 + seed)
    M = len(OD)
    theta = np.tile(INIT, (M, 1))
    psi = np.zeros(M)
    acc = np.zeros(M); sq = np.zeros(M); sy = np.zeros(M); n = 0
    for t in range(STEPS):
        live = np.ones(N, bool) if mode == "always_on" else (OFF_REGIME != (t // T_S) % R)
        z = r.standard_normal(N)
        x = np.sqrt(1 - RHO ** 2) * z + RHO * r.standard_normal()
        c = np.cos(psi[:, None] + DELTA[None, :])
        y = (c[:, live] * x[live]).sum(1)
        yh = (np.cos(theta[:, live]) * x[live]).sum(1)
        e = y - yh
        dth = np.zeros((M, N))
        # TRUE DESCENT: theta -= K * d(.5 e^2)/dtheta = -K * e * x * sin(theta)... so +=:
        dth[:, live] = OMEGA0 - K * e[:, None] * x[live][None, :] * np.sin(theta[:, live])
        if mode == "drift":
            dth[:, ~live] = OMEGA0
        theta = theta + dth
        psi = psi + OD
        if t >= BURN:
            acc += dth[:, live].mean(1)           # mean d(theta) over LIVE phases
            sq += e * e; sy += y * y; n += 1
    return (acc / n) / OD, sq / sy                # W (vs demand), NMSE

def band(Ws):
    slope = np.abs(np.gradient(Ws, OD))
    m = (np.abs(Ws - 1.0) < LOCK_TOL) & (slope < SLOPE_TOL)
    return m

def dips(nmse, top=6):
    idx = [i for i in range(1, len(OD) - 1) if nmse[i] < nmse[i - 1] and nmse[i] <= nmse[i + 1]]
    idx.sort(key=lambda i: nmse[i])
    return [(float(OD[i]), float(nmse[i])) for i in idx[:top]]

def near(v, grid, tol=0.04):
    return any(abs(v - g) < tol for g in grid)

def main():
    print(f"network: N={N}, duty=2/3, T_S={T_S}, omega0={OMEGA0}, rho={RHO}, seeds={SEEDS}")
    print(f"grid A (freeze re-entry): {[round(g,3) for g in GRID_A]}")
    print(f"grid B (drift  re-entry): {[round(g,3) for g in GRID_B]}\n")

    W0, E0 = run(0.0, "always_on", 0)             # K=0 sanity
    print(f"K=0 sanity: lock-band {band(W0).mean()*100:.0f}% of axis, NMSE mean {E0.mean():.2f} (expect 0%, ~1)\n")

    results = {}
    print(f"{'mode':>10} {'K':>5} | {'lock-band width':>15} | {'NMSE in-band':>12} {'out-band':>9} | dips on own grid?")
    print("-" * 88)
    for mode in ["always_on", "freeze", "drift"]:
        for K in [0.25, 0.5, 0.9]:
            Wm = np.zeros(len(OD)); Em = np.zeros(len(OD))
            for s in range(SEEDS):
                W, E = run(K, mode, s)
                Wm += W; Em += E
            Wm /= SEEDS; Em /= SEEDS
            results[(mode, K)] = (Wm, Em)
            m = band(Wm)
            width = m.sum() * (OD[1] - OD[0])
            ein = Em[m].mean() if m.any() else float("nan")
            eout = Em[~m].mean() if (~m).any() else float("nan")
            dd = dips(Em)
            grid = GRID_A if mode == "freeze" else (GRID_B if mode == "drift" else [])
            hits = [f"{w:.2f}" for w, _ in dd if grid and near(w, grid)]
            print(f"{mode:>10} {K:>5.2f} | {width:>15.2f} | {ein:>12.3f} {eout:>9.3f} | "
                  f"{('YES: ' + ','.join(hits)) if hits else ('n/a' if not grid else 'no')}"
                  f"   top dips: {[(round(w,2), round(v,2)) for w, v in dd[:4]]}")

    print("\n--- verdicts (pre-registered) ---")
    w25 = band(results[('always_on', 0.25)][0]).sum(); w90 = band(results[('always_on', 0.9)][0]).sum()
    Wm, Em = results[('always_on', 0.9)]; m = band(Wm)
    A = w90 > w25 > 0 and m.any() and Em[m].mean() < Em[~m].mean()
    print(f"A scaling : tongue width K=0.25 -> 0.9: {w25} -> {w90} pts; "
          f"NMSE in {Em[m].mean() if m.any() else float('nan'):.3f} vs out {Em[~m].mean():.3f}  => {'CONFIRMED' if A else 'NOT confirmed'}")
    fz = [w for w, _ in dips(results[('freeze', 0.9)][1]) if near(w, GRID_A)]
    dr = [w for w, _ in dips(results[('drift', 0.9)][1]) if near(w, GRID_B)]
    print(f"B re-entry: freeze dips on grid A: {[round(w,2) for w in fz]} | drift dips on grid B: {[round(w,2) for w in dr]}")
    if fz and dr:
        print("  => CONFIRMED: re-entry commensurability is real physics (two-grid signature).")
    elif not fz and not dr:
        print("  => C holds: no grid dips; re-entry cost negligible, golden rule unrefined.")
    else:
        print("  => PARTIAL: one grid only -- report exactly that, investigate before claiming.")

if __name__ == "__main__":
    main()
