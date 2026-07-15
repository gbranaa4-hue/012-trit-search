#!/usr/bin/env python3
"""STEP 2 -- the operating-envelope DATASHEET for two-timescale ternary.

Axis 1 (noise vs capture): capture width over (K x batch B), always_on mode.
  PRE-REGISTERED: capture requires kick K*sigma/sqrt(B) < capture halfwidth (prop to K)
  -> the B threshold B* is roughly K-INDEPENDENT (sigma/sqrt(B) < c), and above it
  width grows ~linearly with K.

Axis 2 (re-entry): freeze-mode optimum displacement vs the nearest commensurate point,
  sweeping T_S at fixed K=0.5, B=16.
  PRE-REGISTERED: displacement (omega* - omega0) has the SAME SIGN as
  (nearest 2pi*k/T_S - omega0) and |displacement| < |grid pull| (partial compromise,
  slope between 0 and 1 through the origin).
"""
import numpy as np, json
import network_tongues_step1b as S

def width_run(K, B):
    S.B = B
    E = np.zeros(len(S.OD))
    for s in range(2):
        _, Ev = S.run(K, "always_on", s)
        E += Ev
    E /= 2
    return float((E < 0.5).sum() * (S.OD[1] - S.OD[0]))

def main():
    out = {}
    print("AXIS 1: capture width (NMSE<0.5) over K x B   [pre-reg: B* ~ K-independent; width ~ K above it]")
    Ks, Bs = [0.25, 0.5, 0.9, 1.3], [1, 4, 16, 64]
    print(f"{'':>6}" + "".join(f"B={b:>4} " for b in Bs))
    grid = {}
    for K in Ks:
        row = [width_run(K, b) for b in Bs]
        grid[K] = row
        print(f"K={K:<4}" + "".join(f"{w:>6.2f} " for w in row))
    out["axis1"] = {str(k): v for k, v in grid.items()}

    print("\nAXIS 2: re-entry displacement vs grid pull (freeze, K=0.5, B=16)")
    S.B = 16
    rows = []
    for ts in [11, 13, 15, 17, 19, 21, 23, 26, 29]:
        S.T_S = ts
        E = np.zeros(len(S.OD))
        for s in range(2):
            _, Ev = S.run(0.5, "freeze", s)
            E += Ev
        E /= 2
        opt = float(S.OD[int(np.argmin(E))])
        ks = np.arange(1, 8)
        gridpts = 2 * np.pi * ks / ts
        nearest = float(gridpts[np.argmin(np.abs(gridpts - S.OMEGA0))])
        rows.append((ts, opt, nearest, opt - S.OMEGA0, nearest - S.OMEGA0))
        print(f"  T_S={ts:>3}: optimum {opt:.2f} | nearest commensurate {nearest:.3f} | "
              f"displacement {opt - S.OMEGA0:+.2f} vs pull {nearest - S.OMEGA0:+.3f}")
    S.T_S = 23
    out["axis2"] = rows

    # verdicts
    same_sign = sum(1 for r in rows if r[3] * r[4] > 0 or abs(r[3]) < 0.015)
    partial = sum(1 for r in rows if abs(r[3]) <= abs(r[4]) + 0.02)
    d, n = np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
    slope = float((d @ n) / (n @ n))
    print(f"\nverdict axis2: same-sign {same_sign}/{len(rows)}, |disp|<=|pull| {partial}/{len(rows)}, "
          f"fitted slope {slope:.2f} (pre-reg: 0<slope<1)")
    json.dump(out, open("phase_datasheet.json", "w"))
    print("saved phase_datasheet.json")

if __name__ == "__main__":
    main()
