#!/usr/bin/env python3
"""Gated rotor -> do Arnold tongues actually appear in the LEARNING dynamics?

We promoted binary polarity {-1,+1} to a phase theta on a circle (the toroidal move).
A rotating structure DEMANDS a polarity that itself rotates at drive frequency omega_d;
the learner's polarity has its own intrinsic clock omega_0 and is pulled toward the
demand by the REAL error-gradient (coupling K = learning rate). The phase update is
DERIVED, not assumed:

  target coeff  c = cos(psi),  psi advances at omega_d      (rotating demand / structure)
  learner coeff a = cos(theta)
  error         e = (c - a) x,   x ~ N(0,1)
  gradient      dtheta ∝ -d(½e²)/dtheta = e * x * sin(theta);  E_x[x²]=1 ->
    theta_{n+1} = theta_n + omega_0 + K * (cos(psi) - cos(theta)) * sin(theta)
    psi_{n+1}   = psi_n   + omega_d

This is a driven nonlinear phase oscillator. IF the toroidal framework governs it,
the rotation number  W = <dtheta>/<dpsi>  locks on rational p/q over BANDS of omega_d
(Arnold tongues) that WIDEN with coupling K -- a devil's staircase. IF the framework
is decoration, W is a smooth featureless curve (no plateaus, locked-fraction ~0 for all K).

PRE-REGISTERED:
  CONFIRM   locked_fraction(K) ~ 0 at K=0 and grows with K; visible plateaus at simple
            rationals (1/2, 1, 2, ...) whose widths increase with K; low tracking error
            inside the 1:1 tongue.
  DISCONFIRM  locked_fraction flat/~0 for all K; W smooth in omega_d; no plateaus.
"""
import numpy as np

OMEGA0 = 0.60                      # learner's intrinsic polarity frequency (rad/step)
K_GRID = [0.0, 0.1, 0.25, 0.5, 0.9]
OMEGAD = np.linspace(0.20, 1.60, 281)   # drive-frequency sweep
STEPS, BURN = 6000, 2500
SIMPLE = [0, 1/3, 1/2, 2/3, 1, 3/2, 2, 3]   # rationals to test locking against
LOCK_TOL = 0.02

def winding_and_error(omega_d, K):
    theta = 0.3; psi = 0.0
    acc_dth = 0.0; sq = 0.0; n_tail = 0; var = 0.0
    for n in range(STEPS):
        c = np.cos(psi)
        dth = OMEGA0 + K * (c - np.cos(theta)) * np.sin(theta)
        theta += dth
        psi += omega_d
        if n >= BURN:
            acc_dth += dth
            resid = c - np.cos(theta)         # how well polarity tracks the demand
            sq += resid * resid; var += c * c; n_tail += 1
    W = (acc_dth / n_tail) / omega_d          # theta winding relative to the drive
    nmse = sq / var if var > 0 else np.nan
    return W, nmse

SLOPE_TOL = 0.05      # a TRUE plateau is flat: |dW/d omega_d| ~ 0 (validates vs K=0 hyperbola)

def locked_mask(Ws):
    """Locked = flat plateau (slope ~ 0) AND near a simple rational. The flatness test
    rejects the K=0 hyperbola merely crossing a rational (its slope is always >> 0)."""
    slope = np.abs(np.gradient(Ws, OMEGAD))
    flat = slope < SLOPE_TOL
    near = np.array([any(abs(W - r) < LOCK_TOL for r in SIMPLE) for W in Ws])
    return flat & near

def locked_fraction(Ws):
    return locked_mask(Ws).mean()

def main():
    print(f"gated rotor: omega_0={OMEGA0}, drive sweep [{OMEGAD[0]:.2f},{OMEGAD[-1]:.2f}], "
          f"{len(OMEGAD)} pts, steps={STEPS}")
    print(f"{'K':>6} | {'locked frac':>11} | main plateaus (rational : approx width in omega_d)")
    print("-" * 74)
    rows = []
    for K in K_GRID:
        Ws = np.array([winding_and_error(od, K)[0] for od in OMEGAD])
        lf = locked_fraction(Ws)
        # measure width of each simple-rational plateau (flat AND near the rational)
        lm = locked_mask(Ws)
        widths = {}
        for r in SIMPLE:
            mask = lm & (np.abs(Ws - r) < LOCK_TOL)
            if mask.any():
                w = mask.sum() * (OMEGAD[1] - OMEGAD[0])
                if w > 0.01:
                    widths[r] = w
        rows.append((K, lf, widths))
        plateaus = "  ".join(f"{r:.2f}:{w:.2f}" for r, w in sorted(widths.items()))
        print(f"{K:>6.2f} | {lf:>11.3f} | {plateaus}")

    # error inside vs outside the 1:1 tongue at the strongest coupling
    K = K_GRID[-1]
    errs = np.array([winding_and_error(od, K)[1] for od in OMEGAD])
    Ws = np.array([winding_and_error(od, K)[0] for od in OMEGAD])
    in11 = np.abs(Ws - 1.0) < LOCK_TOL
    if in11.any() and (~in11).any():
        print(f"\nat K={K}:  tracking NMSE inside 1:1 tongue = {errs[in11].mean():.3f}  "
              f"vs outside = {errs[~in11].mean():.3f}")

    print("\n--- verdict (against pre-registration) ---")
    lf0 = rows[0][1]; lfmax = rows[-1][1]
    grows = all(rows[i][1] <= rows[i + 1][1] + 1e-9 for i in range(len(rows) - 1))
    if lf0 < 0.05 and lfmax > 0.15 and grows:
        print(f"locked fraction {lf0:.3f} (K=0) -> {lfmax:.3f} (K={K_GRID[-1]}), monotone up.")
        print("Arnold tongues PRESENT: the learning dynamics mode-lock; toroidal framework governs.")
    else:
        print(f"locked fraction {lf0:.3f} -> {lfmax:.3f}, grows={grows}.")
        print("NO clean tongue structure -> toroidal framing does NOT govern this system. Report the null.")

if __name__ == "__main__":
    main()
