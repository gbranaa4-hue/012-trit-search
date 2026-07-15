#!/usr/bin/env python3
"""Does 'noble ratio -> more stable training' transfer to a MAINSTREAM two-clock mechanism?

The DQN target network is the canonical two-clock trick: a fast online net + a slow target
net hard-refreshed every T steps. In a PERIODIC environment (period P) the refresh is a
periodic perturbation at period T while the environment drives at period P -- two periods on
one learner, the same torus setup as the gated rotor. Rotor rule 2: independent clocks
should sit at INCOMMENSURATE ratios; commensurate = parasitic entrainment.

Testbed (target-network mechanism in isolation, NOT a full agent): linear semi-gradient
TD(0) with a target network, tracking a periodic reward -- the classic setting where target
nets matter. Instability metric = steady-state RMS TD error (residual the refresh injects).

CONFOUND CONTROL: bigger T = staler target, a smooth monotone trend independent of ratio.
So we sweep T densely, DETREND (rolling median), and read the RESIDUAL at each ratio -- a
commensurability resonance shows as a residual BUMP over the smooth trend, exactly like the
rotor's 'flat plateau vs smooth hyperbola'.

PRE-REGISTERED:
  CONFIRM   residual instability is POSITIVE (bump) at simple-rational rho=T/P (1,1/2,2,2/3,
            3/2) and <=0 at golden rho (1/phi, phi); golden also below the generic irrationals.
  DISCONFIRM  residuals flat / no structure at rationals -> target-net timing is ratio-
            agnostic, the rule does NOT transfer. Golden not below rationals -> HALF-transfer
            (like the rotor's phi branch); report exactly which.
"""
import numpy as np

import sys
GAMMA = float(sys.argv[sys.argv.index("--gamma") + 1]) if "--gamma" in sys.argv else 0.9
ALPHA = float(sys.argv[sys.argv.index("--alpha") + 1]) if "--alpha" in sys.argv else 0.02
SIG = float(sys.argv[sys.argv.index("--sig") + 1]) if "--sig" in sys.argv else 0.10
P, D = 100, 6
STEPS, BURN, SEEDS = 14000, 5000, 6
phi = (1 + 5 ** 0.5) / 2
ks = np.arange(1, D // 2 + 1)
ph = 2 * np.pi * np.arange(P) / P
FEAT = np.concatenate([np.cos(np.outer(ph, ks)), np.sin(np.outer(ph, ks))], axis=1)  # (P,D)
REW = np.cos(ph) + 0.5 * np.cos(2 * ph)                                              # (P,)
NOISE = np.random.default_rng(0).normal(0, SIG, (STEPS, SEEDS))                      # shared across T
W0 = np.random.default_rng(7).normal(0, 0.01, (SEEDS, D))

def rms_for_T(T):
    w = W0.copy(); wt = w.copy(); sq = np.zeros(SEEDS); cnt = 0; mx = np.zeros(SEEDS)
    for t in range(STEPS):
        x, xn, r = FEAT[t % P], FEAT[(t + 1) % P], REW[t % P]
        delta = r + NOISE[t] + GAMMA * (wt @ xn) - (w @ x)      # (SEEDS,)
        w = w + ALPHA * delta[:, None] * x[None, :]
        w = np.clip(w, -1e6, 1e6)                               # keep finite; huge = diverged
        if (t + 1) % T == 0:
            wt = w.copy()
        if t >= BURN:
            sq += delta ** 2; cnt += 1; mx = np.maximum(mx, np.abs(w).max(axis=1))
    rms_for_T.maxnorm = float(mx.mean())
    return np.sqrt(sq / cnt).mean()

def rollmed(a, w=7):
    pad = w // 2; ap = np.pad(a, pad, mode="edge")
    return np.array([np.median(ap[i:i + w]) for i in range(len(a))])

Ts = np.arange(30, 201, 3)
rms, norms = [], []
for T in Ts:
    rms.append(rms_for_T(int(T))); norms.append(rms_for_T.maxnorm)
rms, norms = np.array(rms), np.array(norms)
resid = rms - rollmed(rms)
print(f"[instrument check] alpha={ALPHA} gamma={GAMMA} sig={SIG}: "
      f"peak |w| over sweep = {norms.min():.2f}..{norms.max():.2f} "
      f"({'DIVERGES (>1e3) -- instability available' if norms.max() > 1e3 else 'bounded -- benign/contractive'})\n")

LAB = {"1/2  rational": 0.5, "2/3  rational": 2/3, "1/1  rational": 1.0, "3/2  rational": 1.5,
       "2/1  rational": 2.0, "1/phi GOLDEN": 1/phi, "phi   GOLDEN": phi,
       "sqrt2-1 irrat": 2**0.5-1, "pi-3   irrat": np.pi-3, "e-2    irrat": np.e-2}

def at(rho):
    T = round(rho * P); i = int(np.argmin(np.abs(Ts - T))); return Ts[i], rms[i], resid[i]

print(f"P={P} env period; T sweep {Ts[0]}..{Ts[-1]}; instability = steady-state RMS TD error")
print(f"(detrended: residual>0 = resonance bump over the smooth staleness trend)\n")
print(f"{'ratio T/P':>15} | {'T':>4} | {'RMS':>7} | {'residual':>9} | kind")
print("-" * 58)
rows = []
for label, rho in LAB.items():
    T, r, res = at(rho)
    kind = "GOLDEN" if "GOLDEN" in label else ("rational" if "rational" in label else "irrat")
    rows.append((label, res, kind))
    print(f"{label:>15} | {T:>4} | {r:>7.4f} | {res:>+9.4f} | {kind}")

rat = np.mean([res for lb, res, k in rows if k == "rational"])
gol = np.mean([res for lb, res, k in rows if k == "GOLDEN"])
irr = np.mean([res for lb, res, k in rows if k == "irrat"])
print(f"\nmean residual  rationals={rat:+.4f}  golden={gol:+.4f}  irrationals={irr:+.4f}")
print("\n--- verdict vs pre-registration ---")
if rat > 0 and gol < rat and gol <= irr + 1e-4:
    print("TRANSFERS: rationals resonate (residual>0), golden is quietest. Noble-ratio rule holds outside the ternary map.")
elif rat > 0 and gol < rat:
    print("HALF-TRANSFERS: rationals resonate and golden beats them, but a generic irrational is as quiet or quieter (rotor's phi-branch story).")
elif abs(rat) < 0.002 and abs(gol) < 0.002:
    print("NULL: no commensurability structure -> target-net timing is ratio-agnostic here. Report the null.")
else:
    print("Mixed/does-not-transfer -> report the raw table honestly.")
