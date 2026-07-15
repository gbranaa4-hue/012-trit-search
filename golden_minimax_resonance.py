#!/usr/bin/env python3
"""DECISIVE TEST: does 'noble ratio -> stabler training' hold for a genuinely OSCILLATORY
learner -- an adversarial two-player game (where the target-net null didn't apply, because
a GAN self-oscillates and a target net doesn't)?

Two-player bilinear game min_x max_y x*y under ALTERNATING gradient descent-ascent is
marginally stable: a pure intrinsic oscillation at angular frequency w0 = arccos(1-eta^2/2)
per step (this IS the rotational/limit-cycle dynamics the GAN-dynamics literature studies,
but analyses with centripetal/Hopf tools -- NOT mode-locking). We add the SECOND clock: a
cyclic learning rate eta_t = eta0*(1 + a*sin(w_d t)). Sweep rho = w_d/w0. If two competing
frequencies in adversarial training mode-lock, PARAMETRIC RESONANCE destabilizes at rational
rho (energy pumped into the oscillation -> radius grows); incommensurate (golden) rho stays
bounded. Metric: mean log-growth of the radius per step (Floquet exponent); >0 = unstable.

PRE-REGISTERED
  CONFIRM   growth PEAKS at simple-rational rho (esp. rho=2, the principal Mathieu tongue;
            also 1, 3/2, 2/3) and golden rho (phi, 1/phi) sits in a low-growth valley --
            incommensurate = stable, rational = resonant. Golden at/below matched irrationals
            = the strong 'golden is special' version.
  DISCONFIRM  growth flat in rho (no resonance structure) -> the mode-locking framing does
            not manifest in adversarial training; report the null (as with the target net).
"""
import numpy as np

phi = (1 + 5 ** 0.5) / 2

def growth(rho, eta0=0.25, a=0.9, steps=4000):
    """log10 of the final radius (init=1). ~0 = stable (bounded oscillation); large = the
    cyclic LR pumped the game oscillation into divergence (parametric resonance)."""
    w0 = float(np.arccos(np.clip(1 - eta0 ** 2 / 2, -1, 1)))   # intrinsic freq, rad/step
    wd = rho * w0
    x, y = 1.0, 0.0
    for t in range(steps):
        eta = eta0 * (1 + a * np.sin(wd * t))
        xn = x - eta * y
        yn = y + eta * xn                                      # alternating GDA (marginally stable base)
        x, y = xn, yn
        r = np.hypot(x, y)
        if not np.isfinite(r) or r > 1e12:
            return 12.0                                        # diverged
    return float(np.log10(max(np.hypot(x, y), 1e-12)))

# dense sweep for the tongue structure
RHOS = np.round(np.arange(0.35, 2.55, 0.03), 3)
g = np.array([growth(r) for r in RHOS])
print("parametric-resonance sweep: growth (Floquet exponent) vs rho=w_d/w0  (>0 = unstable)\n")
gmax = g.max() + 1e-9
for r, val in zip(RHOS, g):
    if abs((r * 100) % 15) < 3 or val > 0.2 * gmax:            # thin the printout, keep peaks
        bar = "#" * int(round(40 * max(val, 0) / gmax))
        print(f"  rho={r:4.2f}  growth={val:+.4f} {bar}")

LAB = {"1/2 rational": 0.5, "2/3 rational": 2/3, "1/1 rational": 1.0, "3/2 rational": 1.5,
       "2/1 rational": 2.0, "1/phi GOLDEN": 1/phi, "phi   GOLDEN": phi,
       "sqrt2-1 irr": 2 ** 0.5 - 1, "e-2 irr": np.e - 2, "pi-3 irr": np.pi - 3}
print(f"\n{'ratio':>13} | {'rho':>5} | {'growth':>8} | kind")
print("-" * 46)
rows = []
for lab, rho in LAB.items():
    val = growth(rho)
    kind = "GOLDEN" if "GOLDEN" in lab else ("rational" if "rational" in lab else "irrat")
    rows.append((lab, rho, val, kind))
    print(f"{lab:>13} | {rho:5.3f} | {val:+8.4f} | {kind}")

rat = np.mean([v for l, r, v, k in rows if k == "rational"])
gol = np.mean([v for l, r, v, k in rows if k == "GOLDEN"])
irr = np.mean([v for l, r, v, k in rows if k == "irrat"])
peak_is_rational = RHOS[int(np.argmax(g))]
print(f"\nmean growth  rational={rat:+.4f}  golden={gol:+.4f}  irrational={irr:+.4f}")
print(f"sweep peak (most unstable) at rho={peak_is_rational:.2f}")
print("(growth = log10 final radius; ~0 stable, 12 = diverged)")
print("\n--- verdict vs pre-registration ---")
if g.max() > 1.0 and rat > gol + 0.5:
    strong = ("and at/below the generic irrationals (golden is special)" if gol <= irr + 0.3
              else "(as stable as generic irrationals -- 'incommensurate good', not 'golden unique')")
    print(f"TRANSFERS: rational ratios parametrically RESONATE and diverge (mean {rat:+.2f}), golden stays "
          f"bounded (mean {gol:+.2f}) {strong}. The mode-locking framing DOES manifest in adversarial "
          f"training -- the effect the target net lacked (no intrinsic frequency) appears once the learner "
          f"self-oscillates.")
elif g.max() <= 1.0:
    print("NULL: no resonance structure -> framing does not manifest here.")
else:
    print("MIXED: resonance exists but golden not clearly stabler -- read the raw table.")
