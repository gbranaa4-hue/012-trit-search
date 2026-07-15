#!/usr/bin/env python3
"""Two independent clocks: which frequency ratio resists PARASITIC lock the longest?

Premise under test: two clocks meant to run INDEPENDENTLY should sit at incommensurate
frequency ratios -- entrainment by the wrong clock is the parasitic failure mode. KAM
theory predicts the GOLDEN ratio (most irrational, continued fraction [1;1,1,...]) is the
LAST invariant torus to break, so it should tolerate the highest coupling K before
spuriously locking.

We reuse the SAME derived gated-rotor map (arnold_tongues_gated_rotor.winding_and_error).
For each target ratio we set omega_d = omega_0 / ratio (so the uncoupled winding W ~ ratio),
then raise K until the winding pins to a simple rational AND goes flat under a small detune
(a real tongue, not a passing crossing). K_lock = that coupling. Higher = more robust.

PRE-REGISTERED: golden 1/phi and phi have the HIGHEST K_lock (lock last); simple rationals
lock earliest; generic irrationals fall in between (golden beats even them). DISCONFIRM:
golden lock threshold not above the rationals' -> the KAM/quasicrystal story does not govern.
"""
import numpy as np
import arnold_tongues_gated_rotor as gr

gr.STEPS, gr.BURN = 3500, 1500          # a little lighter than the tongue-map run
phi = (1 + 5 ** 0.5) / 2
O0 = gr.OMEGA0

CANDS = {                               # ratio  ->  label
    0.5: "1/2  rational", 2/3: "2/3  rational", 1.0: "1/1  rational",
    1.5: "3/2  rational", 2.0: "2/1  rational",
    1/phi: "1/phi GOLDEN", phi: "phi   GOLDEN",
    2 ** 0.5 - 1: "sqrt2-1 irrat", np.pi - 3: "pi-3   irrat", np.e - 2: "e-2    irrat",
}
KS = np.round(np.arange(0.0, 1.31, 0.05), 3)
DET, TOL = 0.012, gr.LOCK_TOL

def locked(omega_d, K):
    Ws = [gr.winding_and_error(omega_d * (1 + d), K)[0] for d in (-DET, 0.0, DET)]
    flat = (max(Ws) - min(Ws)) < TOL
    near = any(abs(Ws[1] - r) < TOL for r in gr.SIMPLE)
    return flat and near

print(f"omega_0={O0}; sweeping K in [0,{KS[-1]}]; lock = flat plateau at a simple rational\n")
rows = []
for ratio, label in CANDS.items():
    od = O0 / ratio
    klock = next((K for K in KS if K > 0 and locked(od, K)), None)
    rows.append((label, ratio, klock))
for label, ratio, kl in sorted(rows, key=lambda r: (r[2] is not None, -(r[2] or 99))):
    kind = "GOLDEN" if "GOLDEN" in label else ("rational" if "rational" in label else "irrational")
    shown = f"K_lock = {kl:.2f}" if kl is not None else "NEVER locks (K<=1.3)"
    print(f"  {label:>14}  (W~{ratio:.3f})   {shown:>22}   [{kind}]")

golden = [kl for lb, r, kl in rows if "GOLDEN" in lb]
rats   = [kl for lb, r, kl in rows if "rational" in lb]
gmin = min([k for k in golden if k is not None], default=99)
rmax = max([k for k in rats if k is not None], default=0)
print("\n--- verdict vs pre-registration ---")
if all(k is None for k in golden) or gmin > rmax:
    print(f"golden lock threshold ({'never' if gmin==99 else f'>={gmin:.2f}'}) ABOVE every rational "
          f"(worst rational locks at K={rmax:.2f}). KAM/quasicrystal story holds: golden is most lock-resistant.")
else:
    print("golden did NOT out-resist the rationals -> report the null.")
