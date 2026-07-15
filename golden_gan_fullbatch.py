#!/usr/bin/env python3
"""DIAGNOSTIC: why did the real-GAN parametric-resonance test go null -- minibatch NOISE, or
the GAN's BROADBAND (no sharp intrinsic frequency) dynamics? Remove noise entirely: fix the
real data and the generator's noise inputs, so training is a DETERMINISTIC dynamical system.

  tongues REAPPEAR  -> the killer was minibatch noise (fixable: big batches / averaging);
                       the effect is real in low-noise adversarial training.
  tongues STILL GONE-> the GAN's dynamics are broadband even without noise; the clean effect
                       is a bilinear-toy artifact. Definitive bound.

Same 1-D GAN as golden_gan_resonance.py, but with a FIXED real batch + FIXED z reused every
step (full-batch, deterministic). Re-estimate intrinsic period, sweep cyclic-LR rho.
"""
import numpy as np, torch, torch.nn as nn

torch.set_num_threads(4)
phi = (1 + 5 ** 0.5) / 2

def real_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(n, generator=g) < 0.5).float() * 3.0 - 1.5
    return (m + 0.3 * torch.randn(n, generator=g)).unsqueeze(1)

def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64),
                         nn.LeakyReLU(0.2), nn.Linear(64, o))

def train(P=None, A=0.0, lr0=0.03, steps=2500, seed=0, N=1024):
    torch.manual_seed(seed)
    G, D = mlp(1, 1), mlp(1, 1)
    oG = torch.optim.SGD(G.parameters(), lr0, momentum=0.5)
    oD = torch.optim.SGD(D.parameters(), lr0, momentum=0.5)
    bce = nn.BCEWithLogitsLoss()
    X = real_data(N, seed + 100)                            # FIXED real batch (no resampling)
    Z = torch.randn(N, 1)                                   # FIXED generator noise (no resampling)
    one, zero = torch.ones(N, 1), torch.zeros(N, 1)
    dl = np.zeros(steps)
    for t in range(steps):
        lr = lr0 * (1 + A * np.sin(2 * np.pi * t / P)) if P else lr0
        for gp in oD.param_groups: gp["lr"] = lr
        for gp in oG.param_groups: gp["lr"] = lr
        fake = G(Z).detach()
        ld = bce(D(X), one) + bce(D(fake), zero)
        oD.zero_grad(); ld.backward(); oD.step()
        lg = bce(D(G(Z)), one)
        oG.zero_grad(); lg.backward(); oG.step()
        dl[t] = ld.item()
    wn = float(sum(p.detach().pow(2).sum() for p in D.parameters()).sqrt())
    return dl, wn

dl, _ = train(seed=0)
sig = dl[500:] - np.polyval(np.polyfit(np.arange(len(dl) - 500), dl[500:], 1), np.arange(len(dl) - 500))
sp = np.abs(np.fft.rfft(sig)); fr = np.fft.rfftfreq(len(sig)); band = (fr > 1 / 200) & (fr < 1 / 6)
peak = sp[band].max() / (sp[band].mean() + 1e-9)
P0 = 1.0 / fr[band][np.argmax(sp[band])]
print(f"deterministic run: intrinsic period P0 = {P0:.1f} steps; spectral peak sharpness "
      f"(peak/mean) = {peak:.1f}  ({'SHARP' if peak > 8 else 'broadband' if peak < 4 else 'moderate'})\n")

def instability(rho, seeds=3, A=0.6):
    return np.mean([np.std(train(P=P0 / rho, A=A, seed=s)[0][1250:]) for s in range(seeds)])

base = instability(1e6)
print(f"baseline (constant LR) loss-osc std = {base:.3f}\n")
LAB = {"2/1 rational": 2.0, "1/1 rational": 1.0, "3/2 rational": 1.5,
       "1/phi GOLDEN": 1 / phi, "phi   GOLDEN": phi, "sqrt2-1 irr": 2 ** 0.5 - 1}
print(f"{'ratio':>13} | {'rho':>5} | {'loss-osc std':>12} | kind")
print("-" * 46)
rows = []
for lab, rho in LAB.items():
    o = instability(rho); kind = "GOLDEN" if "GOLDEN" in lab else ("rational" if "rational" in lab else "irrat")
    rows.append((lab, rho, o, kind)); print(f"{lab:>13} | {rho:5.3f} | {o:12.3f} | {kind}")

rat = np.mean([o for l, r, o, k in rows if r in (2.0, 1.0)]); gol = np.mean([o for l, r, o, k in rows if k == "GOLDEN"])
print(f"\nmean loss-osc  resonant-rational(2,1)={rat:.3f}  golden={gol:.3f}  baseline={base:.3f}")
print("\n--- diagnostic verdict ---")
if rat > gol * 1.25 and rat > base * 1.2:
    print("NOISE was the killer: without minibatch noise the tongues REAPPEAR (rational resonates, "
          "golden calmer). The effect is real in low-noise adversarial training.")
elif peak < 4:
    print("BROADBAND is the killer: even noise-free, the GAN's oscillation has no sharp frequency "
          "(peak/mean<4) -> no resonance to lock. The clean effect is a bilinear-toy artifact. Definitive.")
else:
    print("Still no clean tongues even deterministic -> leans toward broadband/landscape, not noise. "
          "Read the table; the toy effect does not carry to a real GAN.")
