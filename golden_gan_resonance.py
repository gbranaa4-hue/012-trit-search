#!/usr/bin/env python3
"""REAL-GAN TEST: does the parametric-resonance tongue structure survive on an actual neural
GAN (nonlinearity + minibatch noise), or was it an artifact of the bilinear toy?

Small 1-D GAN, target = mixture of two Gaussians. Generator & discriminator are MLPs trained
by ALTERNATING SGD (SGD, not Adam, so the cyclic learning rate actually reaches the dynamics).
Step 1: constant-LR run -> FFT the discriminator loss -> intrinsic oscillation period P0.
Step 2: cyclic LR  lr_t = lr0*(1 + A*sin(2*pi*t/P)),  P = P0/rho, sweep rho.
Metric: std of the D-loss over the 2nd half of training (oscillation amplitude) + weight-norm
growth = instability. If mode-locking is real, rho=2 (principal resonance) is most unstable
and golden/incommensurate rho are calmer.

PRE-REGISTERED: rho=2 (and other low rationals) show the LARGEST loss oscillation / weight
growth; golden (phi, 1/phi) among the calmest. DISCONFIRM: flat across rho, or golden not
calmer -> the clean toy result does NOT survive real GAN noise (report the null honestly --
a sim-to-reality boundary, like your reservoir 'didn't replicate' until it had richness).
"""
import numpy as np, torch, torch.nn as nn

torch.set_num_threads(4)
phi = (1 + 5 ** 0.5) / 2
DEV = "cpu"

def real_batch(n):
    m = (torch.rand(n) < 0.5).float() * 3.0 - 1.5           # means -1.5 / +1.5
    return (m + 0.3 * torch.randn(n)).unsqueeze(1)

def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64),
                         nn.LeakyReLU(0.2), nn.Linear(64, o))

def train(P=None, A=0.0, lr0=0.03, steps=2500, seed=0, bs=256):
    torch.manual_seed(seed)
    G, D = mlp(1, 1).to(DEV), mlp(1, 1).to(DEV)
    oG, oD = torch.optim.SGD(G.parameters(), lr0, momentum=0.5), torch.optim.SGD(D.parameters(), lr0, momentum=0.5)
    bce = nn.BCEWithLogitsLoss()
    dloss = np.zeros(steps)
    for t in range(steps):
        lr = lr0 * (1 + A * np.sin(2 * np.pi * t / P)) if P else lr0
        for g in oG.param_groups: g["lr"] = lr
        for g in oD.param_groups: g["lr"] = lr
        x = real_batch(bs)
        z = torch.randn(bs, 1); fake = G(z).detach()
        ld = bce(D(x), torch.ones(bs, 1)) + bce(D(fake), torch.zeros(bs, 1))
        oD.zero_grad(); ld.backward(); oD.step()
        z = torch.randn(bs, 1)
        lg = bce(D(G(z)), torch.ones(bs, 1))               # non-saturating generator loss
        oG.zero_grad(); lg.backward(); oG.step()
        dloss[t] = ld.item()
    wnorm = float(sum(p.detach().pow(2).sum() for p in D.parameters()).sqrt())
    return dloss, wnorm

# --- step 1: intrinsic oscillation period from a constant-LR run ---
dl, _ = train(P=None, seed=0)
sig = dl[500:] - np.polyval(np.polyfit(np.arange(len(dl) - 500), dl[500:], 1), np.arange(len(dl) - 500))
sp = np.abs(np.fft.rfft(sig)); fr = np.fft.rfftfreq(len(sig))
band = (fr > 1 / 150) & (fr < 1 / 6)
P0 = 1.0 / fr[band][np.argmax(sp[band])]
print(f"intrinsic discriminator-loss oscillation period P0 = {P0:.1f} steps\n")

def instability(rho, seeds=3, A=0.6):
    osc, grow = [], []
    for s in range(seeds):
        dl, wn = train(P=P0 / rho, A=A, seed=s)
        osc.append(float(np.std(dl[len(dl) // 2:])))       # 2nd-half loss oscillation amplitude
        grow.append(wn)
    return np.mean(osc), np.mean(grow)

LAB = {"2/1 rational": 2.0, "1/1 rational": 1.0, "3/2 rational": 1.5,
       "1/phi GOLDEN": 1 / phi, "phi   GOLDEN": phi, "sqrt2-1 irr": 2 ** 0.5 - 1}
base_osc, _ = instability(1e6, seeds=3)                     # ~constant LR baseline (huge P)
print(f"baseline (constant LR) loss-oscillation std = {base_osc:.3f}\n")
print(f"{'ratio':>13} | {'rho':>5} | {'loss-osc std':>12} | {'D weight-norm':>13} | kind")
print("-" * 62)
rows = []
for lab, rho in LAB.items():
    o, w = instability(rho)
    kind = "GOLDEN" if "GOLDEN" in lab else ("rational" if "rational" in lab else "irrat")
    rows.append((lab, rho, o, w, kind))
    print(f"{lab:>13} | {rho:5.3f} | {o:12.3f} | {w:13.1f} | {kind}")

rat = np.mean([o for l, r, o, w, k in rows if k == "rational" and r in (2.0, 1.0)])   # resonant rationals
gol = np.mean([o for l, r, o, w, k in rows if k == "GOLDEN"])
print(f"\nmean loss-osc  resonant-rational(2,1)={rat:.3f}  golden={gol:.3f}  baseline={base_osc:.3f}")
print("\n--- verdict vs pre-registration ---")
if rat > gol * 1.25 and rat > base_osc:
    print("SURVIVES: rho=2/1 resonances inflate the GAN's loss oscillation vs golden -- the toy "
          "tongue structure persists through real nonlinearity + minibatch noise.")
elif rat > base_osc * 1.1:
    print("PARTIAL: some resonant inflation but golden not clearly calmer -- noisy; read the table.")
else:
    print("NULL: no resonance structure survives the real GAN's noise -- honest sim-to-reality boundary.")
