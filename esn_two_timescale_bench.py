#!/usr/bin/env python3
"""STEP 4 -- does two-timescale ternary COMPUTE, against the boring baseline?

Task: online one-step prediction of Mackey-Glass with the delay parameter SWITCHING
between tau=17 and tau=30 every 600 steps (recurring regimes -- the setting the
mechanism is FOR). Reservoir: fixed 100-node ESN. Readouts, all online:

  dense   : float LMS, updated every step (the boring baseline)
  protect : TwoTimescaleLinear (alpha*G*sign(u)), grads accumulated 8 steps per
            update (datasheet rule 1: average the gradient), dormant signs FROZEN
  erase   : same, but dormant u re-randomized on disconnection (no memory)
  oracle  : per-regime offline ridge (ceiling, not online-fair)

Each online method gets the same lr budget {0.002,0.005,0.01,0.02,0.05}, best taken.

PRE-REGISTERED:
  P1  protect tail-NMSE within 2x of dense (at ~20x weight compression).
  P2  protect beats erase on RETURN-NMSE (first 150 steps after re-entering a
      previously-seen regime) by >20%.
  DISCONFIRM: protect >2x dense (ternary readout too crude here -> report boundary),
      or protect ~= erase (memory worthless in a real task -> report null).
"""
import numpy as np
import torch
import torch.nn.functional as F
from tritkit.twotimescale import TwoTimescaleLinear

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---- Mackey-Glass with switching delay ----
def mackey_glass(total, t_regime=600, taus=(17, 30)):
    hist = 31
    x = list(1.2 + 0.2 * rng.standard_normal(hist))
    tau_seq = []
    for t in range(total):
        tau = taus[(t // t_regime) % len(taus)]
        tau_seq.append(tau)
        xt, xd = x[-1], x[-tau]
        x.append(xt + (0.2 * xd / (1 + xd ** 10) - 0.1 * xt))
    return np.array(x[hist:]), np.array(tau_seq)

TOTAL, T_REG = 24000, 600
series, tau_seq = mackey_glass(TOTAL + 1, T_REG)
series = (series - series.mean()) / series.std()

# ---- reservoir ----
N_RES = 100
Wres = rng.standard_normal((N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
Wres *= 0.9 / max(abs(np.linalg.eigvals(Wres)))
win = 0.5 * rng.standard_normal(N_RES)
LEAK = 0.3

def states(sig):
    s = np.zeros(N_RES); out = np.empty((len(sig), N_RES))
    for t, v in enumerate(sig):
        s = (1 - LEAK) * s + LEAK * np.tanh(Wres @ s + win * v)
        out[t] = s
    return out

S = states(series[:-1])           # state at t predicts series[t+1]
Y = series[1:]
WASH = 300
S, Y = S[WASH:], Y[WASH:]
T = len(Y)
tau_seq = tau_seq[WASH:WASH + T]
St = torch.tensor(S, dtype=torch.float32)
Yt = torch.tensor(Y, dtype=torch.float32)

switches = [t for t in range(1, T) if tau_seq[t] != tau_seq[t - 1]]
seen, returns = set(), []
for t in switches:
    if tau_seq[t] in seen:
        returns.append(t)
    seen.add(tau_seq[t - 1]); seen.add(tau_seq[t])

def metrics(err2):
    var = Y.var()
    tail = err2[T // 2:].mean() / var
    ret = np.mean([err2[t:t + 150].mean() for t in returns if t + 150 < T]) / var
    return tail, ret

# ---- dense float LMS ----
def run_dense(lr):
    w = np.zeros(N_RES + 1); e2 = np.empty(T)
    for t in range(T):
        s = np.append(S[t], 1.0)
        e = Y[t] - w @ s
        e2[t] = e * e
        w += lr * e * s
    return e2

# ---- two-timescale ternary (protect / erase) ----
def run_ttt(lr, erase=False, accum=8, t_gate=64):
    lyr = TwoTimescaleLinear(N_RES, 1, density=0.5, bias=True, evidence_beta=0.9)
    opt = torch.optim.SGD(lyr.parameters(), lr=lr)
    e2 = np.empty(T); n_acc = 0
    G_prev = lyr.G.clone()
    for t in range(T):
        out = lyr(St[t:t + 1])
        loss = F.mse_loss(out, Yt[t:t + 1, None])
        e2[t] = loss.item()
        (loss / accum).backward()
        n_acc += 1
        if n_acc == accum:                      # datasheet rule 1: averaged kick
            opt.step(); opt.zero_grad(); n_acc = 0
        if t % t_gate == t_gate - 1:
            lyr.step_gate()
            if erase:
                with torch.no_grad():
                    dead = (G_prev > 0) & (lyr.G == 0)
                    lyr.u[dead] = 0.01 * torch.randn(int(dead.sum()))
            G_prev = lyr.G.clone()
    return e2

# ---- oracle ridge per regime (ceiling) ----
def run_oracle():
    e2 = np.empty(T)
    for tau in np.unique(tau_seq):
        m = tau_seq == tau
        A = np.hstack([S[m], np.ones((m.sum(), 1))])
        w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(N_RES + 1), A.T @ Y[m])
        e2[m] = (Y[m] - A @ w) ** 2
    return e2

def best(fn, lrs):
    b = None
    for lr in lrs:
        e2 = fn(lr)
        tail, ret = metrics(e2)
        if b is None or tail < b[1]:
            b = (lr, tail, ret)
    return b

LRS = [0.002, 0.005, 0.01, 0.02, 0.05]
print(f"MG switching tau 17/30 every {T_REG}, {T} online steps, {len(returns)} regime returns\n")
d = best(run_dense, LRS)
p = best(lambda lr: run_ttt(lr, erase=False), LRS)
e = best(lambda lr: run_ttt(lr, erase=True), LRS)
ot, orr = metrics(run_oracle())
print(f"{'readout':>10} | {'tail NMSE':>9} | {'return NMSE':>11} | {'best lr':>7} | bits/weight")
print("-" * 62)
print(f"{'oracle':>10} | {ot:>9.4f} | {orr:>11.4f} | {'--':>7} | 32 (offline ceiling)")
print(f"{'dense':>10} | {d[1]:>9.4f} | {d[2]:>11.4f} | {d[0]:>7} | 32")
print(f"{'protect':>10} | {p[1]:>9.4f} | {p[2]:>11.4f} | {p[0]:>7} | ~1.6")
print(f"{'erase':>10} | {e[1]:>9.4f} | {e[2]:>11.4f} | {e[0]:>7} | ~1.6")
print("\n--- pre-registered verdicts ---")
print(f"P1 protect within 2x dense: {p[1]:.4f} vs {2*d[1]:.4f} -> {'CONFIRMED' if p[1] <= 2*d[1] else 'NOT MET (report the boundary)'}")
gain = (e[2] - p[2]) / e[2] * 100 if e[2] > 0 else float('nan')
print(f"P2 protect vs erase on returns: {p[2]:.4f} vs {e[2]:.4f} ({gain:+.0f}%) -> "
      f"{'CONFIRMED' if gain > 20 else 'NOT MET (memory not worth it here)'}")
