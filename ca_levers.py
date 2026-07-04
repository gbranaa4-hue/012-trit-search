"""
Scouting run: three levers to make the rule ROBUST (cross a big ring), each
measured on the honest stress -- hard near-ties at N=149 (unseen), split by
polarity. Reuses the from-scratch pieces in ca_build.py.

Levers:
  1. COMPLEXITY -- a two-layer rule (hidden units + tanh between layers).
  2. TRAINING   -- breed on a MIX of ring sizes {21,49,99}, not just 21.
  3. BOTH       -- two-layer AND multi-size.
Baseline = the one-neuron rule trained only at N=21 (collapsed to ~21% hard@149).
"testing the waters": modest gens, a first look, not the final word.
"""
import numpy as np
import ca_build as cb

IN, HID = cb.IN, 10


# ---- rule variants ------------------------------------------------------
def rule1(theta):                                   # ONE neuron: IN weights + 1 bias
    w, b = theta[:IN], theta[IN]
    return lambda s: cb.ternarize(cb.neighborhoods(s) @ w + b)


DIM1 = IN + 1


def rule2(theta):                                   # TWO layers, tanh between (the trick)
    i = 0
    W1 = theta[i:i + IN * HID].reshape(IN, HID); i += IN * HID
    b1 = theta[i:i + HID];                       i += HID
    W2 = theta[i:i + HID];                       i += HID
    b2 = theta[i]

    def step(s):
        h = np.tanh(cb.neighborhoods(s) @ W1 + b1)  # hidden judgments, squashed
        return cb.ternarize(h @ W2 + b2)            # combine them, then threshold
    return step


DIM2 = IN * HID + HID + HID + 1


# ---- configs / score / evolve at any size ------------------------------
def configs(batch, Nsz, rng):
    d = rng.uniform(0.2, 0.8, size=(batch, 1))
    x = np.where(rng.random((batch, Nsz)) < d, 1.0, -1.0)
    return x, np.sign(x.sum(axis=1))


def score(rule, theta, x, maj, Nsz):
    final = cb.run(rule(theta), x, steps=2 * Nsz)
    return (np.sign(final) == maj[:, None]).mean()


def evolve(rule, DIM, sizes, rng, gens=100, lam=40, mu=10, sigma=0.5):
    center = rng.standard_normal(DIM) * 0.3
    best, best_s = center.copy(), -1.0
    for g in range(gens):
        Nsz = int(sizes[rng.integers(len(sizes))])        # a training size this gen
        x, maj = configs(140, Nsz, rng)
        pop = center + sigma * rng.standard_normal((lam, DIM))
        sc = np.array([score(rule, p, x, maj, Nsz) for p in pop])
        top = np.argsort(-sc)[:mu]
        center = pop[top].mean(axis=0)
        if sc[top[0]] > best_s:
            best_s, best = sc[top[0]], pop[top[0]].copy()
        sigma *= 0.995
    return best


def hard_at(rule, theta, Nt, rng, batch=2500):
    x, maj = configs(batch, Nt, rng)
    final = cb.run(rule(theta), x, steps=2 * Nt)
    solved = (np.sign(final) == maj[:, None]).mean(axis=1) > 0.9
    frac = (x > 0).mean(axis=1); hard = np.abs(frac - 0.5) < 0.12
    pct = lambda m: solved[m].mean() * 100 if m.any() else float("nan")
    return pct(hard), pct(hard & (maj > 0)), pct(hard & (maj < 0))


if __name__ == "__main__":
    levers = [
        ("1 neuron  @ 21",         rule1, DIM1, [21]),
        ("2 layers  @ 21",         rule2, DIM2, [21]),
        ("1 neuron  @ 21,49,99",   rule1, DIM1, [21, 49, 99]),
        ("2 layers  @ 21,49,99",   rule2, DIM2, [21, 49, 99]),
    ]
    print("scouting -- hard near-ties solved at N=149 (unseen), split by polarity\n")
    print(f'{"lever":22} {"hard@149":>9} {"+1":>7} {"-1":>7}')
    for name, rule, DIM, sizes in levers:
        theta = evolve(rule, DIM, sizes, np.random.default_rng(0), gens=100)
        h, hp, hn = hard_at(rule, theta, 149, np.random.default_rng(9))
        print(f'{name:22} {h:8.1f}% {hp:6.1f}% {hn:6.1f}%', flush=True)
    print("\nbaseline (1 neuron @21) was ~21% hard@149. Which lever moves it, and")
    print("does the +1/-1 gap shrink (less polarity bias)?")
