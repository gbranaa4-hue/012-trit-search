"""
Density-classification cellular automaton -- built Socratically, decision by decision.
The architect (you) makes each design call; each line here is one of your decisions.
"""
import numpy as np

# DECISION 1 -- the raw material is a single ROW of N cells.
# To step one tick forward we only need the CURRENT row (each new row is computed
# from the one before it), so we keep just one row alive; the grid/history is an
# optional scrapbook, not required to run.

N = 21          # how many cells sit in the row

# DECISION 4a -- sight radius, made a KNOB so we can sweep it (your call: try sizes).
# A cell sees itself + R neighbors on EACH side. R is the speed limit (max cells
# info can jump per tick); time carries it the rest of the way across the ring.
R = 3           # start at 3: rule stays small, info still moves ~3 cells/tick
IN = 2 * R + 1  # total cells a cell looks at = itself + R left + R right = 7

# A "row of many values" in numpy is an ARRAY -- one box holding a sequence:
#   np.array([1, -1, 1, 1, -1, ...])   <- N slots, one per cell

# DECISION 2 -- each cell holds one of two opinions, labelled +1 and -1.
# Payoff (your reasoning): the MAJORITY is just the SIGN of the row's sum.
#   sum > 0  ->  more +1s  ->  majority is +1
#   sum < 0  ->  more -1s  ->  majority is -1
# N is odd, so the sum is never 0 -> there is never a tie.

def majority(row):
    return np.sign(row.sum())        # +1 or -1: the correct answer for this row


# DECISION 3 -- make starting rows with a SPREAD of difficulty.
# A fair (50%) coin only ever gives near-ties (hard). So we turn ONE dial -- the
# coin's bias -- to a random spot for each row: densities near 0.2 or 0.8 give
# lopsided easy rows, near 0.5 gives hard near-ties. That covers easy -> hard.
#
# We make many rows at once (a "batch") for speed: x has shape (batch, N), where
# each ROW is one independent world. (This batch-grid is separate worlds stacked,
# NOT one world's time-history -- don't confuse it with the space-time grid.)

def random_start(batch, rng):
    density = rng.uniform(0.2, 0.8, size=(batch, 1))            # each row's own +1-bias
    x = np.where(rng.random((batch, N)) < density, 1.0, -1.0)   # fill cells at that bias
    maj = np.sign(x.sum(axis=1))                                # true answer per row
    return x, maj


# DECISION 4b -- the row is a RING (your call): no dead ends, every cell identical,
# information can circulate all the way around. np.roll slides the whole ring by
# an offset and WRAPS the end back to the start -- that wrap IS the ring.

def neighborhoods(state):
    """For every cell, gather its IN-cell neighborhood. state:(batch,N)->(batch,N,IN)."""
    return np.stack([np.roll(state, -off, axis=1) for off in range(-R, R + 1)], axis=2)


# DECISION 5 -- turn the weighted sum into a valid cell value {-1, 0, +1}.
# A weak sum (near 0) is an UNCONFIDENT vote -> stay at the 0 baseline (blank).
# Only a sum past the confidence bar THR commits to +1 or -1.
THR = 0.4

def ternarize(v):
    return np.where(v > THR, 1.0, np.where(v < -THR, -1.0, 0.0))


# DECISION 6 -- start SIMPLE (measure before adding complexity): ONE weighted sum.
# Knobs = 7 weights (one per neighbor) + 1 offset = 8 numbers, stored in `theta`.

DIM = IN + 1

def rule_step(theta):
    """Turn a knob-vector into a one-tick update for the whole ring."""
    w, b = theta[:IN], theta[IN]
    def step(s):
        summed = neighborhoods(s) @ w + b        # weighted vote per cell -> (batch, N)
        return ternarize(summed)
    return step

def run(step, x, steps):
    s = x.copy()
    for _ in range(steps):
        s = step(s)
    return s


# The corrected-repetition engine (your phrase). The KNOWN TRUE SOURCE is `maj`.
# Score = fraction of cells that reach the true answer. Evolution repeats:
# try `lam` random variants, keep the best `mu`, move toward them, repeat.

def score(theta, x, maj):
    final = run(rule_step(theta), x, steps=2 * N)
    return (np.sign(final) == maj[:, None]).mean()

def evolve(rng, gens=120, lam=40, mu=10, sigma=0.5):
    center = rng.standard_normal(DIM) * 0.3
    best, best_s = center.copy(), -1.0
    for g in range(gens):
        x, maj = random_start(160, rng)                        # fresh truth each gen
        pop = center + sigma * rng.standard_normal((lam, DIM))  # random variants
        scores = np.array([score(p, x, maj) for p in pop])
        top = np.argsort(-scores)[:mu]                          # keep the best few
        center = pop[top].mean(axis=0)
        if scores[top[0]] > best_s:
            best_s, best = scores[top[0]], pop[top[0]].copy()
        sigma *= 0.995
    return best, best_s


# DECISION 7 -- ROBUSTNESS (your call): stress the rule in ways N=21 can't show.
# Test the SAME rule at bigger ring sizes it never trained on, split by polarity.
# (The rule works at any size: radius R and np.roll don't care about N.)

def measure_at(theta, N_test, rng, batch=3000):
    density = rng.uniform(0.2, 0.8, size=(batch, 1))
    x = np.where(rng.random((batch, N_test)) < density, 1.0, -1.0)
    maj = np.sign(x.sum(axis=1))
    final = run(rule_step(theta), x, steps=2 * N_test)
    solved = (np.sign(final) == maj[:, None]).mean(axis=1) > 0.9
    frac = (x > 0).mean(axis=1); hard = np.abs(frac - 0.5) < 0.12
    pct = lambda m: solved[m].mean() * 100 if m.any() else float("nan")
    return pct(np.ones(batch, bool)), pct(hard), pct(hard & (maj > 0)), pct(hard & (maj < 0))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, maj = random_start(3000, rng)
    print(f"world N={N}, radius R={R}, {DIM} knobs (ONE neuron)\n")
    best, best_s = evolve(rng)
    final = run(rule_step(best), x, steps=2 * N)
    solved = ((np.sign(final) == maj[:, None]).mean(axis=1) > 0.9).mean()
    frac = (x > 0).mean(axis=1); hard = np.abs(frac - 0.5) < 0.12
    solved_hard = ((np.sign(final[hard]) == maj[hard, None]).mean(axis=1) > 0.9).mean()
    print(f"in-training best score {best_s:.3f}  (flattering -- selected on it)")
    print(f"held-out solved {solved*100:.1f}%   HARD near-ties {solved_hard*100:.1f}%")
    np.save("ca_build_best.npy", best)
