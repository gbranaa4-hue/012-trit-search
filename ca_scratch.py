"""
Density classification with a ternary cellular automaton, bred by evolution.
Built from the ground up. Each section is one idea from the lesson:

  1. THE WORLD    -- a ring of cells updated by a shared local rule (the CA)
  2. THE TASK     -- make every cell agree on the STARTING majority
                     (a global fact, from purely local sight)
  3. BASELINES    -- do-nothing / local-majority, to prove the task is HARD
  4. THE RULE     -- (next) a tiny network over a neighborhood = the tunable knobs
  5. EVOLUTION    -- (next) breed the knobs, no gradients
  6. MEASURE      -- (next) overall, hard near-ties, per-polarity

Numpy only. Run it: python ca_scratch.py
"""
import numpy as np

# ============================================================
# 1. THE WORLD
# ============================================================
# N cells in a CIRCLE (a "ring": cell 0's left neighbor is cell N-1, so there
# are no edges). Each cell holds a value in {-1, 0, +1} -- two opinions and a
# "blank". Every cell updates at each tick by looking at itself + R neighbors
# on each side, and every cell uses the SAME rule. That's a cellular automaton.

N = 21              # number of cells (ODD, so a majority always exists -- no ties)
R = 3               # sight radius: a cell sees itself + 3 on each side
IN = 2 * R + 1      # neighborhood size = 7 cells


def neighborhoods(state):
    """For every cell, gather its 7-cell neighborhood. state:(batch,N) -> (batch,N,IN).
    np.roll slides the ring so we can grab all neighbors without loops."""
    return np.stack([np.roll(state, -off, axis=1) for off in range(-R, R + 1)], axis=2)


def run(step_fn, x, steps):
    """Apply a step function to the whole ring, `steps` times."""
    s = x.copy()
    for _ in range(steps):
        s = step_fn(s)
    return s


# ============================================================
# 2. THE TASK
# ============================================================
# Start each ring with cells set to -1 or +1 at some random density. The job:
# after running the rule, EVERY cell should equal the value that was in the
# MAJORITY at the start. No single cell can see the whole ring -- so the only
# way to win is to let information TRAVEL around the ring (this is why velocity
# will matter later).

def random_start(batch, rng):
    density = rng.uniform(0.2, 0.8, size=(batch, 1))            # fraction of +1 per row
    x = np.where(rng.random((batch, N)) < density, 1.0, -1.0)   # the starting ring
    majority = np.sign(x.sum(axis=1))                           # the TRUE answer per row
    majority[majority == 0] = 1
    return x, majority


def solved_fraction(final, majority):
    """A row is 'solved' if >90% of its cells reached the true majority."""
    agree = (np.sign(final) == majority[:, None]).mean(axis=1)
    return (agree > 0.9).mean()


# ============================================================
# 3. BASELINES -- prove the task is hard
# ============================================================
# do-nothing: never changes the ring. Wins only if the ring already STARTED
#   >90% one way (rare) -- so it should score near zero. This is the floor.
# local-majority: each cell becomes the majority of its neighborhood. It smooths
#   locally but freezes into stripes/domains -- it canNOT carry a global count.

def do_nothing(s):
    return s


def local_majority(s):
    return np.sign(neighborhoods(s).sum(axis=2))


# ============================================================
# 4. THE RULE -- the tunable knobs
# ============================================================
# A cell's next value is decided by a tiny 2-layer network over its 7-cell
# neighborhood: multiply by weights, squash with tanh, again, then TERNARIZE the
# result to {-1,0,+1}. The ~91 weights (theta) are the KNOBS we will tune. This
# is exactly the "machine with knobs" from the lesson -- nothing more.

HID = 10                                   # hidden units in the little network
DIM = IN * HID + HID + HID + 1             # W1, b1, W2, b2 flattened = 91 numbers


def unpack(theta):
    i = 0
    W1 = theta[i:i + IN * HID].reshape(IN, HID); i += IN * HID
    b1 = theta[i:i + HID];                       i += HID
    W2 = theta[i:i + HID].reshape(HID, 1);       i += HID
    b2 = theta[i:i + 1]
    return W1, b1, W2, b2


def ternarize(x, thr=0.4):
    return np.where(x > thr, 1.0, np.where(x < -thr, -1.0, 0.0))


def rule_step(theta):
    """Turn a knob-vector into a step function the ring can run."""
    W1, b1, W2, b2 = unpack(theta)

    def step(s):
        nb = neighborhoods(s).reshape(-1, IN)      # every cell's 7-neighborhood
        h = np.tanh(nb @ W1 + b1)                  # layer 1
        o = np.tanh(h @ W2 + b2).reshape(s.shape)  # layer 2
        return ternarize(o)                        # snap to {-1,0,+1}
    return step


# ============================================================
# 5. EVOLUTION -- breed the knobs, no gradients
# ============================================================
# Score a knob-vector = fraction of cells that reach the true majority.
# Each generation: throw `lam` random variants around the current center (this
# is the "handful of balls" from the lesson), keep the best `mu`, average them
# into the new center, repeat. No slope, no backprop -- just try and keep.

def score(theta, x, maj):
    final = run(rule_step(theta), x, steps=2 * N)
    return (np.sign(final) == maj[:, None]).mean()


def evolve(rng, gens=150, lam=40, mu=10, sigma=0.5):
    center = rng.standard_normal(DIM) * 0.3
    best, best_s = center.copy(), -1.0
    for g in range(gens):
        x, maj = random_start(160, rng)                        # fresh problems each gen
        pop = center + sigma * rng.standard_normal((lam, DIM))  # the handful of variants
        scores = np.array([score(p, x, maj) for p in pop])
        top = np.argsort(-scores)[:mu]                          # keep the best few
        center = pop[top].mean(axis=0)                          # move to their average
        if scores[top[0]] > best_s:
            best_s, best = scores[top[0]], pop[top[0]].copy()
        sigma *= 0.995                                          # shrink the search over time
        if g % 30 == 0 or g == gens - 1:
            print(f"  gen {g:3d}  best-so-far score {best_s:.3f}")
    return best


# ============================================================
# 6. MEASURE HONESTLY
# ============================================================
# Overall solved% flatters a rule by including easy skewed rings. The honest
# tests: HARD near-ties (where cheating on skew is impossible), and each
# POLARITY separately (an average hides a rule that's great on +1 and weak on -1).

def measure(theta, rng, batch=4000):
    x, maj = random_start(batch, rng)
    final = run(rule_step(theta), x, steps=2 * N)
    solved = (np.sign(final) == maj[:, None]).mean(axis=1) > 0.9
    frac = (x > 0).mean(axis=1)
    hard = np.abs(frac - 0.5) < 0.12                    # near-tie rings
    def pct(mask): return solved[mask].mean() * 100 if mask.any() else float("nan")
    print(f"  overall solved   {solved.mean()*100:5.1f}%")
    print(f"  HARD (near-tie)  {pct(hard):5.1f}%   <- the honest test: no skew to lean on")
    print(f"  on +1-majority   {pct(maj > 0):5.1f}%")
    print(f"  on -1-majority   {pct(maj < 0):5.1f}%   <- gap vs +1 = polarity bias")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, maj = random_start(3000, rng)
    print(f"ring N={N}, sight radius R={R}, running {2*N} ticks\n")
    print("baselines:")
    for name, fn in [("do-nothing", do_nothing), ("local-majority", local_majority)]:
        print(f"  {name:20s} solved {solved_fraction(run(fn, x, 2*N), maj)*100:5.1f}%")

    print("\nevolving a rule (no gradients)...")
    theta = evolve(rng)
    final = run(rule_step(theta), x, steps=2 * N)
    print(f"\n  EVOLVED rule        solved {solved_fraction(final, maj)*100:5.1f}%")
    np.save("ca_scratch_best.npy", theta)
    print("\nhonest measurement of the evolved rule (Stage 6):")
    measure(theta, rng)
