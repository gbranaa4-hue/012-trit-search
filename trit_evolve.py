"""
trit_evolve -- cross the compute frontier with EVOLUTION, not gradient descent.

Gradient descent failed to make the ternary CA compute (compute_ca_findings.md):
backprop through tens of unrolled CA steps has vanishing/exploding gradients,
and the loss diverged every time. The field solves CA computation with
evolutionary search instead (the classic density-classification result --
Mitchell/Crutchfield/Das, and the hand-designed GKL rule ~81.6%). This does
that, with a ternary rule to keep the project's theme.

Setup (canonical 1D density classification):
  - Ring of N (odd) ternary cells {-1,0,+1}, initialized +/-1 at density d.
  - A small shared local rule (tiny MLP over a radius-r neighborhood) whose
    output is ternarized -- the "ternary neuron rule." Its ~90 real weights are
    EVOLVED, not trained by backprop.
  - Task: every cell must converge to the MAJORITY initial value -- global info
    from local ternary talk.
  - Fitness = consensus amplification: fraction of cells at the TRUE majority
    after T steps, averaged over random inputs. (Do-nothing keeps the initial
    state, so its fitness = the majority density ~0.65; real computation must
    beat that and push toward the ~0.8 ceiling.)

Honest bars, stated up front: exact majority is provably unsolvable by a
uniform CA; GKL (the best hand rule) ~0.816. Beating do-nothing (~0.65) and
chance-accuracy (0.5) with an EVOLVED ternary rule is already a real "the cells
compute a global property" result. ~1.0 is impossible; do not expect it.
"""
import numpy as np

N = 21          # ring size (odd -> no ties)
R = 3           # neighborhood radius (7 cells -- classic density-classification radius)
HID = 10
T = 2 * N       # CA steps (info must cross the ring)
rng = np.random.default_rng(0)

IN = 2 * R + 1
# weight layout: W1 (IN,HID), b1 (HID), W2 (HID,1), b2 (1)
DIM = IN * HID + HID + HID * 1 + 1


def unpack(theta):
    i = 0
    W1 = theta[i:i + IN * HID].reshape(IN, HID); i += IN * HID
    b1 = theta[i:i + HID]; i += HID
    W2 = theta[i:i + HID].reshape(HID, 1); i += HID
    b2 = theta[i:i + 1]; i += 1
    return W1, b1, W2, b2


def ternary(x, thr=0.4):
    return np.where(x > thr, 1.0, np.where(x < -thr, -1.0, 0.0))


def ca_run(theta, state, steps=T):
    """Run the evolved ternary rule for `steps` on a batch of ring states.
    state: (B, N) in {-1,0,+1}. Vectorized over batch and cells."""
    W1, b1, W2, b2 = unpack(theta)
    B = state.shape[0]
    for _ in range(steps):
        neigh = np.stack([np.roll(state, -o, axis=1) for o in range(-R, R + 1)], axis=2)  # (B,N,IN)
        flat = neigh.reshape(B * N, IN)
        h = np.tanh(flat @ W1 + b1)
        o = np.tanh(h @ W2 + b2).reshape(B, N)
        state = ternary(o)
    return state


def make_configs(batch):
    d = rng.uniform(0.2, 0.8, size=batch)
    x = (rng.random((batch, N)) < d[:, None]).astype(np.float64) * 2 - 1
    maj = np.sign(x.sum(axis=1))
    maj[maj == 0] = 1
    return x, maj


def fitness(theta, x, maj):
    final = ca_run(theta, x.copy())
    at_maj = (np.sign(final) == maj[:, None]).mean(axis=1)   # per-config consensus
    return at_maj.mean()


def evaluate(theta, n=2000):
    x, maj = make_configs(n)
    final = ca_run(theta, x.copy())
    init_cons = (np.sign(x) == maj[:, None]).mean()
    final_cons = (np.sign(final) == maj[:, None]).mean()
    solved = ((np.sign(final) == maj[:, None]).mean(axis=1) > 0.9).mean()
    skew = np.abs((x > 0).mean(axis=1) - 0.5)
    easy = skew >= 0.12
    solved_easy = ((np.sign(final[easy]) == maj[easy, None]).mean(axis=1) > 0.9).mean() if easy.any() else float("nan")
    solved_hard = ((np.sign(final[~easy]) == maj[~easy, None]).mean(axis=1) > 0.9).mean() if (~easy).any() else float("nan")
    print(f"  init consensus (do-nothing): {init_cons*100:.1f}%   final consensus: {final_cons*100:.1f}%")
    print(f"  solved (>90% agree, correct): {solved*100:.1f}%   easy: {solved_easy*100:.1f}%   hard: {solved_hard*100:.1f}%")
    return final_cons, solved


def evolve(gens=300, lam=48, mu=12, sigma=0.5, batch=160):
    mean = rng.standard_normal(DIM) * 0.3
    best_theta, best_fit = mean.copy(), -1
    for g in range(gens):
        x, maj = make_configs(batch)                     # fresh batch each gen (noisy but fair within gen)
        pop = mean[None, :] + sigma * rng.standard_normal((lam, DIM))
        fits = np.array([fitness(p, x, maj) for p in pop])
        idx = np.argsort(-fits)[:mu]
        mean = pop[idx].mean(axis=0)
        gen_best = fits[idx[0]]
        if gen_best > best_fit:
            best_fit, best_theta = gen_best, pop[idx[0]].copy()
        sigma *= 0.995                                   # mild annealing
        if g % 25 == 0 or g == gens - 1:
            print(f"gen {g:3d}  best_fit(this gen)={gen_best:.3f}  sigma={sigma:.3f}")
    return best_theta


def main():
    print(f"Evolving a ternary CA rule for density classification "
          f"(N={N}, radius={R}, {DIM} params, {T} steps)\n")
    print("do-nothing baseline fitness ~= majority density ~0.65; chance accuracy 0.5;")
    print("GKL (best hand rule) ~0.816; ~1.0 impossible (provably).\n")
    theta = evolve()
    print("\n=== held-out evaluation of best evolved rule ===")
    evaluate(theta)
    np.save("trit_evolve_best.npy", theta)
    print("\nHonest read: final consensus >> 65% and solved >> chance = the evolved")
    print("ternary cells genuinely compute a global property from local talk.")
    print("~65% / near-chance = evolution found only local smoothing, not real computation.")


if __name__ == "__main__":
    main()
