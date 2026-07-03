"""
WHY do some evolved ternary CA rules generalize from N=21 to N=149 and others
don't? The 5-seed run (trit_evolve_seeds.py) showed the unconstrained rule's
large-N generalization is BIMODAL -- 2/5 seeds find excellent generalizers,
3/5 plateau -- and I flagged it as "seed luck." This tries to replace "luck"
with a MECHANISM.

Hypothesis (from CA computational-mechanics theory -- Crutchfield/Hanson/Das):
good density classifiers compute by propagating signals/particles across the
ring. A rule can only integrate global info on a big ring if information travels
fast enough to cross it in the allotted ~2N steps. So a rule's INFORMATION-
PROPAGATION SPEED, measurable at small N without ever seeing N=149, should
predict its N=149 generalization.

Measure: damage-spreading light cone. Take a random config and a copy with one
center cell flipped; run both; the set of cells where the two trajectories
differ is a growing "cone." Its radius grows ~ v*t. v = propagation velocity.

If v correlates with N=149 hard-solved across many evolved seeds, the bimodality
is explained: fast rules cross the big ring, slow ones can't. If it doesn't,
the honest answer stays "we don't know / it's luck."
"""
import sys
import numpy as np
import trit_evolve as te

te.EQUIVARIANT = False
te.DIM = te.IN * te.HID + te.HID + te.HID + 1


def signal_velocity(theta, N=99, trials=80, seed=0):
    """Domain-boundary closing speed -- the signal that carries the computation.
    Seed a MINORITY block; a good rule shrinks it, its boundaries moving inward.
    Speed = (half the block) / (steps to halve the minority) / 2 boundaries.
    Measured on BOTH polarities and averaged, so a merely sign-biased rule (that
    resolves one polarity fast and the other never) scores only moderate."""
    te.N = N                                          # ca_run reshapes using global te.N
    rng = np.random.default_rng(seed)
    c, W = N // 2, N // 3                             # W < N/2 -> the block is the minority
    target = 0.9 * N                                  # near-consensus at the CORRECT majority
    speeds = []
    for minority_sign in (-1.0, +1.0):
        maj_sign = -minority_sign
        state = np.full((trials, N), maj_sign)
        widths = rng.integers(W - 2, W + 3, size=trials)
        for i in range(trials):
            w = int(widths[i]); idx = (np.arange(w) + (c - w // 2)) % N
            state[i, idx] = minority_sign
        init_maj = (np.sign(state) == maj_sign).sum(axis=1).astype(float)
        reached = np.full(trials, np.nan)
        for t in range(1, N + 1):
            state = te.ca_run(theta, state, steps=1)
            mc = (np.sign(state) == maj_sign).sum(axis=1)   # cells now at CORRECT majority
            newly = np.isnan(reached) & (mc >= target)
            reached[newly] = t
        # cells converted to correct majority, per boundary, per step. Collapsing
        # to 0 or noise never reaches correct consensus -> speed 0 (as it should).
        conv = (target - init_maj) / 2.0
        sp = np.where(np.isnan(reached), 0.0, conv / np.maximum(reached, 1))
        speeds.append(sp.mean())
    return float(np.mean(speeds)), None


def asymmetry(theta, N=21, trials=1000, seed=1):
    """Fraction of single-step outputs violating f(-x) == -f(x) (broken-symmetry
    magnitude). The 5-seed note wondered if a little asymmetry is load-bearing."""
    rng = np.random.default_rng(seed)
    te.N = N                                         # ca_run reshapes using global te.N
    d = rng.uniform(0.2, 0.8, size=trials)
    x = (rng.random((trials, N)) < d[:, None]).astype(np.float64) * 2 - 1
    fx = te.ca_run(theta, x.copy(), steps=1)
    fnx = te.ca_run(theta, (-x).copy(), steps=1)
    return (fnx != -fx).mean()


def outcome(theta, seed):
    """N=21 solved (trained size) and N=149 hard-solved (generalization target)."""
    def solved_at(Nv):
        te.N = Nv
        te.rng = np.random.default_rng(7000 + seed)
        x, maj = te.make_configs(1500)
        fin = te.ca_run(theta, x.copy(), steps=2 * Nv)
        solved = ((np.sign(fin) == maj[:, None]).mean(axis=1) > 0.9).mean()
        skew = np.abs((x > 0).mean(axis=1) - 0.5)
        hard = ~(skew >= 0.12)
        h = ((np.sign(fin[hard]) == maj[hard, None]).mean(axis=1) > 0.9).mean()
        return solved, h
    s21, _ = solved_at(21)
    _, h149 = solved_at(149)
    return s21, h149


def sanity():
    """Does the velocity measure behave sensibly on known rules?"""
    print("SANITY: propagation velocity on known rules (N=99, cone must grow)\n")
    best = np.load("trit_evolve_best.npy")
    rngr = np.random.default_rng(42)
    rand = rngr.standard_normal(te.DIM) * 0.3
    zero = np.zeros(te.DIM)                           # ~do-nothing-ish (tanh(0)=0 -> ternary 0)
    for name, th in [("evolved-best", best), ("random", rand), ("near-zero", zero)]:
        v, radii = signal_velocity(th, N=99)
        s21, h149 = outcome(th, 0)
        print(f"  {name:13s}: velocity={v:5.2f} cells/step   asym={asymmetry(th)*100:4.1f}%   "
              f"N21 solved={s21*100:4.1f}  N149 hard={h149*100:4.1f}")
    print("\n  Expect: evolved-best has clear finite velocity + solves; near-zero ~0")
    print("  velocity + fails; random somewhere between. If so, the measure is real.")


def full(K=16):
    """Evolve K unconstrained seeds; test if velocity@N21 predicts N149 generalization."""
    print(f"Evolving {K} unconstrained seeds, correlating propagation velocity "
          f"(measured at small N) with N=149 hard generalization\n")
    rows = []
    for seed in range(K):
        te.N = 21
        te.rng = np.random.default_rng(seed)
        theta = te.evolve(gens=180, lam=40, mu=10)
        np.save(f"predict_theta_{seed}.npy", theta)
        v, _ = signal_velocity(theta, N=99, seed=seed)
        asym = asymmetry(theta)
        s21, h149 = outcome(theta, seed)
        rows.append((seed, v, asym, s21, h149))
        print(f"seed {seed:2d}: velocity={v:5.2f}  asym={asym*100:4.1f}%  "
              f"N21 solved={s21*100:4.1f}  N149 hard={h149*100:4.1f}", flush=True)

    a = np.array([r[1:] for r in rows])              # cols: v, asym, s21, h149
    v, asym, s21, h149 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    def corr(u, w):
        if u.std() < 1e-9 or w.std() < 1e-9:
            return float("nan")
        return np.corrcoef(u, w)[0, 1]

    print("\n" + "=" * 60)
    print("  PREDICTORS OF N=149 HARD GENERALIZATION (across seeds)")
    print("=" * 60)
    print(f"  corr(velocity, N149 hard) : {corr(v, h149):+.3f}   <- main hypothesis")
    print(f"  corr(asymmetry, N149 hard): {corr(asym, h149):+.3f}")
    print(f"  corr(N21 solved, N149 hard): {corr(s21, h149):+.3f}   (does trained-size skill predict?)")
    print(f"\n  velocity range {v.min():.2f}-{v.max():.2f}; N149 hard range "
          f"{h149.min()*100:.0f}-{h149.max()*100:.0f}%")
    print("\n  Honest read: strong positive corr(velocity, N149) => the bimodality is")
    print("  MECHANISM (fast rules cross the big ring), not luck. Near-zero corr =>")
    print("  velocity is not the explanation; the honest answer stays 'unexplained'.")


if __name__ == "__main__":
    if "sanity" in sys.argv:
        sanity()
    else:
        full(K=int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 16)
