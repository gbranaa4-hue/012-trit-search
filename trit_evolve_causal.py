"""
CAUSAL test of the signal-velocity mechanism. We showed velocity PREDICTS which
evolved rules generalize (corr +0.98, out-of-sample). Correlation. The causal
question: if we SELECT for velocity during evolution, do we PRODUCE more
generalizers than plain consensus fitness (~19% base rate)?

Design (paired, controlled): for each seed, both conditions start from the
IDENTICAL initialization and see the SAME config draws (te.rng seeded per seed);
the ONLY difference is the fitness:
  baseline:  fitness = consensus                 (lambda_v = 0)
  treatment: fitness = consensus + lambda_v * signal_velocity(probe)

Outcome measured is the REAL target -- N=149 hard-solve -- NOT the probe, so a
rule can't "win" by gaming the probe without actually generalizing. A rule
counts as a generalizer if N=149 hard > 0.35 (clean gap in the 16-seed data:
winners 51-60%, losers 13-21%).

Honest pre-registration:
  H0: velocity term doesn't help -> treatment hit rate ~ baseline (~19%).
  H1 (causal): treatment hit rate > baseline.
With K=12/condition the hit-rate difference has a wide binomial CI; a large,
consistent gap is suggestive, a small one is noise. Report both rates plus the
mean N=149 hard, and flag if the velocity term produced fast-but-WRONG rules
(high probe velocity, low real N=149) -- that would mean the probe was gameable.
"""
import numpy as np
import trit_evolve as te
import trit_evolve_predict as tp

te.EQUIVARIANT = False
te.DIM = te.IN * te.HID + te.HID + te.HID + 1

_orig_fitness = te.fitness
LAMBDA_V = 0.0                         # set per condition; read by the patched fitness


def combined_fitness(theta, x, maj):
    te.N = 21                          # consensus is measured on the N=21 batch x
    cons = _orig_fitness(theta, x, maj)
    if LAMBDA_V > 0.0:
        # reduce='min' -> a sign-biased flooder (fast on one polarity, 0 on the
        # other) scores 0 and cannot game the reward; only genuine both-polarity
        # signal propagation is rewarded.
        v, _ = tp.signal_velocity(theta, N=35, trials=12, seed=0, reduce="min")   # sets te.N=35
        te.N = 21                      # RESTORE so the next fitness call's ca_run matches x
        return cons + LAMBDA_V * v
    return cons


te.fitness = combined_fitness          # monkeypatch: te.evolve resolves fitness at call time


def outcome_balanced(theta, seed):
    """N=21 solved, and N=149 hard-solve measured SEPARATELY per polarity. A
    sign-biased flooder gets ~half the near-tie cases for free on the mean, so
    the honest generalization criterion is min(hard+, hard-) -- both signs."""
    te.N = 21
    te.rng = np.random.default_rng(7000 + seed)
    x, maj = te.make_configs(1500)
    fin = te.ca_run(theta, x.copy(), steps=42)
    s21 = ((np.sign(fin) == maj[:, None]).mean(axis=1) > 0.9).mean()

    te.N = 149
    te.rng = np.random.default_rng(7000 + seed)
    x, maj = te.make_configs(1500)
    fin = te.ca_run(theta, x.copy(), steps=2 * 149)
    skew = np.abs((x > 0).mean(axis=1) - 0.5)
    hard = ~(skew >= 0.12)
    solved = (np.sign(fin) == maj[:, None]).mean(axis=1) > 0.9
    pos, neg = hard & (maj > 0), hard & (maj < 0)
    hp = solved[pos].mean() if pos.any() else 0.0
    hn = solved[neg].mean() if neg.any() else 0.0
    return s21, float(hp), float(hn)


def run_condition(name, lam_v, K, gens):
    global LAMBDA_V
    LAMBDA_V = lam_v
    rows = []
    print(f"\n########## condition: {name} (lambda_v={lam_v}) ##########", flush=True)
    for seed in range(K):
        te.N = 21
        te.rng = np.random.default_rng(seed)      # seeds BOTH init-mean and config draws -> paired
        theta = te.evolve(gens=gens, lam=40, mu=10)
        v, _ = tp.signal_velocity(theta, N=49, seed=seed, reduce="min")
        te.N = 21
        s21, hp, hn = outcome_balanced(theta, seed)
        hmin = min(hp, hn)
        gen = hmin > 0.30                          # genuine both-polarity generalizer
        rows.append((seed, v, s21, hp, hn, hmin, gen))
        print(f"[{name}] seed {seed:2d}: probe_vel={v:5.2f}  N21={s21*100:4.1f}  "
              f"N149 hard +{hp*100:4.1f}/-{hn*100:4.1f} min={hmin*100:4.1f}  gen={gen}", flush=True)
    return rows


def summarize(name, rows):
    v = np.array([r[1] for r in rows]); hmin = np.array([r[5] for r in rows])
    gen = np.array([float(r[6]) for r in rows])
    hit = gen.mean()
    print(f"  {name:10s}: generalizer hit rate {gen.sum():.0f}/{len(rows)} = {hit*100:4.1f}%   "
          f"mean N149 min-hard {hmin.mean()*100:4.1f}%   mean probe vel(min) {v.mean():.3f}")
    # gaming check: nonzero min-velocity but fails the both-polarity outcome
    gamed = [(r[0], round(r[1], 2), round(r[5], 2)) for r in rows if r[1] > 0.05 and r[5] < 0.30]
    if gamed:
        print(f"    NOTE possible gaming (min-vel>0.05 but min-hard<30%): {gamed}")
    return hit


def main():
    K, gens = 12, 150
    base = run_condition("baseline", 0.0, K, gens)
    treat = run_condition("treatment", 2.0, K, gens)
    print("\n" + "=" * 66)
    print("  CAUSAL TEST: does selecting for velocity PRODUCE generalizers?")
    print("=" * 66)
    hb = summarize("baseline", base)
    ht = summarize("treatment", treat)
    print(f"\n  hit-rate change: {hb*100:.1f}% -> {ht*100:.1f}%  ({(ht-hb)*100:+.1f}pp)")
    print("\n  Honest read: large consistent increase = velocity is CAUSAL (optimizing")
    print("  the predictor produces the outcome). No change = velocity predicts but")
    print("  doesn't steer. Watch the gaming note -- if treatment rules have high probe")
    print("  velocity but low N=149, the term was gamed, not genuinely helpful.")


if __name__ == "__main__":
    main()
