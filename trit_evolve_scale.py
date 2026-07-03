"""
Push the signal-velocity mechanism two ways, both falsifiable:

1. OUT-OF-SAMPLE SCALE. Velocity was shown to predict N=149 generalization.
   If it is really the mechanism it must predict at scales never looked at.
   Evaluate the 16 saved rules at N=149, 299, 499 and check velocity still
   ranks them -- AND whether the 3 winners (velocity 0.12) themselves survive
   or collapse at larger N (0.12 cells/step may be too slow to cross a big ring).

2. SAME BASIN? The 3 winners all landed at IDENTICAL velocity 0.12. The rule is
   a pure function of a 7-cell ternary neighborhood, so its COMPLETE behavior is
   a 3^7 = 2187-entry lookup table. Compute each rule's exact table and measure
   pairwise agreement: are the 3 winners literally the same CA (one solution
   basin), or different rules that happen to share a velocity?

Reuses the 16 evolved rules saved by trit_evolve_predict.py (predict_theta_*.npy).
No re-evolution.
"""
import itertools
import numpy as np
import trit_evolve as te
import trit_evolve_predict as tp

te.EQUIVARIANT = False
te.DIM = te.IN * te.HID + te.HID + te.HID + 1

WINNERS = [0, 4, 13]                                   # nonzero-velocity seeds from the 16-seed run
ALL = list(range(16))
NB = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=te.IN)))   # (2187, 7)


def rule_table(theta):
    """Exact, complete behavior: output on every possible 7-cell neighborhood."""
    W1, b1, W2, b2 = te.unpack(theta)
    h = np.tanh(NB @ W1 + b1)
    o = np.tanh(h @ W2 + b2)
    return te.ternary(o).ravel()                       # (2187,) in {-1,0,1}


def eval_hard(theta, N, nconf, seed):
    te.N = N
    te.rng = np.random.default_rng(9000 + seed)
    x, maj = te.make_configs(nconf)
    fin = te.ca_run(theta, x.copy(), steps=2 * N)
    skew = np.abs((x > 0).mean(axis=1) - 0.5)
    hard = ~(skew >= 0.12)
    return ((np.sign(fin[hard]) == maj[hard, None]).mean(axis=1) > 0.9).mean()


def main():
    thetas = {s: np.load(f"predict_theta_{s}.npy") for s in ALL}
    vel = {s: tp.signal_velocity(thetas[s], N=49, seed=s)[0] for s in ALL}

    # ---- 1. out-of-sample scale ----
    scales = [(149, 900), (299, 500), (499, 300)]
    hard = {N: {s: eval_hard(thetas[s], N, nc, s) for s in ALL} for N, nc in scales}

    print("=" * 68)
    print("  OUT-OF-SAMPLE: does small-N velocity predict at larger rings?")
    print("=" * 68)
    vv = np.array([vel[s] for s in ALL])
    for N, _ in scales:
        hv = np.array([hard[N][s] for s in ALL])
        c = np.corrcoef(vv, hv)[0, 1] if hv.std() > 1e-9 else float("nan")
        wmean = np.mean([hard[N][s] for s in WINNERS]) * 100
        lmean = np.mean([hard[N][s] for s in ALL if s not in WINNERS]) * 100
        print(f"  N={N:3d}: corr(vel,hard)={c:+.3f}   winners {wmean:4.1f}%   others {lmean:4.1f}%   "
              f"gap {wmean-lmean:+4.1f}pp")
    print("\n  per-winner hard-solve degradation with scale:")
    print(f"    {'seed':>5} {'vel':>5} " + " ".join(f"N{N:<4}" for N, _ in scales))
    for s in WINNERS:
        print(f"    {s:>5} {vel[s]:>5.2f} " + " ".join(f"{hard[N][s]*100:4.0f} " for N, _ in scales))

    # ---- 2. same basin? exact 3^7 rule-table agreement ----
    tables = {s: rule_table(thetas[s]) for s in ALL}

    def agree(a, b):                                   # permutation-invariant: exact function match
        return (tables[a] == tables[b]).mean()

    win_pairs = [(a, b) for i, a in enumerate(WINNERS) for b in WINNERS[i + 1:]]
    los = [s for s in ALL if s not in WINNERS]
    los_pairs = [(a, b) for i, a in enumerate(los) for b in los[i + 1:]]
    cross = [(a, b) for a in WINNERS for b in los]

    def mean_agree(pairs):
        return np.mean([agree(a, b) for a, b in pairs])

    print("\n" + "=" * 68)
    print("  SAME BASIN? exact 3^7 rule-table agreement (1.0 = identical CA)")
    print("=" * 68)
    print(f"  winner-winner : {mean_agree(win_pairs):.3f}   (are the 3 generalizers one rule?)")
    print(f"  winner-loser  : {mean_agree(cross):.3f}")
    print(f"  loser-loser   : {mean_agree(los_pairs):.3f}")
    print("\n  winner-winner pairwise:")
    for a, b in win_pairs:
        print(f"    seed {a} vs seed {b}: {agree(a, b):.3f}")

    print("\n  Honest read: if winner-winner >> winner-loser, the 3 generalizers are")
    print("  the SAME discrete solution (one basin evolution rarely finds). If")
    print("  winner-winner ~ loser-loser, 'velocity 0.12' is a shared property of")
    print("  DIFFERENT rules, not a single basin.")


if __name__ == "__main__":
    main()
