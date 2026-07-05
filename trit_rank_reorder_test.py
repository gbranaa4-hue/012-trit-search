#!/usr/bin/env python3
"""
Fifth external check of the consensus-gate scoping rule -- the sharper
follow-up that paper/ensemble_ml_findings.md identified but did not run.

The fourth check (trit_ensemble_test.py) came back weak/ambiguous, and its
honest diagnosis was: the generic covariate shift degraded all five
classifiers' ABSOLUTE accuracy together without strongly reordering their
RELATIVE reliability ranking -- and a log-odds weighted combiner mainly
cares about ranking. That sharpened the rule into a falsifiable form:

    The weighted decoder degrades in proportion to how much the shift
    REORDERS relative reliability, not how much it lowers absolute
    accuracy.

This test targets exactly that, with the two regimes the refinement
demands plus the two replication regimes:

REGIME A (no shift)        -- control; replicates check 4.
REGIME B (generic shift)   -- the original magnitude-only shift; replicates
                              check 4's weak result.
REGIME C (rank-reordering) -- a shift SELECTED (from a fixed mechanical
                              candidate family) to make the previously-best
                              validation classifier maximally worse than
                              its peers. Selection uses ONLY train/val
                              data -- test labels are never touched until
                              final scoring.
REGIME D (magnitude-matched control) -- a generic rank-PRESERVING shift
                              tuned (again on validation only) so its mean
                              absolute accuracy damage matches regime C's.
                              This is the control that separates "bigger
                              shift" from "reordering shift."

The decisive contrast is C vs D: same average damage, different ranking
damage. If the refined rule is right, the weighted-minus-majority gap
collapses or reverses in C but survives in D.

RUN 2 NOTE (disclosed change, made after seeing run 1): run 1 confirmed
all six predictions, but regime D failed its own manipulation check --
the generic offset+scale family saturated at a mean validation-accuracy
drop of 0.134 vs regime C's 0.190, leaving a "maybe C just does more
damage" objection open. Run 2 adds REGIME E: drawn from the SAME
candidate family as C, but selected (validation-only, mechanically) for
maximum rank PRESERVATION subject to damage within tolerance of C's --
the exact mirror of C's selection rule. Nothing about A/B/C/D or the
outcome measures was changed; run 1's numbers reproduce identically
under the same seeds.

PREDICTIONS, stated before the first run and not edited after:
  1. Regime A: weighted wins by a small margin (replication, t_A > 0).
  2. Regime B: weighted still wins weakly (replication of check 4).
  3. Regime C: the gap flips -- majority vote wins (t_C < 0).
  4. Regime D: weighted keeps roughly its regime-A/B edge despite damage
     matched to C (t_D > 0 or ~0, clearly above t_C).
  5. The paired contrast gap_D - gap_C is positive and significant.
  6. Across seeds/regimes, the gap correlates positively with the
     Spearman correlation between validation ranking and test-time
     ranking (more rank preservation -> weighted better).

Manipulation checks (reported, not tuned on): test-time rank of the
previously-best classifier per regime, Spearman rho(val ranking, test
ranking) per regime, and mean per-classifier accuracy drop C vs D.

Setup is otherwise identical to trit_ensemble_test.py: same dataset
generator, same 5 diverse classifiers, same clipped log-odds
validation-accuracy weights, same majority-vote baseline.

Run it:
    python trit_rank_reorder_test.py
"""

import numpy as np
from math import sqrt
from scipy.stats import spearmanr
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

N_SEEDS = 30
N_SAMPLES = 4000
N_FEATURES = 20
N_INFORMATIVE = 8

# regime B: the exact generic shift from trit_ensemble_test.py
B_SHIFT_FRAC = 0.4
B_SHIFT_OFFSET = 2.5
B_SHIFT_SCALE = 1.8

# regime C: fixed mechanical candidate family (sizes x transforms), scored
# on the validation split only
C_SUBSET_SIZES = (4, 8, 12)
C_N_RANDOM_SUBSETS = 10          # random subsets tried per (size, transform)

# regime D: generic offset+scale intensity grid searched to match C's mean
# validation-accuracy damage
D_LAMBDA_GRID = np.linspace(0.25, 4.0, 16)
D_SUBSETS_PER_LAMBDA = 8
D_DROP_TOLERANCE = 0.01          # match mean drop within 1pp if possible


def make_classifiers():
    return [
        ("logreg", LogisticRegression(max_iter=500)),
        ("tree", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
        ("nb", GaussianNB()),
        ("rf", RandomForestClassifier(n_estimators=15, max_depth=4, random_state=0)),
    ]


# ---------------------------------------------------------------- shifts

def shift_offset_scale(X, idx, offset, scale):
    Xs = X.copy()
    Xs[:, idx] = Xs[:, idx] * scale + offset
    return Xs


def shift_signflip(X, idx):
    Xs = X.copy()
    Xs[:, idx] = -Xs[:, idx]
    return Xs


def shift_permute(X, idx, rng):
    # destroys the feature-label relationship on those columns while
    # preserving each column's marginal distribution
    Xs = X.copy()
    for j in idx:
        Xs[:, j] = Xs[rng.permutation(Xs.shape[0]), j]
    return Xs


def candidate_shifts_C(rng):
    """The fixed candidate family for regime C: (name, fn) pairs where fn
    maps X -> shifted X. Purely mechanical -- no per-result tuning."""
    cands = []
    for size in C_SUBSET_SIZES:
        for _ in range(C_N_RANDOM_SUBSETS):
            idx = rng.choice(N_FEATURES, size=size, replace=False)
            for offset, scale in [(2.5, 1.8), (4.0, 3.0)]:
                cands.append((
                    f"offscale{offset}/{scale}-k{size}",
                    (lambda X, i=idx, o=offset, s=scale:
                     shift_offset_scale(X, i, o, s)),
                ))
            cands.append((
                f"signflip-k{size}",
                lambda X, i=idx: shift_signflip(X, i),
            ))
            seed_j = int(rng.integers(1 << 31))
            cands.append((
                f"permute-k{size}",
                (lambda X, i=idx, sj=seed_j:
                 shift_permute(X, i, np.random.default_rng(sj))),
            ))
    return cands


# ------------------------------------------------------------- helpers

def accs(fitted, X, y):
    return np.array([np.mean(clf.predict(X) == y) for _, clf in fitted])


def decode(fitted, weights, X, y):
    preds = np.array([clf.predict(X) for _, clf in fitted])
    signed = preds * 2 - 1
    majority = (np.sign(signed.sum(axis=0)) > 0).astype(int)
    weighted = (np.sign((signed * weights[:, None]).sum(axis=0)) > 0).astype(int)
    return np.mean(majority == y), np.mean(weighted == y), preds


def rank_of_best(acc_vec, best_i):
    """1 = best, n = worst, average ranks on ties."""
    order = (-acc_vec).argsort()
    ranks = np.empty(len(acc_vec))
    ranks[order] = np.arange(1, len(acc_vec) + 1)
    return ranks[best_i]


# ------------------------------------------------------------- one seed

def run_seed(seed):
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES, n_informative=N_INFORMATIVE,
        n_redundant=4, n_clusters_per_class=2, flip_y=0.08, class_sep=0.9,
        random_state=seed,
    )
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)

    fitted = []
    for name, clf in make_classifiers():
        clf.fit(X_train, y_train)
        fitted.append((name, clf))

    val_acc = np.clip(accs(fitted, X_val, y_val), 0.05, 0.95)
    weights = np.log(val_acc / (1 - val_acc))
    best_i = int(np.argmax(val_acc))

    # ---- regime B shift: exact original generic shift
    n_shift = int(B_SHIFT_FRAC * N_FEATURES)
    b_idx = rng.choice(N_FEATURES, size=n_shift, replace=False)

    def shift_B(Xs):
        return shift_offset_scale(Xs, b_idx, B_SHIFT_OFFSET, B_SHIFT_SCALE)

    # ---- regime C shift: pick, on VALIDATION only, the candidate that
    # makes the previously-best classifier maximally worse than its peers
    others = [i for i in range(len(fitted)) if i != best_i]
    cand_evals = []   # (name, fn, val accs under shift) -- shared with regime E
    for name, fn in candidate_shifts_C(rng):
        cand_evals.append((name, fn, accs(fitted, fn(X_val), y_val)))
    best_score, shift_C, c_name, c_val_drop = None, None, None, None
    for name, fn, a in cand_evals:
        score = a[best_i] - a[others].mean()   # minimize
        if best_score is None or score < best_score:
            best_score, shift_C, c_name = score, fn, name
            c_val_drop = (val_acc - a).mean()

    # ---- regime D shift: generic offset+scale tuned on VALIDATION to
    # match C's mean accuracy damage while preserving ranking as much as
    # the family allows
    target = c_val_drop
    d_best = None   # (within_tol, rho, -|drop-target|, fn)
    for lam in D_LAMBDA_GRID:
        for _ in range(D_SUBSETS_PER_LAMBDA):
            idx = rng.choice(N_FEATURES, size=8, replace=False)
            fn = (lambda X, i=idx, l=lam:
                  shift_offset_scale(X, i, l * 2.5, 1 + l * 0.8))
            a = accs(fitted, fn(X_val), y_val)
            drop = (val_acc - a).mean()
            rho = spearmanr(val_acc, a).statistic
            if np.isnan(rho):
                rho = 0.0
            key = (abs(drop - target) <= D_DROP_TOLERANCE, rho, -abs(drop - target))
            if d_best is None or key > d_best[0]:
                d_best = (key, fn, drop)
    shift_D, d_val_drop = d_best[1], d_best[2]

    # ---- regime E (run 2): same candidate family as C, mirror selection --
    # maximum rank preservation (Spearman rho vs clean val ranking) subject
    # to mean damage within tolerance of C's, widening tolerance until a
    # candidate qualifies. Validation-only, mechanical, no outcome peeking.
    shift_E, e_val_drop = None, None
    for tol in (0.01, 0.02, 0.04, 0.08, np.inf):
        pool = []
        for name, fn, a in cand_evals:
            drop = (val_acc - a).mean()
            if abs(drop - c_val_drop) <= tol and fn is not shift_C:
                rho = spearmanr(val_acc, a).statistic
                pool.append((0.0 if np.isnan(rho) else rho, drop, fn))
        if pool:
            rho_e, e_val_drop, shift_E = max(pool, key=lambda p: p[0])
            break

    # ---- score all regimes on the test set (first time test labels used)
    out = {"seed": seed, "best_name": fitted[best_i][0],
           "c_shift": c_name, "c_val_drop": c_val_drop, "d_val_drop": d_val_drop,
           "e_val_drop": e_val_drop}
    regimes = {
        "A": X_test,
        "B": shift_B(X_test),
        "C": shift_C(X_test),
        "D": shift_D(X_test),
        "E": shift_E(X_test),
    }
    for r, Xr in regimes.items():
        maj, wtd, _ = decode(fitted, weights, Xr, y_test)
        test_acc = accs(fitted, Xr, y_test)
        rho = spearmanr(val_acc, test_acc).statistic
        out[f"{r}_majority"] = maj
        out[f"{r}_weighted"] = wtd
        out[f"{r}_rho"] = 0.0 if np.isnan(rho) else rho
        out[f"{r}_bestrank"] = rank_of_best(test_acc, best_i)
        out[f"{r}_meandrop"] = (val_acc - test_acc).mean()
    return out


# ----------------------------------------------------------------- main

def paired_t(diffs):
    diffs = np.asarray(diffs)
    return diffs.mean() / (diffs.std(ddof=1) / sqrt(len(diffs)))


def main():
    print("Rank-reordering test: does the weighted decoder break when the shift")
    print("reorders classifier reliability, at matched absolute damage?")
    print(f"N_SEEDS={N_SEEDS}  classifiers=[logreg, tree, knn, nb, rf]\n")

    results = [run_seed(s) for s in range(N_SEEDS)]

    labels = {
        "A": "A (no shift)",
        "B": "B (generic shift)",
        "C": "C (rank-reordering)",
        "D": "D (damage-matched)",
        "E": "E (mirror control)",
    }
    print(f"{'Regime':<24}{'Majority':<20}{'Weighted':<20}{'Gap':<12}{'t':<8}")
    print("-" * 84)
    gaps = {}
    for r in "ABCDE":
        m = np.array([x[f"{r}_majority"] for x in results])
        w = np.array([x[f"{r}_weighted"] for x in results])
        gaps[r] = w - m
        print(f"{labels[r]:<24}{m.mean():>7.4f} +/- {m.std():<8.4f}"
              f"{w.mean():>7.4f} +/- {w.std():<8.4f}"
              f"{gaps[r].mean():>+9.4f}   {paired_t(gaps[r]):>+6.2f}")

    print("\nManipulation checks (means over seeds):")
    print(f"{'Regime':<24}{'rho(val,test) rank':<22}{'best clf test rank':<22}{'mean acc drop':<14}")
    print("-" * 84)
    for r in "ABCDE":
        rho = np.mean([x[f"{r}_rho"] for x in results])
        br = np.mean([x[f"{r}_bestrank"] for x in results])
        dr = np.mean([x[f"{r}_meandrop"] for x in results])
        print(f"{labels[r]:<24}{rho:>10.3f}{br:>20.2f}{dr:>20.4f}")
    frac_flipped = np.mean([x["C_bestrank"] >= 4 for x in results])
    print(f"\nRegime C: previously-best classifier lands in bottom-2 at test in "
          f"{frac_flipped:.0%} of seeds")
    print(f"Val-side damage match: C drop {np.mean([x['c_val_drop'] for x in results]):.4f} "
          f"vs D drop {np.mean([x['d_val_drop'] for x in results]):.4f} "
          f"vs E drop {np.mean([x['e_val_drop'] for x in results]):.4f}")

    contrast = gaps["D"] - gaps["C"]
    print(f"\nContrast gap_D - gap_C: {contrast.mean():+.4f}  "
          f"t={paired_t(contrast):+.2f}  (prediction: positive, significant)")
    contrast_e = gaps["E"] - gaps["C"]
    print(f"Contrast gap_E - gap_C: {contrast_e.mean():+.4f}  "
          f"t={paired_t(contrast_e):+.2f}  (run-2 decisive contrast: same family, "
          f"matched damage, opposite ranking)")

    # prediction 6: pooled across shifted regimes, gap should track rank
    # preservation
    rhos = np.concatenate([[x[f"{r}_rho"] for x in results] for r in "BCDE"])
    gs = np.concatenate([gaps[r] for r in "BCDE"])
    pr = spearmanr(rhos, gs)
    print(f"Pooled (B,C,D,E) Spearman corr between rank-preservation rho and gap: "
          f"rho={pr.statistic:+.3f} p={pr.pvalue:.2g}  (prediction: positive)")

    print("\nPredictions (stated in header before first run):")
    print("  t_A > 0, t_B > 0 weakly, t_C < 0, t_D >> t_C, contrast positive,")
    print("  gap correlates positively with rank preservation.")


if __name__ == "__main__":
    main()
