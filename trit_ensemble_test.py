#!/usr/bin/env python3
"""
Fourth external check of the consensus-gate scoping rule (weighted
combination wins under calibrated evidence, voting wins under
uncalibrated/shifted evidence) -- this time against real scikit-learn
classifiers and a real (if synthetic) dataset, not another i.i.d.
Bernoulli simulation like trit_tmr_test.py. Genuinely different failure
mode too: TMR/population-coding tested independent or outlier-corrupted
per-unit noise; this tests CORRELATED degradation from covariate shift,
where several classifiers can degrade together because they share
sensitivity to the same shifted features -- a much more realistic
real-world failure mode for an ML ensemble than independent random noise.

BACKGROUND (real, established field): ensemble learning has the same
two established results as the other three substrates --
  - hard/majority VOTING across diverse base classifiers, with no
    confidence weighting, is the classical robust baseline (no
    assumption about which classifier is more trustworthy);
  - WEIGHTED voting (weight by validation-set accuracy, equivalently a
    log-odds combiner) is provably better when those validation-time
    accuracy estimates still reflect test-time reality.
The open question in the ML literature -- ensemble weights tuned on a
validation set can become stale/overconfident under distribution shift
-- is exactly this project's scoping rule, independently arrived at.

SETUP: 5 diverse base classifiers (LogisticRegression, DecisionTree,
KNN, GaussianNB, shallow RandomForest) trained on one split, with
ensemble weights CALIBRATED from a held-out validation split (not the
training labels directly -- log-odds of validation accuracy).

REGIME A (stable): test set drawn i.i.d. from the same distribution as
train/validation.
REGIME B (shifted): test set has a covariate shift applied to a subset
of features (offset + rescale) -- unknown to the ensemble, and likely to
hurt distance/scale-sensitive classifiers (KNN) more than others,
producing correlated, uneven degradation rather than independent noise.

DECODER 1 (hard/majority vote): unweighted vote across classifiers.
DECODER 2 (weighted vote): validation-accuracy log-odds weighted vote.

Prediction, stated before running: weighted wins regime A, majority vote
wins or closes the gap in regime B.

Run it:
    python trit_ensemble_test.py
"""

import numpy as np
from math import sqrt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

N_SEEDS = 20
N_SAMPLES = 4000
N_FEATURES = 20
N_INFORMATIVE = 8
SHIFT_FRAC = 0.4    # fraction of features covariate-shifted in regime B
SHIFT_OFFSET = 2.5
SHIFT_SCALE = 1.8


def make_classifiers():
    return [
        ("logreg", LogisticRegression(max_iter=500)),
        ("tree", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
        ("nb", GaussianNB()),
        ("rf", RandomForestClassifier(n_estimators=15, max_depth=4, random_state=0)),
    ]


def apply_shift(X, shift_idx, rng):
    Xs = X.copy()
    Xs[:, shift_idx] = Xs[:, shift_idx] * SHIFT_SCALE + SHIFT_OFFSET
    return Xs


def run_seed(seed):
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES, n_informative=N_INFORMATIVE,
        n_redundant=4, n_clusters_per_class=2, flip_y=0.08, class_sep=0.9,
        random_state=seed,
    )
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)

    classifiers = make_classifiers()
    fitted = []
    val_preds = []
    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        fitted.append((name, clf))
        val_preds.append(clf.predict(X_val))
    val_preds = np.array(val_preds)   # (n_clf, n_val)

    val_acc = np.clip((val_preds == y_val[None, :]).mean(axis=1), 0.05, 0.95)
    weights = np.log(val_acc / (1 - val_acc))

    n_shift = int(SHIFT_FRAC * N_FEATURES)
    shift_idx = rng.choice(N_FEATURES, size=n_shift, replace=False)
    X_test_shifted = apply_shift(X_test, shift_idx, rng)

    results = {}
    for regime, Xr, yr in [("A", X_test, y_test), ("B", X_test_shifted, y_test)]:
        preds = np.array([clf.predict(Xr) for _, clf in fitted])   # (n_clf, n_test), 0/1
        signed = preds * 2 - 1   # -1/+1

        majority_pred = (np.sign(signed.sum(axis=0)) > 0).astype(int)
        weighted_pred = (np.sign((signed * weights[:, None]).sum(axis=0)) > 0).astype(int)

        results[f"{regime}_majority"] = np.mean(majority_pred == yr)
        results[f"{regime}_weighted"] = np.mean(weighted_pred == yr)

    return results


def main():
    print("Ensemble-ML test: hard/majority voting vs validation-weighted voting")
    print(f"N_SEEDS={N_SEEDS}  classifiers=[logreg, tree, knn, nb, rf]  shift_frac={SHIFT_FRAC}\n")

    all_results = [run_seed(s) for s in range(N_SEEDS)]
    keys = ["A_majority", "A_weighted", "B_majority", "B_weighted"]
    means = {k: np.mean([r[k] for r in all_results]) for k in keys}
    stds = {k: np.std([r[k] for r in all_results]) for k in keys}

    print(f"{'Regime':<22}{'Majority vote':<22}{'Weighted (val-acc)':<22}{'Gap (weighted-maj)':<18}")
    print("-" * 84)
    for regime, label in [("A", "A (no shift)"), ("B", "B (covariate shift)")]:
        m, w = means[f"{regime}_majority"], means[f"{regime}_weighted"]
        msd, wsd = stds[f"{regime}_majority"], stds[f"{regime}_weighted"]
        gap = w - m
        print(f"{label:<22}{m:>7.4f} +/- {msd:<10.4f}{w:>7.4f} +/- {wsd:<10.4f}{gap:>+12.4f}")

    gap_a = np.array([r["A_weighted"] - r["A_majority"] for r in all_results])
    gap_b = np.array([r["B_weighted"] - r["B_majority"] for r in all_results])
    t_a = gap_a.mean() / (gap_a.std(ddof=1) / sqrt(N_SEEDS))
    t_b = gap_b.mean() / (gap_b.std(ddof=1) / sqrt(N_SEEDS))
    print(f"\nRegime A paired t (weighted-majority): t={t_a:.2f}")
    print(f"Regime B paired t (weighted-majority): t={t_b:.2f}")
    print("\nPrediction: weighted should win A (t_A positive), majority should win or close the gap in B (t_B smaller/negative)")


if __name__ == "__main__":
    main()
