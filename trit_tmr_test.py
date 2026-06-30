#!/usr/bin/env python3
"""
Fourth test of the consensus-gate scoping rule, this time against
classical fault-tolerant systems engineering -- Triple (here, N-way)
Modular Redundancy -- rather than another internal replication.

BACKGROUND (real, established field): TMR / N-modular redundancy combines
several independent, replicated units (sensors, processors) by majority
vote specifically because no per-unit confidence/reliability model is
assumed -- von Neumann's classical TMR reliability formula
R_TMR = 3R^2 - 2R^3 treats all units as equally reliable and shows
majority voting only helps when R > 0.5. This project's hardware
(hardware/consensus_gate.sv, 36/36 testbench pass) implements exactly
this majority-vote primitive in synthesizable RTL.

A separate, also-established result (weighted majority voting /
Bayesian sensor fusion for independent binary classifiers with KNOWN,
unequal error rates): a log-odds-weighted combiner is provably optimal
(MAP-optimal) when each unit's reliability is accurately known --
strictly better than plain majority vote whenever units are
heterogeneous.

These two results, read together, predict exactly the scoping rule
already found in this project's game-logic and population-coding tests:
weighted combination wins when reliability is accurately calibrated;
plain majority vote (egalitarian, no per-unit trust) is more robust when
calibration goes stale -- a 'confidently wrong' weighted combiner can be
worse than ignoring its (outdated) weights entirely.

SETUP: N_UNITS=5 independent binary units, each correct with probability
r_i (heterogeneous, calibrated once from finite calibration data, so the
weighted decoder's weights are realistic ESTIMATES, not ground truth).

REGIME A (stable): test-time true reliabilities match calibration.
REGIME B (drifted): test-time true reliabilities silently shift from
their calibrated values (representing real-world sensor/component
degradation or miscalibration) -- neither decoder is told this happened.

DECODER 1 (majority vote / classic TMR): sign(sum of unit votes), no
weighting at all.
DECODER 2 (weighted / log-odds Bayesian fusion): sign(sum of
log-odds-weighted votes, weights from calibration).

Prediction (stated before running): weighted wins regime A, majority
vote wins or closes the gap in regime B.

Run it:
    python trit_tmr_test.py
"""

import numpy as np
from math import sqrt, log

N_UNITS = 5
N_CALIB_TRIALS = 3000
N_TEST_TRIALS = 8000
N_SEEDS = 25

DRIFT_STD = 0.25   # how much a unit's true reliability can silently shift by test time


def sample_reliabilities(rng):
    return rng.uniform(0.55, 0.95, size=N_UNITS)


def run_trials(rng, true_r, n_trials):
    """Each unit is independently correct w.p. true_r[i]. Returns
    (ground_truth bits, unit votes), votes are +1/-1 matching truth or not."""
    truth = rng.integers(0, 2, size=n_trials) * 2 - 1   # +1/-1
    correct_mask = rng.uniform(size=(n_trials, N_UNITS)) < true_r[None, :]
    votes = np.where(correct_mask, truth[:, None], -truth[:, None])
    return truth, votes


def calibrate(rng, true_r):
    """Estimate each unit's reliability from finite calibration data (not
    ground truth) and compute log-odds weights, clipped to avoid infinities."""
    truth, votes = run_trials(rng, true_r, N_CALIB_TRIALS)
    correct = (votes == truth[:, None])
    r_hat = np.clip(correct.mean(axis=0), 0.05, 0.95)
    weights = np.log(r_hat / (1 - r_hat))
    return weights


def decode(votes, weights):
    majority_pred = np.sign(np.sum(votes, axis=1))
    weighted_pred = np.sign(np.sum(votes * weights[None, :], axis=1))
    return majority_pred, weighted_pred


def run_seed(seed):
    rng = np.random.default_rng(seed)
    true_r_calib = sample_reliabilities(rng)
    weights = calibrate(rng, true_r_calib)

    results = {}

    # Regime A: stable, same reliabilities as calibration
    truth_a, votes_a = run_trials(rng, true_r_calib, N_TEST_TRIALS)
    maj_a, wt_a = decode(votes_a, weights)
    results["A_majority"] = np.mean(maj_a == truth_a)
    results["A_weighted"] = np.mean(wt_a == truth_a)

    # Regime B: drifted, true reliabilities silently shift, decoder unaware
    true_r_drift = np.clip(true_r_calib + rng.normal(0, DRIFT_STD, size=N_UNITS), 0.05, 0.95)
    truth_b, votes_b = run_trials(rng, true_r_drift, N_TEST_TRIALS)
    maj_b, wt_b = decode(votes_b, weights)
    results["B_majority"] = np.mean(maj_b == truth_b)
    results["B_weighted"] = np.mean(wt_b == truth_b)

    return results


def main():
    print("TMR test: classic majority vote vs log-odds weighted (Bayesian) fusion")
    print(f"N_UNITS={N_UNITS}  N_SEEDS={N_SEEDS}  N_TEST_TRIALS/regime={N_TEST_TRIALS}  drift_std={DRIFT_STD}\n")

    all_results = [run_seed(s) for s in range(N_SEEDS)]
    keys = ["A_majority", "A_weighted", "B_majority", "B_weighted"]
    means = {k: np.mean([r[k] for r in all_results]) for k in keys}
    stds = {k: np.std([r[k] for r in all_results]) for k in keys}

    print(f"{'Regime':<22}{'Majority vote':<22}{'Weighted (log-odds)':<22}{'Gap (weighted-maj)':<18}")
    print("-" * 84)
    for regime, label in [("A", "A (stable)"), ("B", "B (drifted)")]:
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
    print("\nPrediction: weighted should win A (t_A positive, large), majority should win or close the gap in B (t_B smaller/negative)")


if __name__ == "__main__":
    main()
