#!/usr/bin/env python3
"""
Second independent-application test: does consensus-gate voting improve
on tribe/tribemember.gd's order-acceptance logic (give_order(),
lines 866-891) -- a DIFFERENT shape of decision from the fight/flee test
in trit_npc_consensus_test.py.

The real code (grounded exactly, including real constants from
tribemember.gd):

    RANK_LOYALTY = {Stranger:15, Acquaintance:45, Friend:75, Loyal:100, Devoted:125}
    PERSONALITIES courage = {Steady:0, Trusting:15, Wary:-15, Brave:40, Greedy:-5}
    ORDER_BASE = 70
    ORDER_RISK = {gather:100, hunt:130, scout:165, wood:100}

    drive = ORDER_BASE + loyalty + courage
    accept if drive >= risk

Unlike npc.gd's fight/flee chain (a 2-signal OR-gate, information-poor),
this is already a WEIGHTED-SUM threshold over continuous evidence
(loyalty + courage compared to risk) -- structurally the opposite
starting point. The hypothesis worth testing honestly: does discretizing
that continuous evidence into 3 booleans and voting (as worked for the
OR-gate case) help here too, or does it lose information a linear
threshold already captures?

GROUND TRUTH (hidden from both policies): a noisy logistic compliance
model built from the SAME underlying continuous quantities the real
formula uses (loyalty + courage - risk), so Policy A (the real formula)
is, by construction, the noiseless version of the ground truth's mean --
this is the fair, honest setup: it doesn't artificially handicap the
existing design, which IS the intended behavior, not an approximation of
something else.

POLICY A -- CURRENT (linear threshold, exact real formula):
    accept if (ORDER_BASE + loyalty + courage) >= risk

POLICY B -- CONSENSUS-3 (discretize the same 3 ingredients into booleans,
vote):
    loyalty_high = loyalty >= 75          (Friend rank or above)
    courage_high = courage > 0
    risk_low     = risk <= 130            (gather/hunt, not scout)
    accept if at least 2 of {loyalty_high, courage_high, risk_low}

Run it:
    python trit_order_acceptance_test.py
"""

import numpy as np

N_TRIALS = 20000
N_SEEDS = 30

RANK_LOYALTY = {"Stranger": 15, "Acquaintance": 45, "Friend": 75, "Loyal": 100, "Devoted": 125}
COURAGE = {"Steady": 0, "Trusting": 15, "Wary": -15, "Brave": 40, "Greedy": -5}
ORDER_BASE = 70
ORDER_RISK = {"gather": 100, "hunt": 130, "scout": 165, "wood": 100}

RANKS = list(RANK_LOYALTY.keys())
PERSONALITIES = list(COURAGE.keys())
ORDERS = list(ORDER_RISK.keys())

NOISE_STD = 25.0   # compliance has real-world noise the deterministic formula doesn't model


def run_trial(rng):
    rank = RANKS[rng.integers(0, len(RANKS))]
    personality = PERSONALITIES[rng.integers(0, len(PERSONALITIES))]
    order = ORDERS[rng.integers(0, len(ORDERS))]

    loyalty = RANK_LOYALTY[rank]
    courage = COURAGE[personality]
    risk = ORDER_RISK[order]

    drive = ORDER_BASE + loyalty + courage
    margin = drive - risk

    # ground truth: same underlying margin, but real compliance has noise
    # the deterministic formula doesn't capture (mood, fatigue, etc.)
    true_margin = margin + rng.normal(0, NOISE_STD)
    should_accept = true_margin > 0

    policy_a_accept = drive >= risk   # exact real formula, no noise

    loyalty_high = loyalty >= 75
    courage_high = courage > 0
    risk_low = risk <= 130
    policy_b_accept = sum([loyalty_high, courage_high, risk_low]) >= 2

    return should_accept, policy_a_accept, policy_b_accept


def score_seed(seed):
    rng = np.random.default_rng(seed)
    correct_a = correct_b = 0
    for _ in range(N_TRIALS):
        truth, a, b = run_trial(rng)
        correct_a += (a == truth)
        correct_b += (b == truth)
    return correct_a / N_TRIALS, correct_b / N_TRIALS


def main():
    print("Order-acceptance test: current linear-threshold formula vs consensus-gate vote-of-3")
    print("Grounded in tribe/tribemember.gd:117-121,866-891 (real RANK_LOYALTY/COURAGE/ORDER_RISK values)")
    print(f"N_SEEDS={N_SEEDS}  N_TRIALS per seed={N_TRIALS}  compliance noise std={NOISE_STD}\n")

    a_scores, b_scores = [], []
    for s in range(N_SEEDS):
        a, b = score_seed(s)
        a_scores.append(a)
        b_scores.append(b)
    a_scores = np.array(a_scores)
    b_scores = np.array(b_scores)

    print(f"Policy A (current, linear threshold): accuracy = {a_scores.mean():.4f} +/- {a_scores.std():.4f}")
    print(f"Policy B (consensus-gate, 2-of-3):     accuracy = {b_scores.mean():.4f} +/- {b_scores.std():.4f}")
    gap = b_scores.mean() - a_scores.mean()
    print(f"Gap (B - A): {gap:+.4f}")

    diffs = b_scores - a_scores
    from math import sqrt
    t = diffs.mean() / (diffs.std(ddof=1) / sqrt(len(diffs)))
    print(f"Paired diff across seeds: mean={diffs.mean():+.4f}  t={t:.2f}")
    print(f"A beat B in {int((diffs < 0).sum())}/{N_SEEDS} seeds, B beat A in {int((diffs > 0).sum())}/{N_SEEDS} seeds")


if __name__ == "__main__":
    main()
