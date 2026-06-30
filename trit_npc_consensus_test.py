#!/usr/bin/env python3
"""
Independent application test: does the consensus-gate (3-signal majority
vote) primitive improve on a real game's existing fight/flee decision
logic?

Grounded in tribe/npc.gd's actual code (take_hit(), lines 363-409):
the NPC currently resolves fight-vs-flee through a sequential if/elif
override chain:

    if hp_ratio < 0.3: flee                      # badly hurt
    elif nearby_rival_count >= OUTNUMBER_THRESHOLD (4): flee   # outnumbered
    else: fight

Because both branches produce the SAME action (flee), the order of the
checks doesn't matter for the final decision -- this chain is logically
just a 2-input OR gate: flee if (low_hp OR outnumbered). The game has no
third "am I actually weaker" signal at all today.

This script asks the concrete, falsifiable question the consensus-gate
work (trit_adaptive_scheduler.py) suggests: if you add ONE more
independent noisy signal (an estimate of relative combat strength) and
combine all three via majority vote (consensus-gate, flee if >=2 of 3
agree) instead of plain OR, does that produce better real decisions
against a hidden ground truth -- or does it just add noise?

GROUND TRUTH (hidden from both policies, used only to grade them):
    npc_power   = hp_ratio * base_power_self      (base_power_self ~ N(1,0.15))
    rival_power = rival_count * avg_rival_power    (avg_rival_power ~ U(0.5,1.3))
    should_flee = npc_power < rival_power

Both policies see only NOISY, DISCRETIZED versions of the underlying
state (matching how the real game only has hp/rival-count thresholds,
not true combat-power knowledge):
    low_hp        = hp_ratio < 0.3                              (matches real threshold)
    outnumbered   = rival_count >= 4                             (matches OUTNUMBER_THRESHOLD)
    weak_estimate = (noisy estimate of npc_power < rival_power)  (NEW signal, not in the game today)

POLICY A -- CURRENT (OR-of-2, matches npc.gd exactly):
    flee if low_hp OR outnumbered

POLICY B -- CONSENSUS-3 (majority vote over all 3 signals, the proposed change):
    flee if at least 2 of {low_hp, outnumbered, weak_estimate} are true

Run it:
    python trit_npc_consensus_test.py
"""

import numpy as np

N_TRIALS = 20000
N_SEEDS = 30

OUTNUMBER_THRESHOLD = 4   # tribe/npc.gd:86
LOW_HP_FRAC = 0.3         # tribe/npc.gd:371


def run_trial(rng):
    hp_ratio = rng.uniform(0.0, 1.0)
    rival_count = rng.integers(0, 8)
    base_power_self = max(0.1, rng.normal(1.0, 0.15))
    avg_rival_power = rng.uniform(0.5, 1.3)

    npc_power = hp_ratio * base_power_self
    rival_power = rival_count * avg_rival_power
    should_flee = npc_power < rival_power

    low_hp = hp_ratio < LOW_HP_FRAC
    outnumbered = rival_count >= OUTNUMBER_THRESHOLD

    # the NPC doesn't have true power values -- only a noisy guess,
    # representing imperfect in-game "sizing up" of the situation
    est_self = npc_power + rng.normal(0, 0.25)
    est_rival = rival_power + rng.normal(0, 0.5)
    weak_estimate = est_self < est_rival

    policy_a_flee = low_hp or outnumbered
    policy_b_flee = sum([low_hp, outnumbered, weak_estimate]) >= 2

    return should_flee, policy_a_flee, policy_b_flee


def score_seed(seed):
    rng = np.random.default_rng(seed)
    correct_a = 0
    correct_b = 0
    survived_a = 0   # "survived" proxy: correct flee OR correctly chose to fight
    survived_b = 0
    for _ in range(N_TRIALS):
        truth, a, b = run_trial(rng)
        correct_a += (a == truth)
        correct_b += (b == truth)
    return correct_a / N_TRIALS, correct_b / N_TRIALS


def main():
    print("NPC fight/flee decision test: current OR-of-2 logic vs consensus-gate vote-of-3")
    print(f"Grounded in tribe/npc.gd:363-409 (take_hit), OUTNUMBER_THRESHOLD={OUTNUMBER_THRESHOLD}, low_hp<{LOW_HP_FRAC}")
    print(f"N_SEEDS={N_SEEDS}  N_TRIALS per seed={N_TRIALS}\n")

    a_scores, b_scores = [], []
    for s in range(N_SEEDS):
        a, b = score_seed(s)
        a_scores.append(a)
        b_scores.append(b)
    a_scores = np.array(a_scores)
    b_scores = np.array(b_scores)

    print(f"Policy A (current, OR-of-2):    accuracy = {a_scores.mean():.4f} +/- {a_scores.std():.4f}")
    print(f"Policy B (consensus-gate, 2-of-3): accuracy = {b_scores.mean():.4f} +/- {b_scores.std():.4f}")
    gap = b_scores.mean() - a_scores.mean()
    print(f"Gap (B - A): {gap:+.4f}")

    # paired t-test across seeds
    diffs = b_scores - a_scores
    from math import sqrt
    t = diffs.mean() / (diffs.std(ddof=1) / sqrt(len(diffs)))
    print(f"Paired diff across seeds: mean={diffs.mean():+.4f}  t={t:.2f}  (|t|>2 ~ significant at n={N_SEEDS})")
    print(f"B beat A in {int((diffs > 0).sum())}/{N_SEEDS} seeds")


if __name__ == "__main__":
    main()
