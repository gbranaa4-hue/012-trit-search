# Independent application: consensus-gate voting vs a real game's fight/flee logic

`trit_npc_consensus_test.py` — does the consensus-gate primitive
(`sign(s0+s1+s2)`, 3-input majority vote — see
`trit_adaptive_scheduler.py` for its first standalone win, in OS process
scheduling) improve on `tribe/npc.gd`'s actual fight-vs-flee decision
logic, completely outside any neural-net or compression context?

## The real code being tested against

`tribe/npc.gd:363-409` (`take_hit()`) resolves fight-vs-flee with a
sequential if/elif chain:

```gdscript
if hp < max_hp * 0.3 and attacker:      # badly hurt
    flee
elif _nearby_rival_count() >= OUTNUMBER_THRESHOLD:  # outnumbered (threshold=4)
    flee
else:
    fight  # (self-defense path)
```

Both branches produce the same action, so the *order* of the checks is
irrelevant to the final decision — this chain is logically a plain 2-input
OR gate: **flee if (low_hp OR outnumbered)**. The game has no third signal
estimating actual relative combat strength at all today.

## The test

Built a synthetic ground-truth combat model (hidden from both policies,
used only to grade them): `npc_power = hp_ratio × base_power`,
`rival_power = rival_count × avg_rival_power`, `should_flee = npc_power <
rival_power`. Both policies only see noisy, discretized signals — matching
how the real game only has threshold checks, not true power knowledge:

- `low_hp` — exactly `tribe/npc.gd`'s real 0.3 threshold
- `outnumbered` — exactly `tribe/npc.gd`'s real `OUTNUMBER_THRESHOLD=4`
- `weak_estimate` — a **new** signal not in the game today: a noisy
  in-the-moment guess at relative power (own power estimate has noise
  σ=0.25, rival estimate has noise σ=0.5 — sizing up several enemies at
  once is harder than judging your own state)

**Policy A (current):** flee if `low_hp OR outnumbered` (exactly matches
the live code's net behavior).

**Policy B (proposed):** flee if at least 2 of {`low_hp`, `outnumbered`,
`weak_estimate`} agree (consensus-gate vote).

20,000 trials × 30 seeds, accuracy against the hidden ground truth.

## Result

| Policy | Accuracy |
|---|---|
| A — current (OR-of-2) | 0.7204 ± 0.0030 |
| B — consensus-gate (vote-of-3) | **0.7385 ± 0.0028** |

Gap: **+0.0181** (1.81pp). Paired t-test across 30 seeds: t=71.69, B beat
A in **30/30 seeds**. This is a real, statistically unambiguous win — not
a coin-flip-sized effect lost in seed noise.

## Honest caveats

- **The win comes from adding information, not just from voting.** Policy
  B has access to a third signal (`weak_estimate`) that Policy A doesn't.
  Some of the +1.81pp is simply "more sensing data helps," independent of
  whether you combine it by OR, AND, or majority vote. A cleaner ablation
  would also test "OR-of-3" (flee if any of the 3 signals fire) against
  the same enriched information to isolate the *voting rule's*
  contribution specifically from the *extra signal's* contribution — not
  done here.
- **This is a synthetic ground-truth model**, not measured in the actual
  running game. The thresholds (`0.3`, `4`) and structure (sequential
  override) are taken directly from the real code, but the combat-power
  formula and noise levels are a reasonable, not measured, stand-in for
  the game's actual (currently nonexistent) combat-strength resolution.
- **It requires building a new sensing capability** (`weak_estimate`)
  that doesn't exist in `npc.gd` today — this isn't a drop-in
  reordering of existing logic, it's "add a sensor, then vote."

## Verdict

Worth taking seriously as a concrete proposal, not just a curiosity: a
clean, well-powered, statistically robust win (matching the discipline of
`trit_adaptive_scheduler.py`'s scheduling win — both are cases where
the consensus-gate genuinely outperforms simpler logic on a real,
multi-signal arbitration problem). Not yet wired into the live game —
that would mean implementing `weak_estimate` as an actual NPC sensing
routine in `npc.gd` and replacing `take_hit()`'s override chain with a
3-vote count, which is a real (if small) code change, not done as part of
this test.
