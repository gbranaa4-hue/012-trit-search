# Independent application, round 2: consensus-gate loses against a linear-threshold decision

`trit_order_acceptance_test.py` — a follow-up to
`npc_consensus_findings.md`'s fight/flee win, testing the consensus-gate
primitive against a *structurally different* real decision in the same
game: `tribe/tribemember.gd`'s order-acceptance logic (`give_order()`,
lines 866-891).

## The real code being tested against

```gdscript
var loyalty: int = RANK_LOYALTY.get(current_rank, 0)        # 15-125
var courage: int = PERSONALITIES[personality]["courage"]     # -15 to 40
var drive: int = ORDER_BASE + loyalty + courage               # ORDER_BASE=70
var risk: int = ORDER_RISK.get(kind, 999)                     # 100/130/165
if drive >= risk:
    accept
else:
    refuse
```

Unlike `npc.gd`'s fight/flee chain (a 2-signal OR-gate — information-poor,
no notion of *how* low-hp or *how* outnumbered), this is already a
**weighted-sum threshold over continuous evidence**: loyalty and courage
add up, risk is the bar to clear. Structurally the opposite starting
point from the first test.

## The test

Ground truth: a noisy logistic compliance model built from the *same*
underlying quantities the real formula uses (`loyalty + courage - risk`,
real game constants throughout), so Policy A is, by construction, the
noiseless version of the ground truth's mean — this doesn't artificially
handicap the existing design, which *is* the intended behavior, not an
approximation of something hidden.

**Policy A (current):** `accept if (70 + loyalty + courage) >= risk` —
the real formula, exactly.

**Policy B (consensus-3):** discretize the same three ingredients into
booleans (`loyalty_high = loyalty>=75`, `courage_high = courage>0`,
`risk_low = risk<=130`) and accept if at least 2 of 3 agree.

## Result

| Policy | Accuracy |
|---|---|
| A — current (linear threshold) | **0.8732 ± 0.0023** |
| B — consensus-gate (vote-of-3) | 0.8270 ± 0.0022 |

Gap: **-0.0462** (B loses by 4.62pp). t=-107.31 across 30 seeds. **A beat
B in 30/30 seeds** — even more decisive than the fight/flee win was in
the other direction.

## Why, honestly — and predicted before running

This result was anticipated *before* running the test (see the script's
own docstring), not rationalized after: discretizing continuous evidence
into booleans and voting throws away **margin information** a linear
threshold already uses. A tribesperson with loyalty=124 (just under
Devoted) and one with loyalty=76 (just over Friend) both count as
`loyalty_high=true` under Policy B — identical vote — but Policy A
correctly treats them as meaningfully different amounts of evidence. When
you already have the continuous combination available (as this formula
does), throwing it away to vote on yes/no bins is strictly worse than
using it.

## The contrast with the fight/flee test — the actual finding

| Test | Decision shape | Consensus-gate result |
|---|---|---|
| Fight/flee (`npc.gd`) | 2-signal OR-gate, no magnitude info, **plus a genuinely new signal added** | **Win**, +1.81pp |
| Order acceptance (`tribemember.gd`) | Already a weighted-sum **threshold over continuous evidence** | **Loss**, -4.62pp |

This sharpens the consensus-gate's actual scope, the same way the
resonator-bank test sharpened the symmetry-breaking finding: **the
primitive helps when you're combining genuinely separate, otherwise-
unintegrated signals (especially when you're also adding new
information), and actively hurts when it replaces a decision rule that
already has access to the continuous combination of its inputs.** It is
not a generally-superior decision rule — it is the right tool only when
discretization is the right move, which is precisely *not* the case when
a clean weighted sum is already in hand.

## Verdict

A real, predicted-then-confirmed negative result, recorded with the same
honesty as everything else in this project. Combined with
`npc_consensus_findings.md`, the consensus-gate now has one clean win and
one clean loss against real game logic — exactly the kind of scoped,
falsifiable picture (not "consensus voting is good," but "consensus
voting helps under this specific condition, hurts under that one") this
project's whole resonance/symmetry series has consistently aimed for.

**External validation:** this same scoping rule (weighted combination
wins under calibrated evidence, voting wins under uncalibrated/
contaminated evidence) was then checked against two independent,
externally-published fields — population-coding theory and robust
statistics — using a simulated LIF neuron population. Both predictions
held. See `Spikeling-Project/research/POPULATION_CODING_FINDINGS.md`.
