# Fifth check: the rank-reordering test the ensemble study asked for

**Script:** `trit_rank_reorder_test.py` · **Seeds:** 30 · **Date:** 2026-07-05

## What this is

`paper/ensemble_ml_findings.md` ended with an identified-but-not-run
follow-up: the fourth check's generic covariate shift produced a weak,
ambiguous result, and the honest diagnosis was that the shift degraded
all five classifiers' *absolute* accuracy together without reordering
their *relative* reliability ranking — the thing a log-odds weighted
combiner actually depends on. That diagnosis sharpened the consensus-gate
scoping rule into its most falsifiable form yet:

> The weighted decoder degrades in proportion to how much the shift
> **reorders relative reliability**, not how much it lowers absolute
> accuracy.

This test runs the manipulation that diagnosis called for: a shift
deliberately engineered to make the previously-best validation
classifier the worst test-time classifier — plus the control that makes
the comparison mean something, a shift with **matched absolute damage
but preserved ranking**.

## Setup

Identical to the fourth check (`trit_ensemble_test.py`): same synthetic
dataset generator, same five diverse classifiers (logreg, tree, KNN,
GaussianNB, shallow RF), same train/validation/test split, same clipped
log-odds validation-accuracy weights, same unweighted majority-vote
baseline. 30 seeds.

Five regimes, all scored on the same test labels, which are never
touched until final scoring:

- **A (no shift)** — replication control.
- **B (generic shift)** — the fourth check's exact offset+rescale shift.
- **C (rank-reordering)** — selected from a fixed mechanical candidate
  family (offset+scale / sign-flip / column-permute on random feature
  subsets of size 4/8/12) to minimize `acc(best) − mean(acc(others))`
  **on the validation split only**.
- **D (damage-matched, generic)** — offset+scale intensity grid-searched
  on validation to match C's mean accuracy damage.
- **E (mirror control)** — drawn from the **same candidate family as C**,
  selected for maximum rank *preservation* (Spearman ρ vs the clean
  validation ranking) subject to damage within tolerance of C's. The
  exact mirror image of C's selection rule.

Six predictions were written in the script header before the first run
and not edited after (see `trit_rank_reorder_test.py`).

**Run-2 disclosure:** regime E was added *after* seeing run 1, because
regime D failed its own manipulation check — the generic offset+scale
family saturated at a mean validation drop of 0.134 vs C's 0.190,
leaving a "maybe C just does more damage" objection open. Nothing about
regimes A–D or the outcome measures changed, and their numbers
reproduce identically under the same seeds. E's selection is
validation-only and mechanical, like C's.

## Result — the prediction holds, cleanly

| Regime | Majority vote | Weighted (val-acc) | Gap | paired t |
|---|---|---|---|---|
| A (no shift) | 0.8234 ± 0.0418 | 0.8319 ± 0.0327 | **+0.0085** | +3.17 |
| B (generic shift) | 0.7749 ± 0.0542 | 0.7808 ± 0.0516 | **+0.0059** | +2.62 |
| C (rank-reordering) | 0.6347 ± 0.1163 | 0.6316 ± 0.1171 | **−0.0030** | **−2.70** |
| D (damage-matched) | 0.7101 ± 0.0706 | 0.7148 ± 0.0683 | **+0.0046** | +2.75 |
| E (mirror control) | 0.6339 ± 0.1181 | 0.6388 ± 0.1190 | **+0.0050** | +1.95 |

Manipulation checks (means over seeds):

| Regime | ρ(val rank, test rank) | test rank of prev-best clf | mean acc drop |
|---|---|---|---|
| A | +0.916 | 1.00 | 0.002 |
| B | +0.736 | 1.10 | 0.062 |
| C | **−0.391** | **4.83 / 5** | 0.193 |
| D | +0.834 | 1.30 | 0.133 |
| E | +0.531 | 1.77 | 0.189 |

The manipulation worked: in regime C the previously-best classifier
landed in the bottom two at test time in **30/30 seeds**, and the
validation-to-test reliability correlation went *negative*. And the
damage match E was built for is tight: C dropped mean accuracy 19.3pp,
E dropped 18.9pp, from the same shift family — the two regimes have
essentially identical ensemble-member test accuracy (0.634 vs 0.634
under majority). The **only** systematic difference between C and E is
whether the ranking survived.

The decisive contrasts:

- **gap_E − gap_C = +0.0080, t = +2.55** — same family, matched damage,
  opposite ranking treatment → weighted flips from losing to winning.
- gap_D − gap_C = +0.0077, t = +2.93 (the run-1 version of the same
  contrast, with the damage-match caveat noted above).
- Pooled across B/C/D/E (120 seed×regime points): Spearman correlation
  between rank-preservation ρ and the weighted-minus-majority gap is
  **+0.444, p = 3.9×10⁻⁷**. More rank preservation → weighted better,
  dose-responsively.

All six pre-stated predictions held, including the two replications
(A and B reproduce the fourth check's weak weighted win).

## Effect size, honestly

The sign flip in regime C is statistically solid (t = −2.70, and the
C-vs-E contrast isolates the cause) but *small in absolute terms*:
majority vote wins by 0.3pp, not by the dramatic margins population
coding showed. That is itself consistent with the refined rule. With
five diverse voters and clipped log-odds weights, one permanently
mis-weighted classifier out of five is a *static, bounded* ranking
error — the weighted combiner over-trusts one voice, but four others
still carry calibrated weights. Population coding's dramatic collapse
came from *per-trial* ranking flips (a normally-reliable neuron looking
catastrophically wrong on that specific trial), which no static weight
can be right about. The emerging effect-size hierarchy across all five
checks:

    per-trial rank flips (population coding)  ≫  static rank flip (this test)
    >  partial drift (TMR)  >  rank-preserving shift (check 4, regimes B/D/E: no effect)

is exactly what "degradation proportional to rank reordering" predicts.

## Verdict

The refined scoping rule survived its designed-to-kill test. Absolute
damage alone — even ~19pp of it, from the same shift family — does not
break the weighted decoder (regime E, weighted still wins). Reordering
relative reliability at that same damage level does (regime C, majority
wins, sign flip significant). Combined with the four earlier checks,
the rule can now be stated in its final, bounded form:

> **Weighted combination beats voting exactly as long as the relative
> reliability ranking learned at calibration time still holds at
> decision time. It degrades in proportion to rank reordering — acute
> per-trial reordering is worst, static reordering is measurable but
> bounded by voter diversity, and magnitude-only degradation, however
> large, does not break it.**

This closes the follow-up identified in `ensemble_ml_findings.md`. The
scoping-rule ladder (Tribe fight-or-flee → Tribe order-acceptance →
population coding → TMR → ensembles → this test) is now complete: two
game-logic discoveries, three independent-field checks, one
designed-to-kill confirmation.

## Honesty notes

- Predictions were written in the script header before the first run
  and not edited afterward. Run 1 confirmed all six.
- Run 2 (regime E) is a disclosed addition to fix regime D's failed
  manipulation check, not a retune: regimes A–D and all outcome
  measures are untouched and reproduce run 1's numbers exactly.
- Regime C's selection is adversarial *by design* — that is the
  hypothesis being tested ("IF ranking reorders THEN weighted breaks"),
  not a claim about how often natural shifts reorder rankings. How
  frequently real-world distribution shift is rank-reordering vs
  magnitude-only is an open, empirical question this test does not
  answer.
- Test labels were used only for final scoring and the (reported, not
  selected-on) manipulation checks. All shift selection used the
  validation split only.
- E's own gap (t = +1.95, 30 seeds) is borderline on its own; the
  claim rests on the paired C-vs-E contrast (t = +2.55) and the pooled
  dose-response correlation (p = 3.9×10⁻⁷), not on E's solo
  significance.
