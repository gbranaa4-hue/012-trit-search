# Ensemble ML: fourth check — weak/inconclusive, not a clean confirmation

`trit_ensemble_test.py` — a fourth check of the consensus-gate scoping
rule, this time against real scikit-learn classifiers (LogisticRegression,
DecisionTree, KNN, GaussianNB, shallow RandomForest) on a real
classification pipeline, not another synthetic Bernoulli/LIF simulation.
Unlike the first three checks (TMR, population coding), this one uses
**correlated, shift-driven** degradation rather than independent or
per-unit-random corruption — a more realistic ML failure mode.

## Setup

5 diverse classifiers trained on one split; ensemble weights calibrated
from a held-out validation split (log-odds of validation accuracy — the
same weighted-majority math as `trit_tmr_test.py`, applied to real
trained models instead of simulated binary units).

- **Regime A:** test set drawn i.i.d. from the same distribution as
  train/validation.
- **Regime B:** covariate shift applied to 40% of features (offset +
  rescale) at test time, unknown to the ensemble — intended to hurt
  scale/distance-sensitive classifiers (KNN, GaussianNB) more than
  tree-based ones, producing uneven, correlated degradation.

**Decoder 1 (hard/majority vote):** unweighted vote.
**Decoder 2 (weighted vote):** validation-accuracy log-odds weighted
vote.

Prediction, stated before running: weighted wins regime A, majority
vote wins or closes the gap in regime B.

## Result — does not clearly confirm the prediction

| Regime | Majority vote | Weighted (val-acc) | Gap |
|---|---|---|---|
| A (no shift) | 0.8292 ± 0.0378 | 0.8364 ± 0.0288 | +0.0072 |
| B (covariate shift) | 0.7732 ± 0.0547 | 0.7787 ± 0.0528 | +0.0055 |

Regime A: t=2.14 (20 seeds) — weighted voting wins, but the margin is
tiny (+0.72pp) and only borderline significant. Regime B: t=1.73 — the
gap **did not collapse or reverse** the way the TMR and population-coding
tests predicted; weighted voting still numerically wins by almost the
same small margin under shift.

This is a **weak, ambiguous result**, reported as run, with no parameter
retuning afterward to chase the expected pattern.

## Why, honestly — a real refinement, not an excuse

The covariate shift applied here likely degraded all five classifiers'
*absolute* accuracy somewhat together, without strongly reordering their
*relative* reliability ranking — and a log-odds weighted combiner mainly
cares about which classifier is more trustworthy than which, not the
absolute accuracy level. If classifier reliability order is roughly
preserved even as overall accuracy drops, the calibrated weights stay
approximately correct and the weighted combiner doesn't actually break.

This sharpens the scoping rule further, consistent with what the TMR
test's softer (gap-closing, not reversing) result already hinted at:
**the weighted decoder degrades in proportion to how much miscalibration
reorders relative reliability, not just how much it lowers absolute
accuracy.** Population coding's acute per-trial outlier corruption could
make a normally-reliable neuron look catastrophically wrong on that
specific trial (a real ranking flip, trial by trial); TMR's gradual drift
only partially reordered units; this ensemble shift, as constructed,
apparently preserved ranking even more than TMR's drift did.

## What would be needed to test this more fairly

A sharper, deliberately rank-reordering manipulation — e.g., a shift
specifically engineered to make the *previously-best* validation
classifier become the *worst* test-time classifier (not just shift
features generically) — would directly test the refined hypothesis
("rank-reordering causes the predicted collapse; magnitude-only
degradation does not"). Not run here, to avoid retuning until a
preferred result appears; left as an identified follow-up.

## Verdict

Combined with `tmr_findings.md`'s softer-than-population-coding result,
this is the second consecutive check where the *direction* of the
scoping rule held weakly or ambiguously rather than dramatically — which
is itself informative: **the rule's strength depends specifically on
whether miscalibration reorders relative trust, not just whether
evidence is "uncalibrated" in some general sense.** This is a more
precise, more defensible version of the rule than the original two
game-logic tests alone implied, even though this particular check did
not produce a clean win for either side. Recorded honestly as a weak
result, not silently dropped or re-run until positive.
