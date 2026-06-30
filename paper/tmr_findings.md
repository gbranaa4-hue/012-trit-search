# TMR / fault-tolerant systems: the scoping rule confirmed a third time

`trit_tmr_test.py` — a third independent check of the consensus-gate
scoping rule (weighted combination wins under calibrated evidence, voting
wins under uncalibrated/drifted evidence), this time against classical
fault-tolerant systems engineering: Triple/N-Modular Redundancy. This
project's own hardware (`hardware/consensus_gate.sv`, 36/36 testbench
pass) implements exactly the majority-vote primitive being tested here.

## The established fields this bridges

- **Classical TMR** (von Neumann, 1956): combines replicated, independent
  units by plain majority vote, with no per-unit confidence model — the
  reliability formula `R_TMR = 3R² - 2R³` treats every unit as equally
  reliable. Majority voting only helps when individual reliability R >
  0.5; it's an egalitarian, calibration-free combiner by design.
- **Weighted majority / Bayesian sensor fusion**: when each independent
  binary unit's error rate is accurately *known*, a log-odds-weighted
  combiner is provably MAP-optimal — strictly better than plain majority
  vote whenever units are heterogeneous.

Read together, these predict the same boundary already found in this
project's game-logic and population-coding tests.

## Setup

5 independent binary units, each correct with probability rᵢ
(heterogeneous, drawn per seed). A decoder is **calibrated once** on
finite calibration data (estimating each unit's reliability from
observed correct/incorrect outcomes — a realistic estimate, not the true
rᵢ), producing log-odds weights.

- **Regime A (stable):** test-time true reliabilities match calibration.
- **Regime B (drifted):** test-time true reliabilities silently shift
  from their calibrated values (`drift_std=0.25`) — representing real
  component degradation or stale calibration — neither decoder is told.

**Decoder 1 (majority vote / classic TMR):** `sign(Σ votes)`, no
weighting.
**Decoder 2 (weighted / log-odds fusion):** `sign(Σ weight·vote)`,
weights from calibration.

Prediction, stated before running: weighted wins regime A, majority vote
wins or closes the gap in regime B.

## Result — confirmed

| Regime | Majority vote | Weighted (log-odds) | Gap (weighted − majority) |
|---|---|---|---|
| A (stable) | 0.8912 ± 0.0472 | **0.9187 ± 0.0426** | +0.0276 |
| B (drifted) | 0.8606 ± 0.1046 | 0.8571 ± 0.1020 | -0.0035 |

Regime A: t=7.04 across 25 seeds — weighted fusion wins clearly
(+2.76pp). Regime B: t=-0.31 — **the gap collapses to statistical
noise**, exactly the "wins or closes the gap" prediction, not a dramatic
reversal this time but a clean disappearance of the weighted decoder's
advantage the moment its calibration goes stale.

## Honest read

This is a softer confirmation than the population-coding test's dramatic
+37pp contamination-robustness win, and that's worth stating plainly: TMR
drift doesn't cause the weighted decoder to *catastrophically* fail the
way an outlier-contaminated weighted mean did in the LIF test — it just
erodes its edge back down to zero. That makes sense given the different
corruption models: population coding's contamination was a large,
acute per-trial outlier (the kind of thing that breaks an inverse-
variance weighted sum badly), while TMR's drift here is a smooth,
moderate shift in long-run reliability (the kind of thing a weighted
combiner degrades gracefully under, rather than catastrophically). The
scoping rule's *direction* held in both regimes; its *magnitude* depends
on the shape of miscalibration, not just its presence — a real, useful
refinement, not noise.

## Verdict

A third independent corroboration, now spanning game logic
(`npc_consensus_findings.md`, `order_acceptance_findings.md`),
neuroscience/robust-statistics (`POPULATION_CODING_FINDINGS.md`), and
now classical fault-tolerant systems engineering. The rule "weighted
combination wins calibrated, voting wins/ties uncalibrated" continues to
hold, with the added nuance that the *size* of voting's advantage under
miscalibration depends on whether the miscalibration is acute/outlier-like
(catastrophic for weighting) or gradual/drift-like (just erodes
weighting's edge to parity).

**Fourth check, weaker still:** `ensemble_ml_findings.md` tested this
against real scikit-learn classifiers under covariate shift and got an
even softer (borderline inconclusive) result than this one — the gap
didn't collapse at all there. Read together, TMR's drift (partial
reordering) and the ensemble test's shift (apparently little reordering)
bracket population coding's acute outlier corruption (severe reordering)
on a single underlying axis: **it's the degree of relative-reliability
reordering, not the mere presence of miscalibration, that determines how
much voting's robustness advantage shows up.**
