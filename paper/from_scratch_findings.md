# From-scratch rebuild + robustness levers: no cheap lever fixes generalization

A clean, dependency-light (numpy-only) rebuild of the density-classification
result, built decision-by-decision as a teaching artifact (`ca_build.py`), then
used to scout whether simple levers make the rule ROBUST (`ca_levers.py`,
`ca_levers_followup.py`, `ca_levers_quick.py`). It independently reproduces the
main investigation's conclusion from the training side.

## The build (`ca_build.py`)

Ring of N ternary cells; opinions labelled +-1 so majority = sign of the row sum;
inputs drawn at a random density per row (spread of easy->hard); ring via
np.roll; rule = weighted sum over a radius-R neighborhood, thresholded to
{-1,0,+1} (weak sum -> 0 baseline); evolution = "corrected repetition" against
the known-true majority (no gradients).

Result: a SINGLE 8-knob neuron (7 weights + 1 bias) solves N=21 at 92% overall /
82% hard near-ties -- matching the 91-knob two-layer version. **The extra
complexity buys nothing at the trained size.** (Measure before you build big.)

## Robustness -- the honest stress (hard near-ties, split by polarity)

The one-neuron rule COLLAPSES off its trained size: hard-case solved falls
83% (N=21) -> 56 -> 32 -> 21 -> 13% (N=299), and it carries a persistent polarity
lean throughout. "Solved at N=21" != "computes majority."

Three levers scouted, scored on hard@N=149 (unseen), split by +1/-1 polarity:

| lever | hard@149 | +1 | -1 | verdict |
|---|---|---|---|---|
| 1 neuron @ 21 | 15% | 20 | 11 | weak |
| 2 layers @ 21 | 55% | 6 | 100 | FAKE (-1 flooder) |
| 1 neuron @ 21,49,99 | 16% | 16 | 16 | honest, weak |
| 2 layers @ 21,49,99 | 55% | 5 | 100 | FAKE (-1 flooder) |

The "55%" flooders are `(5+100)/2` -- a rule stuck shouting one sign, right on
half the near-ties by default and computing nothing. Only the per-polarity split
exposes it; the averaged "hard" number flatters a broken rule (the always-say-
"rain" forecaster).

## Follow-up: extrapolation limit, or biased everywhere? (biased everywhere)

Measured the multi-size 2-layer at its TRAINED sizes too. The +1/-1 gap is large
(24-35) even at N=21/49/99 -- so multi-size training did NOT balance it in-range;
it is biased everywhere, not a clean solver that merely can't extrapolate.

Instability, stated plainly: the two-layer rule was -1-biased at 100 generations
and +1-biased at 50 generations (same seed) -- the bias DIRECTION flipped just
from training length. These solutions are variance-dominated; magnitude and even
direction wander between runs.

## Conclusion

No cheap lever -- complexity, multi-size training, or both -- reliably yields
honest AND strong generalization. Robust, balanced, scale-free majority is not a
knob you turn; it needs the rare traveling-particle structure, which the main
investigation only got reliably by EXPLICITLY rewarding signal velocity (and even
then at a cost to trained-size accuracy; see evolve_compute_findings.md). This
from-scratch scout reconfirms that from the opposite direction.

Fences: modest generations, single seed per lever, N=149 sits beyond the largest
training ring (99), and gen counts differed between the full (100) and lean (50)
follow-up runs -- so the exact rules are not apples-to-apples. The reusable
finding is the PATTERN (biased in-range, unstable across runs), not any single
number.

## Attempt to close the loop with a velocity reward -- HONEST NULL + diagnosis

The main investigation's working lever was REWARDING SIGNAL VELOCITY. Tried to
reproduce that here (`ca_velocity.py`): fitness = consensus + lambda * velocity,
train at N=21 only, measure balanced hard@149. It did NOT reproduce -- and why
is instructive:

1. Sparse-reward failure. With a gaming-proof min-over-polarities velocity, the
   treatment run came out BYTE-IDENTICAL to baseline. Diagnosis: min-velocity is
   nonzero for only ~5% of random rules (2/40), and even ca_build_best -- which
   SOLVES N=21 at 92% -- scores min-velocity 0.0 (it fails the block probe on its
   weak polarity). The reward is a flat field of zeros; evolution has nothing to
   climb. This is exactly why the main run needed ~150 generations (runway to
   stumble on the rare nonzero signal) and 12 seeds.

2. Dense reward steers but doesn't win. Replaced with a DENSE reward (normalized
   both-polarity block-progress, min over polarities). Sanity check caught that
   the all-zero rule scores -2.18 (it ZEROS the ring, destroying correct cells) --
   the reward correctly punishes destruction and has real spread (std 0.31), so
   it steers (treatment now diverges from baseline). But at lean 40-gen settings
   it does NOT produce balanced generalizers: treatment min-hard@149 is no higher
   than baseline and the +1/-1 gap is no smaller (one seed is a +1-flooder, 64/8).

Honest conclusion: the from-scratch loop did NOT close at lean settings. This
does NOT overturn the causal result (which stands from the fuller 150-gen/12-seed
run in evolve_compute_findings.md) -- it shows the reproduction is EXPENSIVE and
FRAGILE: rewarding velocity is a sparse, delicate signal that needs long runway
and careful reward shaping to bite. A valuable negative: it explains WHY the
main result required the settings it did. Not pursued to the full run (poor value
to re-establish a known result at high compute cost).

---
*Scripts: ca_build.py (Socratic build), ca_scratch.py, ca_levers.py,
ca_levers_followup.py, ca_levers_quick.py, ca_velocity.py   Runs: ca_levers*_run.txt*
