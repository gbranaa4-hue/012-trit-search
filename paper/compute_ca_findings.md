# Does the ternary CA COMPUTE, or just look coherent? (Frontier: NOT crossed)

The growing ternary CA demonstrably grows coherent structure (even a
non-trivial heart -- see grow_ca_findings.md). The next honest question: can
the talking cells actually COMPUTE a global property, or do they only look
brain-like? Tested on the classic density/majority classification task.

## The task (and why it's the right test)

Grid starts as a random +/-1 pattern; every cell must converge to whichever
value was in the MAJORITY. That requires GLOBAL information (the overall count)
to emerge from purely LOCAL neighbor talk -- no cell sees the whole grid. It is
genuinely "cells talk to reach one collective decision," and it is measurable.

**Real metric: consensus amplification.** Doing nothing leaves each cell at its
initial value, so the fraction of cells at the majority starts at the majority
density (~62%). Genuine computation must AMPLIFY that toward 1.0 (all agree).
(An earlier metric -- sign of the grid mean -- was discarded as trivial: for a
random +/-1 grid the mean's sign IS the majority by construction, so a
do-nothing CA would score ~100%. Consensus amplification is the honest test.)

## Result: NOT crossed (two attempts)

| | initial consensus | final consensus | solved (>90% agree) | loss trend |
|---|---|---|---|---|
| attempt 1 (per-param grad norm, 16x16, 24 steps) | 62.7% | 57.9% | 0% | UP 0.93->1.5 |
| attempt 2 (grad clip, 14x14, 40 steps) | 62.6% | 61.7% | 0% | UP 0.94->1.37 |

Both: no consensus amplification (final ~= initial or worse), 0% solved, and
the training loss INCREASED rather than descended. The cells run and look
coherent but do not perform the global computation.

## Honest read -- why this is a real boundary, not a tuning miss

1. Density classification is provably unsolvable perfectly by a uniform CA, and
   even good hand-designed / evolved CAs reach only ~75-85% on random inputs.
2. It is specifically hard to reach by GRADIENT DESCENT through unrolled CA
   steps -- vanishing/exploding gradients over tens of steps, exactly what the
   non-descending loss shows. Historically it's solved by evolutionary search
   and hand-design (GKL rule), not backprop. Two failing attempts are
   consistent with that known difficulty, not "one tweak away."

## The line this draws (the actual value of the result)

- Morphogenesis -- growing coherent structure, even non-trivial (heart) -- WORKS.
- Computation -- a global property from local ternary talk -- does NOT (this
  task, gradient-trained, two attempts).

That is the honest current boundary between "looks like a brain forming" and
"computes like one." The structure self-organizes; it does not yet compute.
Crossing it would likely need a different training approach (evolutionary
search, or architectures designed for long-range propagation), not a tweak to
this one.

---
*Script: trit_compute.py   Runs: trit_compute_run.txt, trit_compute_run2.txt*
