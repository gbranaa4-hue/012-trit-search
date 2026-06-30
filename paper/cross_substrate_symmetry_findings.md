# Does symmetry-breaking help computation? Four substrates, four answers

A synthesis across four independent investigations — acoustic MEMS
plates, ternary triadic consensus logic, ternary temporal/resonance
logic, and a software spiking-resonator bank — that all probed
structurally the same question: **does breaking the symmetry of a coupled
nonlinear system unlock extra computational power, or just cost you
something for nothing?**

These were not designed as one study. Each was run independently, in a
different project, for a different reason, with no shared codebase. The
fact that they converge on the same abstract question — and *disagree* on
the answer — is itself the finding worth recording.

---

## The shared question, stated precisely

Take a system built from coupled nonlinear units (oscillators, consensus
gates, resonators) whose *default* configuration has some exact symmetry
(a point-group symmetry, a symmetric mixing formula, a symmetric response
to time). Ask: if you deliberately break that symmetry, does the system
gain the ability to compute something it structurally could not compute
before — and at what cost?

## Substrate 1 — Acoustic MEMS plate (quasicrystal vs periodic)

Source: `acoustic-vortex-sim/reservoir_computing/FINDINGS.txt`,
`PROOF_selection_rule.txt` (theorem proven + confirmed on FEM modes to
1e-9).

**The mechanism (exact, not statistical):** a D4-symmetric (periodic)
plate's antisymmetric vibration modes satisfy φ(-x) = -φ(x), which forces
the quadratic self-nonlinearity coefficient c2 = ∫φ³ dA to vanish
*exactly* for those modes. A quadratic term is what lets a mode compute a
*product* of two inputs (an even-order task, e.g. y = u[n-1]·u[n-2]).
Symmetry doesn't just make this hard — it makes it structurally
**impossible** for ~88% of a periodic plate's modes. Breaking the
symmetry (quasicrystal, or even just a low-symmetry periodic plate) keeps
the quadratic term alive almost everywhere instead.

**Measured result, stress-tested and peer-reviewed:**
- Even-order tasks: quasicrystal wins all 5 tested, gap +0.07 to +0.23 R²
  (1.4x–2.9x significance). Odd-order tasks: all 5 tie, gap within
  ±0.007 — a "textbook-clean even/odd dichotomy," the exact fingerprint
  of the selection rule.
- Cross-validated: paired t-test p≈1e-13, 100% of 40 seed×drive pairs,
  survives k-fold CV, feature standardization, and a 5-decade ridge-λ
  sweep — not a scaling or regularization artifact.
- **Peer-review correction:** the effect tracks *broken point symmetry*
  monotonically (D4 periodic R²=0.273 → low-symmetry periodic R²=0.362 →
  quasicrystal R²=0.457), not aperiodicity specifically. The quasicrystal
  is simply the *maximal* symmetry-breaker available, not special in
  kind.
- **The cost / scope limit:** real but narrow. Only in the
  **weakly-coupled regime** (effective dimensionality ~3) — full coupling
  erases the edge entirely. Only on **shallow** even-order tasks (depth
  ≤ ~3); the reservoir cannot do *deep* even products at all (R² collapses
  to ~0 by lag 5, negative by lag 9), symmetric or not. NARMA-10 (the
  standard benchmark, which needs deep memory) ties — the edge never
  gets a chance to matter there.

**Verdict: breaking symmetry is a clean, real win — but only in a narrow
regime (weak coupling, shallow even-order tasks). It is not a general
"asymmetry is better" result even within this one substrate.**

## Substrate 2 — Ternary triadic consensus logic (012-ternary)

Source: `012-ternary/trit_symmetry_cavity_test.py`, summarized in
`DOCS.md` under "Resonance Hypothesis Test Series."

The triadic mixing formula in `experiments.py`'s `PredictiveTritBlock` is
`output = stream_1·(1-gate) + stream_2·gate` — a fixed, *symmetric-ish*
combination rule. This test made the mixing weights a tunable parameter
and compared `fixed_symmetric`, `fixed_asymmetric`, `adaptive` (learned
per-input), and `input_driven` variants against the original baseline
formula, on CIFAR-10 rotation-robustness (15 epochs, real training, not a
toy task).

**Measured result:**

| Mode | Accuracy @0° | Stability (lower=better) |
|---|---|---|
| baseline (original fixed formula) | **76.97%** | 68.97% |
| fixed_symmetric | 72.41% | 68.82% |
| fixed_asymmetric (broken symmetry) | 70.70% | **66.61%** (best) |
| adaptive (learned, most flexible) | 74.74% | 69.99% (worst) |
| input_driven | 75.76% | 69.23% |

Breaking symmetry (`fixed_asymmetric`) *did* improve stability — the same
qualitative direction as the acoustic plate — but at a real, non-trivial
accuracy cost (6.27pp below baseline). And the *most* flexible,
data-adaptive form of symmetry-breaking (`adaptive`) was the **worst**
performer of all five variants on both axes.

**Verdict: breaking symmetry trades a small stability gain for a real
accuracy cost — a tradeoff, not a clean win. More tunability (adaptive)
made things actively worse, the opposite of what naive "flexibility is
good" intuition would predict.**

## Substrate 2b — Temporal symmetry (012-ternary's resonance series)

A related but distinct axis from the same project: not spatial/structural
symmetry, but *temporal* symmetry — does an instantaneous (memoryless)
response lose to a resonant/time-integrated one? Five tests
(`trit_cache_eviction.py` through `trit_resonant_hopfield.py`) found the
answer depends entirely on whether the underlying signal is actually
time-varying:

- **Static/clean signal** (cache recency, repeated noisy reads of one
  fixed fact): instant response wins; resonance/integration actively
  *hurts* (-1.2 to -5.9pp on the cache test; loses to instant majority
  vote on the noisy-memory test).
- **Genuinely time-varying signal** (a process becoming urgent for a
  bounded window): resonant integration wins decisively, +46.0pp over
  static round-robin during the urgency window.

**Verdict: the same "does breaking the simple/default behavior help"
question, on the time axis instead of the space axis, gives the same
shape of answer as substrate 2 — a real effect, but only in a specific
regime (here: genuinely time-varying signals), and actively harmful
outside that regime.**

## Substrate 3 — Spikeling Resonator bank (run — negative result)

Source: `Spikeling-Project/resonator-prototype/symmetry_selection_test.py`
and `SYMMETRY_TEST_FINDINGS.md` — a bank of 24 damped-harmonic-oscillator
units (`x'' = -ω²x - 2·damping·ω·v + coupling·drive`), the same physics
class as the acoustic plate's nonlinear oscillator network but in fully
controllable software, each given a per-unit quadratic self-nonlinearity
(`c2·x²`) with a controlled "dead fraction" matching the acoustic plate's
measured numbers exactly (88% dead = symmetric/periodic-mirroring config,
38% dead = broken-symmetry/quasicrystal-mirroring config). 20 seeds per
config, ridge-regression readout, same even-vs-odd task battery as the
acoustic plate's rung 6.

**Measured result — does not replicate:**

| Task | Order | Symmetric R² | Broken R² | Gap |
|---|---|---|---|---|
| u[n-1]·u[n-2] | even | -0.013 ± 0.010 | -0.007 ± 0.006 | +0.006 |
| u[n-1]·u[n-3] | even | -0.017 ± 0.009 | -0.010 ± 0.006 | +0.007 |
| u[n-1]² | even | -0.014 ± 0.012 | -0.007 ± 0.007 | +0.007 |
| u[n-1] | odd | 0.303 ± 0.036 | 0.182 ± 0.059 | **-0.121** |
| u[n-1]³ | odd | 0.249 ± 0.031 | 0.147 ± 0.048 | **-0.102** |
| u[n-1]-0.5·u[n-2] | odd | 0.075 ± 0.017 | 0.030 ± 0.013 | -0.045 |

Mean gap: even = **+0.0066** (acoustic plate: +0.150), odd = **-0.0895**
(acoustic plate: -0.002).

This is a genuine generality test, not a rerun, and it came back negative
on both counts: even-order tasks are at floor (R²≈0) for *both* configs —
this minimal reservoir cannot do the product task at all regardless of
symmetry, so there is no capability for symmetry-breaking to unlock — and
the broken-symmetry config measurably *hurts* the odd-order memory the
reservoir does have, the opposite of the acoustic plate's clean "odd
ties" result. No parameter-tuning was done after seeing this result.

**Why, honestly:** the acoustic plate's reservoir had spatial richness
this one lacks — multiple physical drive locations exciting a real
elastic mode-coupling network — and its own rung-1 baseline (generic
oscillators, no selection-rule mechanism) already solved the same task at
R²=0.71 on that richer structure. This resonator bank has a single shared
scalar input line and no inter-unit coupling at all, so it never clears
the bar of "capable enough at the task for symmetry-breaking to matter,"
the precondition the acoustic plate's own findings (rung 1) identified.
**Conclusion: this result does not refute the acoustic finding — it shows
the bare oscillator equation plus a per-unit quadratic term, without the
plate's spatial/coupling richness, is not sufficient on its own to
reproduce the effect.** A fairer follow-up (multiple input channels or
genuine inter-unit coupling, confirmed capable of the even task at
baseline before testing symmetry-breaking on top) is identified in
`SYMMETRY_TEST_FINDINGS.md` but not run here.

---

## The pattern across substrates

| Substrate | Symmetry broken | Net result |
|---|---|---|
| Acoustic plate | Point-group (D4 → low-sym/quasicrystal) | **Clean win**, narrow regime (weak coupling, shallow even-order) |
| 012 triadic mixing | Mixing-formula symmetry | **Tradeoff** (stability up, accuracy down); more flexibility (adaptive) made it worse |
| 012 resonance series | Temporal (instant vs. integrated) | **Win only if signal is genuinely time-varying**; harmful otherwise |
| Spikeling resonator bank | Per-unit quadratic self-term (dead-fraction matched) | **Negative** — no even-order capability to unlock at all; broken-symmetry config actively hurt odd-order memory |

The honest synthesis, consistent with every substrate measured so far:
**symmetry-breaking is not a general computational free lunch, and it is
not even a reliably *available* lunch.** Where it wins (acoustic plate),
it unlocks a specific capability the symmetric default structurally
cannot have, but needs a reservoir already rich enough (spatial drive
diversity, a real coupling network) for that capability to matter. Strip
that richness away — as the bare resonator-bank test did — and the same
mechanism (a per-unit quadratic term, dead-fraction-matched to the exact
acoustic numbers) produces no benefit and even a measurable cost on the
one thing the reservoir could already do. The acoustic plate's result is
the cleanest of the four because the mechanism is an *exact* selection
rule (∫φ³=0 by symmetry) sitting on top of a reservoir already proven
capable (rung 1, R²=0.71 baseline); the 012-ternary and Spikeling results
are messier or null because each lacks one of those two ingredients —
012's triadic "symmetry" is a soft formula property, not a true
point-group symmetry, and Spikeling's reservoir was never shown capable
of the task before symmetry was varied. **The real lesson across all
four: symmetry-breaking is a second-order lever — it only pays off on
top of a substrate that already clears a basic capability bar, and
testing it without first confirming that bar is met (as the acoustic
study's own rung-1 control did, and this Spikeling follow-up did not)
risks measuring noise instead of the effect.**

## Honesty notes

- Substrates 1 and 2/2b are completed, measured, peer-reviewed-or-
  stress-tested results, not estimates. Numbers above are taken directly
  from `FINDINGS.txt`, `PROOF_selection_rule.txt`, and `DOCS.md` —
  not re-derived or rounded favorably.
- Substrate 3 is a proposal, clearly marked as such, not a result.
- This document does not claim a unified theory. It claims three
  independent, honestly-run experiments asked structurally the same
  question and got three different answers, which is itself worth
  recording precisely *because* it resists a tidy unifying takeaway —
  the same discipline applied throughout `FINDINGS.txt` and this
  project's own resonance series (state what's proven, what's scoped,
  and what's still open).
