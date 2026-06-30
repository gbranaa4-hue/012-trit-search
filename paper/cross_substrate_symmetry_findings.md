# Does symmetry-breaking help computation? Three substrates, three answers

A synthesis across three independent, already-completed investigations —
acoustic MEMS plates, ternary triadic consensus logic, and (proposed) a
software spiking-resonator bank — that all probed structurally the same
question: **does breaking the symmetry of a coupled nonlinear system unlock
extra computational power, or just cost you something for nothing?**

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

## Substrate 3 — Spikeling Resonator bank (proposed, not yet run)

Source: `Spikeling-Project/resonator-prototype/` — a damped-harmonic-
oscillator neuron bank (`x'' = -ω²x - 2·damping·ω·v + coupling·drive`),
explicitly built on the same physics class as the acoustic plate's
nonlinear oscillator network, but in fully controllable software instead
of an FEM-simulated MEMS plate.

This is the natural fourth data point: build a small reservoir from a
bank of Resonator neurons with an engineered, controllable coupling
topology (symmetric vs. deliberately broken, mirroring the D4-vs-broken
setup), and re-run the acoustic plate's *own* shallow even-order memory
task (`y[n] = u[n-1]·u[n-2]`, depth ≤ 3) on it.

This is a genuine generality test, not a rerun: if the same even/odd
selection-rule dichotomy appears in a structurally different (software,
spiking, not FEM-acoustic) substrate built from the same oscillator
primitive, that promotes the acoustic finding from "one substrate's
quirk" to "a property of coupled nonlinear oscillators in general." If it
*doesn't* replicate, that's equally informative — it would mean the
effect depends on something acoustic-plate-specific (the literal
∫φ³ spatial-overlap integral over a 2D plate geometry), not just the
abstract oscillator equation.

**Status: not yet run. This is the proposed next experiment, not a
result.**

---

## The pattern across substrates

| Substrate | Symmetry broken | Net result |
|---|---|---|
| Acoustic plate | Point-group (D4 → low-sym/quasicrystal) | **Clean win**, narrow regime (weak coupling, shallow even-order) |
| 012 triadic mixing | Mixing-formula symmetry | **Tradeoff** (stability up, accuracy down); more flexibility (adaptive) made it worse |
| 012 resonance series | Temporal (instant vs. integrated) | **Win only if signal is genuinely time-varying**; harmful otherwise |
| Spikeling resonator bank | Coupling topology | *Proposed — open* |

The honest synthesis, consistent with every substrate measured so far:
**symmetry-breaking is not a general computational free lunch.** It
unlocks a *specific* capability the symmetric default structurally
cannot have (a product/even-order term; a stability mode; a temporal
response) — but only when the task actually needs that specific
capability, and at a real cost (coupling strength, accuracy, or
performance on the unrelated default case) everywhere else. The acoustic
plate's result is the cleanest because the mechanism is an *exact*
selection rule (∫φ³=0 by symmetry, not a statistical tendency); the
012-ternary results are messier because the mixing formula's "symmetry"
is a much softer, less mechanistically pinned-down property than a true
point-group symmetry — which may itself be worth investigating: does a
sharper, more mechanism-grounded notion of symmetry in the 012 triadic
gate produce a cleaner win, the way it did acoustically?

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
