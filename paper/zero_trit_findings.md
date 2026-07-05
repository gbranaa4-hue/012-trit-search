# Prediction 3, tested: the zero trit is not an uncertainty map

**Script:** `trit_zero_uncertainty_test.py` · **Date:** 2026-07-05 ·
**Hardware note:** run on GPU (RTX 5060, cu128 wheels) after the original
CPU-only torch install was replaced; seed-identical losses verified against
the aborted CPU run.

## What was tested

The last untested prediction of `observer_shadow_light_theory.md`: that the
zero trit encodes *epistemic uncertainty* — the Observer gate "withholding
judgment" (s0 ≈ 0.5) produces intermediate blend values that quantize to 0,
so rotation (which makes Shadow ambiguous) should increase the zero-trit
activation fraction, and should do so more for the triadic model than for a
gate-free ternary CNN.

Pre-registered before the run (script header, unedited):
- **P1** — TritFull zero-trit activation fraction higher at 90° than 0°
  (paired t > 2 across test batches).
- **P2** — the 90°-vs-0° increase larger for TritFull than TernaryStdCNN
  (Welch t > 2).
- **P3** (mechanism) — Observer-gate uncertainty fraction (|s0−0.5| < 0.1)
  higher at 90° than 0° (paired t > 2).

Setup: exact model classes and training recipe from
`trit_stream_ablation_test.py` (20 epochs, 4-epoch warmup, batch 512,
standard augmentation), measurement over the full 10k test set (the earlier
script's `measure_activation_sparsity` sampled 5 batches and was never
reported), zero fraction defined by the same 0.7·mean|a| ternary threshold
applied to block outputs, paired per-batch statistics.

## Result — all three predictions fail

| Model | zero-frac @0° | @90° | Δ | paired t |
|---|---|---|---|---|
| TritFull | 0.4098 | 0.4096 | −0.0003 | −7.77 |
| TernaryStdCNN | 0.4846 | 0.4844 | −0.0002 | −2.57 |

- **P1: refuted.** The zero fraction does not rise at 90°; it is flat to
  four decimal places (the "significant" negative t is a practically-nil
  −0.0003 made visible by pairing).
- **P2: refuted.** No differential effect (Welch t = −1.66, wrong sign).
- **P3: refuted.** Gate uncertainty 0.3721 → 0.3720 at 90° (t = −1.85).

## The full curve is the real finding

| Angle | TritFull zero-frac | TritFull gate-unc | TritFull acc | StdCNN zero-frac | StdCNN acc |
|---|---|---|---|---|---|
| 0° | 0.4098 | 0.372 | 66.7% | 0.4846 | 69.7% |
| 45° | **0.4208** | **0.419** | 27.9% | 0.4817 | 26.8% |
| 90° | 0.4096 | 0.372 | 25.6% | 0.4844 | 26.8% |
| 135° | **0.4208** | **0.419** | 22.7% | 0.4817 | 20.5% |
| 180° | 0.4100 | 0.373 | 36.6% | 0.4844 | 38.5% |

Sparsity and gate uncertainty move **only at 45°/135°** — and 90°/180°
rotations of a square image are *lossless pixel permutations*, while
45°/135° pass through bilinear interpolation and corner fill. Meanwhile the
network's actual confusion (accuracy) is just as catastrophic at 90°
(25.6%, near the 10-class floor) as at 45° (27.9%).

So the discriminating comparison the theory needed is right there in the
data, and it goes the wrong way:

- at **90°**: semantic ambiguity is maximal (accuracy floor), pixel
  statistics are pristine → gate and sparsity register **nothing**;
- at **45°**: semantic ambiguity is the same, pixel statistics are blurred
  by interpolation → gate uncertainty rises (+0.047) and sparsity rises
  (+0.011), and the gate-free StdCNN's sparsity actually *falls* (−0.003),
  so the 45° response is genuinely triadic-specific.

**The Observer gate responds to low-level input statistics (interpolation
smoothing), not to task uncertainty.** A network at near-chance accuracy
whose "uncertainty map" is indistinguishable from its confident baseline is
not maintaining an uncertainty map.

## Verdict for the theory

Prediction 3 — the most distinctive claim of the Observer-Shadow-Light
theory, the one that would have made the zero trit "not noise but
confidence information" — is refuted at the pre-registered contrast, and
the exploratory curve refutes the *mechanism* even more directly than the
null does: the gate demonstrably fails to notice the largest semantic
ambiguity in the test while reacting to a resampling artifact.

Standing of the theory after all three predictions:
1. multi-scale receptive field (5×5 Light) → **confirmed** (the robustness
   mechanism);
2. multiplicative gating → **weak**, interaction-only;
3. zero trit as epistemic uncertainty → **refuted** (this test).

The honest summary: the Plato's-cave architecture works, but for the
mundane reason (bigger receptive fields see through local disruption), not
the epistemic one (an Observer tracking what the network doesn't know).
The downstream proposals that depended on prediction 3 — sparsity as a
readable uncertainty map, uncertainty-calibrated per-layer thresholds —
lose their empirical motivation and should not be built on this
interpretation.

## Honesty notes

- P1–P3 pre-registered in the script header before the run; run once;
  reported as-is. The 45°/135° observation is *exploratory* (not
  pre-registered) and framed as a hypothesis about interpolation artifacts,
  not a confirmed mechanism; the natural confirmatory follow-up (e.g.
  Gaussian blur at 0° should reproduce the 45° gate response if the
  interpolation account is right) is identified but not run here.
- One training run per model (the ablation study's precedent); the paired
  per-batch design gives the tiny deltas their significance, which is why
  effect sizes, not t-values, carry the interpretation.
- Accuracy at 0° (66.7%) is below the 50-epoch ablation study's clean
  accuracy, as expected at 20 epochs; the theory's prediction is about the
  *direction* of sparsity change, which does not require a fully converged
  model — but a converged-model replication would strengthen the refutation.
