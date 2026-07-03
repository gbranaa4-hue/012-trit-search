# Ternary quantizer: the scale matters, the "shadow" (residual trit) doesn't

Two experiments on the from-scratch triadic encoder (`trit_triadic_encoder.py`),
testing whether a second "residual" trit — keeping the remainder the first
quantization discards, the "shadow the collapse casts off" — improves the model,
and, as a control, what a magnitude scale alone buys.

## Experiment 1 — reconstruction only (`trit_residual_quant_test.py`)

Direct measurement, no training: how well does 1 trit vs 2 residual trits
reconstruct real weight distributions?

| | 1 trit (+scale) | 2 trits (residual) | error reduced |
|---|---|---|---|
| mean rel_err over 6 weight distributions | 0.457 | 0.232 | **50.6%** |

Keeping the shadow roughly halves reconstruction error. Set a pre-registered
rule before running downstream: >30% reduction → worth a training A/B. It
cleared that. But reconstruction error is necessary, not sufficient — it is
not task accuracy.

## Experiment 2 — 3-arm training A/B (`trit_residual_ab.py`)

Real encoder, held-out retrieval accuracy, everything held constant (same
architecture, data, seed, and gradient estimator) so any gap is attributable
to the quantizer alone. 400 epochs/arm, CPU. Component-isolated the way the
rotation ablation (`triadic_robustness_findings.md`) is.

| Arm | quantizer | retrieval acc | isolates |
|---|---|---|---|
| A | bare `{-1,0,+1}`, no scale | 5.1% | current method |
| B | `a·{-1,0,+1}`, optimal scale | 15.2% | **scale: +10.1pp** |
| C | `a1·t1 + a2·t2` (shadow trit) | 15.2% | **shadow: +0.0pp**, 2x storage |

## Findings

### Finding 1: the magnitude scale is the biggest lever — bare ternary wastes it

Adding an optimal least-squares scale `a = <w,t>/<t,t>` tripled retrieval
accuracy (5.1% → 15.2%). The original quantizer returned sign only, discarding
all magnitude. This is now shipped in `trit_triadic_encoder.py`
(`ternary_quantize_scaled`), reproducing the exact measured configuration
(scale + straight-through estimator).

### Finding 2: the shadow (residual trit) added nothing where it counts

Despite halving reconstruction error, the second trit produced +0.0pp held-out
accuracy at 2x storage. Its *training* loss dropped (2.16 → 1.92) with no
val gain — the extra trit fits the training data better but does not
generalize, i.e. added overfit capacity, not useful capacity. Per the
pre-registered rule ("if C−B ~0 or negative, stop"), the full 2000-epoch run
was not justified. Dropped.

### Finding 3: reconstruction error mispredicted task value — the key lesson

The two signals pointed opposite ways:
- The **shadow** crushed reconstruction error (0.44 → 0.21) and added **0pp** accuracy.
- The **scale** barely moved reconstruction error (0.46 → 0.44) and added **+10pp** accuracy.

Trusting the cheap reconstruction proxy alone would have bet on exactly the
wrong component. This is why the reconstruction test was framed as
necessary-not-sufficient and gated behind a training A/B, not shipped on its own.

## Caveats (stated, not buried)

- 400 epochs is undertrained (15% absolute is low); the C−B tie could be partly
  a floor artifact. The train-loss-drops-without-val-gain pattern argues it is
  a real overfit signal rather than pure noise, but a full-length B-vs-C run
  would settle it if the residual is ever revisited.
- This is the from-scratch research encoder, **not** the production OBSERVE
  search (which uses fine-tuned MiniLM). The +10pp improves this encoder; it
  does not change what OBSERVE search currently uses.

---
*Scripts: trit_residual_quant_test.py, trit_residual_ab.py*
*Shipped: ternary_quantize_scaled in trit_triadic_encoder.py*
