# The Observer-Shadow-Light Motif: From Plato's Cave to Measurement Theory

## Design Philosophy — Plato's Cave

The PredictiveTritBlock was designed around a structural analogy to Plato's
Allegory of the Cave (Republic, Book VII):

> Prisoners chained in a cave see only shadows cast on the wall by objects
> passing before a fire behind them. The shadows are the only reality they
> know. A prisoner who escapes and sees the fire — and eventually the sun —
> understands that the shadows were projections of a deeper reality.

The three streams map directly:

| Stream | Kernel | Activation | Role | Cave analogy |
|---|---|---|---|---|
| Shadow | 3×3 | tanh [-1,1] | Local features — edges, textures | The shadows on the wall |
| Light | 5×5 | tanh [-1,1] | Spatial context — what surrounds a feature | The fire casting the shadows |
| Observer | 1×1 | sigmoid [0,1] | Per-location gating signal | The prisoner deciding how much to trust shadow vs infer source |

The triadic blend:
```
out = Shadow * (1 - Observer) + Light * Observer
```

When Observer → 1: the network trusts *context* (Light) over local sensation
(Shadow). When Observer → 0: it trusts local sensation. The gate is learned —
the network discovers per-location which signal is more reliable for the task.

The predictive coding head (`pred`: Conv1x1 reconstructing input `x` from
`out`) is the prisoner reasoning backward from shadow to cause. In
neuroscience this is **predictive coding**: the brain continuously predicts
its sensory input and updates only on prediction error. Here, the auxiliary
loss `MSE(pred(out), x)` forces the network to maintain a model of what caused
the features it extracted — exactly the cave prisoner reasoning from shadow
back to fire.

---

## Connection to the Double-Slit Experiment

The double-slit experiment (Young, 1801; confirmed for electrons by Davisson
& Germer, 1927; for single particles by Tonomura et al., 1989):

- An unobserved particle travels both paths simultaneously → interference
  pattern (wave behavior, superposition of both Shadow and Light paths)
- An observed particle collapses to one path → two bands (particle behavior,
  commitment to one stream)
- **The act of observation changes which signal dominates**

The triadic gate shares this motif exactly:

| QM concept | Triadic gate equivalent |
|---|---|
| Two paths through slits | Shadow (3×3) and Light (5×5) running in parallel |
| Superposition | Both streams computed simultaneously, neither committed |
| Measurement / observation | Observer gate (1×1) reading local context |
| Wavefunction collapse | `out = Shadow*(1-s0) + Light*s0` selecting one stream |
| Which-path information | Observer's output value — high = committed to Light, low = Shadow |

**Important caveat:** the triadic gate does not explain or reproduce quantum
mechanics. Both streams always run as classical floating-point computations —
there is no true superposition, no entanglement, no irreversibility. What
the gate shares with QM measurement is a **structural motif**: two
complementary representations arbitrated by an observation step that collapses
toward one. This motif appears to be a general feature of information
processing systems where two signals compete and a third signal mediates.

The deeper connection is through **decoherence theory**: quantum superposition
collapses because the measuring apparatus entangles with the system —
information about which path was taken leaks into the environment. The
Observer gate does something structurally analogous: it uses local context
(the environment of each spatial location) to select which stream dominates,
and its sigmoid output is the "which-path information" that collapses the blend.

---

## Why This Explains Rotation Robustness

When an image rotates:
- **Shadow (3×3 local)** changes dramatically — local edges point in different
  directions, textures appear different, patch-level features are disrupted
- **Light (5×5 context)** changes more slowly — coarser spatial relationships
  between regions are partially preserved across small rotations
- **Observer** learns to up-weight Light (context) when Shadow is ambiguous

This is the same mechanism that makes the cave prisoner's contextual reasoning
more reliable than shadow-watching when the light source moves: if you
understand the fire (context), you can still interpret shadows even when they
distort. If you only know shadows, any distortion destroys your model.

The stream ablation tests this directly:
- `Trit-3x3Light`: replace 5×5 with 3×3 — removes the stable contextual signal.
  If robustness drops, the 5×5 Light stream is the mechanism.
- `Trit-AddGate`: `out = Shadow + Light` — removes the Observer gating.
  If robustness drops, the multiplicative collapse is the mechanism.
- Both ablations together isolate whether it is the multi-scale receptive
  field, the gating logic, or their combination.

---

## The Zero Trit as Epistemic Uncertainty

In ternary quantization, weights/activations map to {-1, 0, +1}. The standard
interpretation of 0 is "small magnitude — not strongly activated." But the
Observer-Shadow-Light motif suggests a richer interpretation:

**The zero trit = the Observer withholding judgment.**

When the Observer gate output is near 0.5 (neither committing to Shadow nor
Light), the resulting blend `out = 0.5*Shadow + 0.5*Light` produces an
intermediate value that — when quantized — is most likely to round to 0.
The network is in the cave-prisoner state of genuine uncertainty: the shadow
could be either thing, and context doesn't resolve it.

This implies a testable prediction:
> **Zero-trit activation fraction should INCREASE when input is rotated**,
> because rotation makes Shadow ambiguous without fully resolving via Light.
> The network encounters more genuinely uncertain spatial locations and
> produces more zero activations.

If confirmed by the ablation experiments, this means:
1. The zero trit is not noise — it carries information about the network's
   confidence at each spatial location
2. Sparsity is not a compression artifact — it is the network's uncertainty map
3. The 0.7 threshold in `TernaryQuantize` controls how much uncertainty is
   expressed as zero — it should ideally be learned per-layer rather than
   fixed globally

### Toward Uncertainty-Calibrated Ternary Thresholds

Current implementation:
```python
t = 0.7 * x.abs().mean()   # same threshold for every layer
```

Proposed: per-layer learned threshold tied to the Observer's epistemic role:
```python
# Each layer learns how much uncertainty it should express
t = self.uncertainty_scale * x.abs().mean()
# uncertainty_scale initialized to 0.7, learned via gradient
```

Expected behavior under this proposal:
- Early triadic blocks (near raw input, more ambiguity): higher sparsity
- Later blocks (higher-level features, less ambiguity): lower sparsity
- All blocks: higher sparsity at rotated angles than at 0°

This connects ternary quantization to **Bayesian uncertainty estimation** —
the zero trit becomes an analog of a low-confidence prediction, and the
sparsity pattern becomes a spatial uncertainty map that the classifier can
read. Networks that can express where they are uncertain are known to be
more robust to distribution shift (Lakshminarayanan et al., 2017; Gal &
Ghahramani, 2016).

---

## The Motif Across Scales

The Observer-Shadow-Light motif — two complementary signals, one contextual
arbiter, collapse to one — appears across physical and computational systems:

| Domain | Shadow | Light | Observer | Collapse mechanism |
|---|---|---|---|---|
| Plato's cave | Shadows on wall | Fire (cause) | Prisoner's reasoning | Insight / inference |
| Double-slit (QM) | Path A | Path B | Measurement apparatus | Decoherence / entanglement |
| Predictive coding (brain) | Bottom-up sensation | Top-down prediction | Prediction error signal | Error-driven update |
| PredictiveTritBlock | 3×3 local features | 5×5 context | 1×1 gate | Sigmoid-weighted blend |
| Ternary hardware | trit=-1 (Shadow) | trit=+1 (Light) | trit=0 (Observer) | CONSENSUS gate |

The claim is not that these are the same phenomenon. The claim is that this
structural motif — competing signals, mediated by an observation/arbitration
step, collapsing toward one — is a recurring computational primitive that
appears at multiple levels of organization. Building it explicitly into the
network architecture (PredictiveTritBlock) rather than hoping the optimizer
discovers it (StandardCNN) may be why the triadic model learns more
generalizable representations.

---

## Summary of Testable Predictions

From the theory above, three predictions that the stream ablation and
perturbation tests will confirm or refute:

1. **5×5 Light is necessary for rotation robustness** — `Trit-3x3Light`
   should show meaningfully higher worst-drop than `TritFull`

2. **Multiplicative gating is necessary** — `Trit-AddGate` should show
   higher worst-drop than `TritFull` (additive combination can't selectively
   suppress Shadow when it's corrupted)

3. **Zero-trit fraction increases under rotation** — activation sparsity
   measured at 0° vs 90° should be higher at 90° for triadic models,
   and the increase should be larger for `TritFull` than `TernaryStdCNN`
   (which has no Observer gate and no reason to increase sparsity under
   rotation)

## Experimental Results

From `trit_stream_ablation_test.py` (50 epochs, CIFAR-10, 24 rotation angles):

**Prediction 1: 5×5 Light is necessary for rotation robustness**
Confirmed. Trit-3x3Light (no 5×5) drops 3pp clean accuracy and has a lower
floor (21.2% vs 21.6%). The wide receptive field is the primary mechanism.

**Prediction 2: Multiplicative gating is necessary**
Weak. Trit-AddGate (additive `s1+s2`) drops only 1.5pp and barely changes
robustness. The gate contributes less than the 5×5 stream to rotation
robustness specifically. However, when both are removed together
(Trit-3x3+AddGate), the degradation is 8.1pp — larger than either alone,
confirming an interaction effect. The gate's main contribution may be to
non-geometric perturbations (noise, brightness) where scale doesn't help.

**Prediction 3: Zero-trit fraction increases under rotation**
**Refuted (2026-07-05)** — see `zero_trit_findings.md` /
`trit_zero_uncertainty_test.py`. Measured with forward hooks over the full
test set: activation sparsity is flat at 90° (0.4098 → 0.4096) and the
Observer gate's near-0.5 fraction doesn't move (0.3721 → 0.3720), even as
accuracy collapses to 25.6%. Sparsity and gate uncertainty rise only at
45°/135° — the interpolated (lossy) angles — while the lossless 90°/180°
pixel permutations register nothing. The gate responds to low-level
resampling statistics, not to task uncertainty.

**Revised claim:** The Observer-Shadow-Light motif provides rotation robustness
primarily through the multi-scale receptive field (Shadow 3×3 vs Light 5×5),
not through the multiplicative gating logic. The gate's epistemic role —
arbitrating between streams based on which is reliable — likely manifests on
perturbations that corrupt one stream differentially (brightness, noise) rather
than rotating both simultaneously. See perturbation_findings.md.

**Standing after all three predictions (2026-07-05):** 1 confirmed (the 5×5
receptive field), 2 weak (gating, interaction-only), 3 refuted (zero trit as
epistemic uncertainty). The architecture works, but for the mundane reason,
not the epistemic one. The downstream proposals that depended on prediction 3
— sparsity as a readable uncertainty map, uncertainty-calibrated per-layer
thresholds — lose their empirical motivation and are withdrawn rather than
pursued.
