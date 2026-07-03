# Growing ternary neural cellular automaton: staged results

Goal: demonstrate the concept "neurons spawn from a seed, talk only in trits
{-1,0,+1}, and self-organize into one coherent whole." Built as a growing
Neural Cellular Automaton (Mordvintsev et al. 2020 lineage) with ternary
inter-cell communication and a consensus firing rule.

## Two initial failures (recipe bugs, diagnosed)

First two attempts collapsed in opposite ways:
- per-parameter gradient normalization → runaway fill-all (loss 0.08→0.92, frozen)
- global gradient clip → dead, seed decayed, no growth (loss stuck at 0.08 baseline)

Diagnosis: the missing ingredient was the standard Growing-NCA **stochastic
per-cell update mask** (each cell updates only ~50% of steps, chosen randomly).
Without it, all cells move in lockstep and collapse together. Rebuilt staged,
one component at a time (same discipline as triadic_robustness_findings.md).

## Staged results (grown match-to-target MSE; 0.08 = "did nothing", lower = grew the square)

| Stage | what changed | grown MSE | verdict |
|---|---|---|---|
| bare | known-good NCA, continuous comms | 0.012 | GROWS a clean coherent square |
| ternary | cells communicate in {-1,0,+1} | 0.036 | GROWS (the concept, validated) |
| consensus | + consensus growth gate | 0.79 | FAILS — runaway fill-all |

### Finding 1: the base recipe works once the stochastic mask is added

Bare NCA grows a coherent square from a single seed cell by local rules only
(loss 0.126 → 0.0097 over 500 steps). Self-heal after erasing half is partial
(regrows but overshoots, MSE 0.26) — clean self-repair needs persistent-pool
training (train on damaged states), not done here. Growth: real. Clean heal: not yet.

### Finding 2: ternary communication works — the core concept is demonstrated

Forcing every inter-cell message into {-1,0,+1} still self-organizes a coherent
square (MSE 0.036). Cost of ternarizing communication, measured directly against
bare: ~3x the reconstruction error (0.012 → 0.036) — visibly blobbier edges, but
unmistakably a grown coherent whole, far below the 0.08 do-nothing baseline. This
is the honest, measured validation of "neurons that talk only in trits become one
coherent structure."

### Finding 3: the consensus gate (as built) is a runaway accelerant — isolated cleanly

Adding the consensus growth gate broke it (MSE 0.79, grid fills solid, loss never
converges). Because bare and ternary are identical except for this gate, the
failure is attributable to it alone. Root cause: the gate was implemented as
add-only life (`life = clamp(life + consensus, 0, 1)`) — it can only keep cells
alive, never suppress. That is pure positive feedback: any active region drives
its neighbors past the consensus threshold, which keeps them alive, which spreads
unstoppably. A working consensus firing rule would need a brake (suppression when
consensus is not sustained), or would gate the growth DIRECTION rather than just
adding life. Not required for the core concept, which already works without it.

## Follow-up fixes

### Consensus gate redesigned -- runaway fixed (SUCCESS)

The add-only consensus gate (life += consensus) was pure positive feedback ->
runaway fill-all (MSE 0.79). Redesigned: the standard alive-mask stays the
growth BRAKE, and the signed neighbor-consensus is fed as an INPUT to the
update rule instead of touching the life mask. Tested with the same working
from-seed trainer as bare/ternary (so the result is attributable to the gate,
not the trainer):

| consensus version | grown MSE | loss |
|---|---|---|
| old (add-only life) | 0.79 (runaway) | never converged |
| redesigned (input + alive brake) | 0.027 | converged 0.126 -> 0.017 |

Clean bounded growth, actually slightly better than plain ternary (0.036).
The gate redesign works.

### Persistent-pool training for clean self-heal -- FAILED (first attempt)

Attempted to make self-heal robust via pool training with damage. It
regressed: the model never learned the dense square (grown MSE 0.11, worse
than the 0.08 do-nothing baseline) and settled into faint diffuse mush. The
"healed ~= grown" numbers were both equally bad, not clean repair. Likely
cause: resetting the worst sample to seed every step + damaging an untrained
model made optimization unstable from the start. From-seed training remains
the better model; self-heal across working modes stays partial (~0.095).

**Second attempt -- warm-start pool training -- also FAILED.** Phase 1
(from-seed warmup) grew the square fine (loss 0.126 -> 0.042). But phase 2
(pool + damage) immediately regressed it: loss jumped back to ~0.12 and
stayed, final grown MSE 0.119 (mush). Pool training actively UNLEARNED the
growth phase 1 established. Conclusion after two attempts: pool training as
implemented destabilizes the model regardless of warm-start; clean self-heal
is NOT achieved. Real Growing-NCA pool training is genuinely finicky and
would need faithful replication of the original recipe (specific damage rate,
sample selection, gradient handling) -- real effort, uncertain, not cracked
here. From-seed training with partial (~0.095) self-heal stands as the
working result.

**Diagnostic (pool, no damage) -- isolates the cause.** Ran pool training
with damage OFF: it STILL regressed (grown MSE 0.113 mush). So damage is not
the culprit -- the pool mechanism itself is. Root cause: pooled states persist
across training iterations and accumulate into long horizons (hundreds of
steps) that a model trained only to ~40 steps cannot hold; it degrades them to
mush, and training on those degraded states feeds the failure back. Long-horizon
instability, not damage.

**Fourth attempt -- seed-repair (no pool) -- fixes growth, not heal.** Based on
that diagnosis: grow from a fresh seed but erase half PARTWAY through and require
recovery by the end -- teaching repair with no state persistence, so no
long-horizon feedback loop. Result: growth preserved (loss ~0.035, grown MSE
0.033 -- confirms the diagnosis, no regression) BUT heal got WORSE (healed MSE
0.172 vs the 0.095 baseline): trained on half-erasure, the model regrows too
aggressively and overshoots the square boundary at heal time.

**Net after four attempts: clean self-heal is unsolved.** Growth, ternary
communication, and the redesigned consensus gate all work. Robust clean
self-repair does not -- pool training regresses growth (long-horizon
instability); seed-repair preserves growth but overshoots on heal. The plain
from-seed model's partial heal (~0.095) remains the best. A genuinely clean
solution likely needs faithful replication of the original Growing-NCA recipe
(long-horizon stability training + careful pool management), which is real
research effort, not a tweak.

## Honest scope

This is morphogenesis — growing coherent STRUCTURE — which is real and now
demonstrated with ternary communication. It is NOT cognition: it does not reason
or do language and does not become an LLM. Growing structure this way is solved;
growing a mind this way is an unclaimed open frontier.

---
*Script: trit_grow.py (modes: bare / ternary / consensus)*
*Runs: trit_grow_bare.txt, trit_grow_ternary.txt, trit_grow_consensus.txt*
