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

## Honest scope

This is morphogenesis — growing coherent STRUCTURE — which is real and now
demonstrated with ternary communication. It is NOT cognition: it does not reason
or do language and does not become an LLM. Growing structure this way is solved;
growing a mind this way is an unclaimed open frontier.

---
*Script: trit_grow.py (modes: bare / ternary / consensus)*
*Runs: trit_grow_bare.txt, trit_grow_ternary.txt, trit_grow_consensus.txt*
