# Ternary GPT on DNA: learns local genomic structure (measured, non-linguistic)

Pointed the SAME ternary GPT (triadic attention, {-1,0,+1} weights, predictive-
coding loss) at DNA instead of human language -- nucleotides {A,C,G,T} as tokens.
The point: a ternary system learning the structure of something NON-linguistic,
judged by a measurable property (does generated DNA match real genome k-mer
statistics?), not by whether it sounds like us.

## Run 1 -- human chr1 (partial, and the auto-conclusion was overstated)

Trained on 800kb of human chr1:1M-1.8M. Result was PARTIAL, and the script's
auto-printed "learned real structure!" line was too generous -- corrected:

- GC content: real 58.6% -> generated 60.2% (learned, real signal).
- 3-mer correlation to real: generated 0.173 vs random -0.274 -- weak positive.
- 3-mer TV distance: generated 0.237 vs random 0.170 -- generated was WORSE than
  random. The model over-produced homopolymer runs (GGGGG), distorting the
  distribution.

Honest read: learned GC bias and a weak directional signal, but did NOT cleanly
reproduce k-mer structure. Diagnosis: the region was pathological -- human chr1
start is GC-rich, CpG-island, repeat-heavy, exactly the homopolymer structure the
model latched onto.

## Run 2 -- honed: clean yeast coding DNA (strong, auto-conclusion earned)

Switched to the MIDDLE of yeast chr IV (coding-dense, ~38% GC, no low-complexity
repeats). Same model, same size, same CPU -- ONLY the genome changed.

| | GC | k=2 corr | k=3 corr | k=4 corr |
|---|---|---|---|---|
| generated vs real | 38.2% (real 37.7%) | 0.882 | 0.851 | 0.778 |
| random baseline | 50% | -0.376 | -0.227 | -0.108 |

TV distance beats random at every k (0.095<0.174, 0.117<0.239, 0.163<0.285) --
fixing run 1's failure where TV was worse than random. Strong k-mer correlation
(0.78-0.88) at all k. Generated sample looks like real DNA (starts ATG, diverse,
includes a real-looking poly-A/T tract).

## Findings

1. **The ternary GPT genuinely learns local DNA structure** -- k-mer correlation
   0.78-0.88, decisively beating random, with zero notion of "language." A
   measured yes to "learn the structure of a non-linguistic domain, judged by
   what it can do."
2. **The human run's weakness was the GENOME REGION, not the architecture.**
   3-mer correlation went 0.17 (human) -> 0.85 (yeast) with only the region
   changed. A repeat-heavy region is an unfair target; clean coding DNA is
   learnable by this same 1M-param ternary CPU model.
3. **Honest ceiling, unchanged:** this is LOCAL structure (2-4-mers, base
   composition). It does NOT capture genes, function, or long-range organization
   -- that needs large models. "Learned local DNA statistics, strongly" is the
   bounded claim.

## Method note

The script's auto-printed "learned real structure" conclusion is only valid when
the numbers support it (yeast), NOT automatically (human run 1, where it was
overstated and corrected here). Always read the k-mer numbers, not the auto-line.

---
*Script: trit_dna.py   Runs: trit_dna_run.txt (human), trit_dna_yeast_run.txt (yeast)*
