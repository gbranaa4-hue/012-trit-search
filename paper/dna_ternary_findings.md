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

## Run 3 -- cross-species: does yeast knowledge "generalize," or is DNA just universal?

Tested whether the yeast-trained model transfers to species it never saw
(human, fly, worm), fetched as clean mid-chromosome regions via Ensembl.

FIRST, the honest prerequisite -- are species even distinguishable at the
k-mer level? Real cross-species 3-mer correlations:

| pair | 3-mer | 4-mer |
|---|---|---|
| yeast-worm | 0.954 | 0.932 |
| yeast-fly | 0.909 | 0.884 |
| yeast-human | 0.891 | 0.874 |
| human-fly | 0.835 | 0.820 |

Local DNA structure is LARGELY UNIVERSAL across these eukaryotes (0.82-0.95
shared). That reframes "cross-species generalization" before any model runs.

Yeast-trained model's generated DNA vs each real species (random baseline is
negative at every k):

| species | model match (3-mer / 4-mer) | real yeast-similarity (3-mer / 4-mer) |
|---|---|---|
| yeast (own) | 0.927 / 0.887 | -- |
| worm | 0.915 / 0.871 | 0.954 / 0.932 |
| fly | 0.843 / 0.801 | 0.909 / 0.884 |
| human | 0.838 / 0.799 | 0.891 / 0.874 |

Findings:
1. The model "transfers" to unseen species at 0.80-0.92 -- but this is NOT
   cross-species generalization. It is explained entirely by the universality
   of local DNA structure (species are 0.84-0.95 alike to begin with).
2. Transfer strength TRACKS real species similarity: model matches worm >
   fly ~= human, the exact order of how similar those genomes really are to
   yeast. The model applies "what yeast looks like"; each species benefits to
   the degree it resembles yeast.
3. It DID capture yeast-SPECIFIC structure, not just the shared floor:
   model-vs-yeast (0.927) exceeds model-vs-human (0.838) by more than the real
   yeast-human gap -- the generated DNA is more yeast-like than the actual human
   genome is. So it memorized yeast's distribution (incl. yeast-specific bias),
   not merely the universal eukaryote skeleton.

Answer to memorize-vs-generalize: it learned yeast's local structure (including
yeast-specific features); that knowledge appears to "generalize" only because
local DNA k-mer structure is mostly universal.

Multi-genome training NOT pursued (deliberate): with species only 5-18% apart at
the k-mer level, training on several genomes averages toward the same shared
distribution -- a held-out species lands near ~0.85 either way, and you can't
separate real multi-genome generalization from single-genome transfer. The
signal isn't there at this level. Where species truly differ (gene organization,
long-range structure) is invisible to a 1M-param local model.

---
*Script: trit_dna.py   Runs: trit_dna_run.txt (human), trit_dna_yeast_run.txt (yeast)*
*Cross-species measured live; regions cached in data/dna_{yeast,human,fly,worm}.txt*
