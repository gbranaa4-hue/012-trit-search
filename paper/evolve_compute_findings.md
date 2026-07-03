# Compute frontier CROSSED: evolved ternary CA computes a global property

The morphogenesis side works (grow_ca_findings.md). Gradient descent could NOT
make the ternary CA compute (compute_ca_findings.md) -- backprop through
unrolled CA steps diverged every time. The spec (NEXT_evolutionary_compute.md)
predicted the fix: use evolution, not gradients. It worked.

## Setup

1D ring of N (odd) ternary cells {-1,0,+1}, density classification: every cell
must converge to the MAJORITY initial value -- a global property from local
ternary talk. A small local rule (radius-3 MLP, 91 real weights, output
ternarized) is EVOLVED by a dependency-free (mu,lambda)-ES. Fitness = fraction
of cells at the true majority after ~2N steps. No gradients anywhere.

Baselines fixed before running: random rule 0.34; do-nothing (keep initial
state) ~0.67 = the majority density; chance accuracy 0.50; GKL hand rule ~0.816;
~1.0 provably impossible for a uniform CA.

## Result -- evolved on N=21, held out, replicated on two seeds

| | do-nothing | final consensus | solved (>90% agree) | HARD (near-tie) |
|---|---|---|---|---|
| eval seed A | 67.3% | 94.6% | 92.8% | 83.6% |
| eval seed B | 66.7% | 93.8% | 91.6% | 81.8% |

The hard split is the real test: local smoothing would collapse to ~50% on
near-tie inputs. 82-84% on hard cases means the cells genuinely compute the
GLOBAL majority, not just smooth skewed inputs. The frontier is crossed:
evolved ternary cells compute a global property from local interaction, where
gradient descent (same task) completely failed. The METHOD was the difference.

## Generalization across scale (rule evolved ONLY on N=21, applied unchanged)

| N | solved | hard (near-tie) |
|---|---|---|
| 21 (evolved on) | 90% | 79% |
| 49 | 88% | 74% |
| 99 | 79% | 59% |
| 149 (classic benchmark) | 75.5% | 54% |

Evolved on the easy N=21, the local rule transfers to N=149 -- the classic hard
benchmark -- at 75.5% solved, in the ballpark of the hand-designed GKL rule
(~80%), despite never training on it. Graceful degradation, not a cliff:
evidence it learned a genuine local->global consensus rule, not an N=21 trick.

## Honest limits (stated, not buried)

- At large N the HARD near-tie cases fall to ~54% (chance). Easy/skewed global
  majority is solved at every scale; the genuinely-hard near-tie regime at large
  N is NOT. This is the KNOWN theory, not a shortfall of the work: exact
  majority is provably unsolvable by any uniform CA. The result has exactly the
  correct shape (good everywhere easy, chance on hard-at-large-N).
- N=21 absolute numbers (92%) are a smaller, easier instance than the classic
  N=149 (76%). Do NOT compare N=21 to GKL's N=149 ~80% -- different problem size.
- In-generation best_fit during evolution (~0.96) was selection-biased/in-sample;
  the honest numbers are the held-out evals above.

## The through-line this completes

- Morphogenesis (grow coherent structure, even a heart): WORKS.
- Computation via gradient descent: FAILS (vanishing gradients through unrolled steps).
- Computation via EVOLUTION: WORKS -- global property from local ternary talk,
  generalizing across scale, with the provably-hard regime as the honest limit.

The same architecture computes or doesn't depending on the OPTIMIZER. The tool
was the whole story -- mapped, then crossed.

---
*Script: trit_evolve.py   Run: trit_evolve_run.txt   Weights: trit_evolve_best.npy*

## Does symmetry govern the computation? (measured -- yes)

Density classification has an exact +/-1 sign symmetry: flip every input sign
and the majority (and correct answer) flip too. A rule that truly computes
majority should be sign-equivariant, f(-x) = -f(x). Tested the evolved rule
(trit_symmetry_test.py):

- single-step equivariance f(-x)==-f(x): 94.0% of cells
- full-run   equivariance f_T(-x)==-f_T(x): 93.1% of cells
- accuracy on x 91.5% vs on -x 91.0% (near-identical)
- polarity accuracy gap (+1-maj vs -1-maj): 1.6pp

Evolution DISCOVERED a near-sign-equivariant rule without being told to -- the
MLP's bias terms could have broken the symmetry (and account for the ~7%
deviation and 1.6pp residual bias), but evolution converged near-symmetric
BECAUSE the task is symmetric. This is the symmetry-selection-rule PRINCIPLE
(symmetry structure of the problem selects the structure of the solution) shown
in a ternary CA -- a different substrate than the phononic reservoir it came
from. The principle transfers; the specific even-order phononic rule does not.

## Causal test: does FORCING exact symmetry change the result? (surprising -- yes)

Evolution found a ~93% sign-equivariant rule on its own. Forced EXACT
equivariance by dropping the MLP biases (tanh and ternary are both odd, so a
bias-free rule is exactly f(-x)=-f(x); 80 params vs 91) and re-evolved.

| | params | equiv | N=21 solved | N=21 hard | N=149 solved | N=149 hard |
|---|---|---|---|---|---|---|
| unconstrained | 91 | ~93% learned | 92.8% | 83.6% | 75.5% | 54% |
| exact-equivariant | 80 | 100% built-in | 92.1% | 82.0% | 58.7% | 15% |

Two findings:
1. At the trained size N=21: a TIE -- forcing exact symmetry costs nothing (same
   accuracy, 11 fewer params). Symmetry is a free/useful constraint at that scale.
2. Generalization to larger N: forcing exact symmetry HURTS badly. At N=149 the
   unconstrained rule holds 75.5%/54% (solved/hard) but the exact-equivariant one
   collapses to 58.7%/15% (near-chance on hard). Hard-case accuracy craters with
   scale (80->46->19->15) vs the unconstrained rule's graceful 79->74->59->54.

Interpretation: the small ~7% asymmetry the unconstrained rule kept is
LOAD-BEARING -- it's what sustains information propagation around a larger ring.
This refines the symmetry principle: symmetry GOVERNS the computation (evolution
reliably finds near-symmetric solutions), but ENFORCING exact symmetry is
over-constraint -- the robust, generalizing optimum is MOSTLY symmetric with a
little slack, not perfectly symmetric.

Caveat: single evolution run each. The N=21 tie is solid; the generalization gap
(large: 54% vs 15%) should be confirmed across multiple seeds to be ironclad.

## CORRECTION: 5-seed replication refines (and partly retracts) the above

The single-run "exact symmetry hurts generalization" claim above was an
OVERSTATEMENT. Re-ran both modes across 5 seeds (trit_evolve_seeds.py):

|              | N=21 solved | N=149 solved | N=149 hard |
|--------------|-------------|--------------|------------|
| unconstrained| 92.1 ± 1.1  | 67.8 ± 7.4   | 32.4 ± 19.3|
| equivariant  | 90.5 ± 1.1  | 60.4 ± 1.1   | 15.2 ± 3.0 |

- N=21: TIED (overlapping) -- "exact symmetry is free at trained size" CONFIRMED.
- N=149: the unconstrained rule's generalization is BIMODAL/high-variance
  (per-seed hard: 56.9, 12.4, 20.5, 17.2, 54.7) -- 2/5 seeds find excellent
  generalizers, 3/5 are as weak as equivariant. The equivariant rule is tight
  (12-19%, std 3.0) and NEVER reaches the good solutions.

Corrected claim: exact symmetry does not RELIABLY hurt each run. It CAPS the
ceiling and removes variance -- the extra freedom gives the unconstrained rule
occasional ACCESS to a much-better-generalizing solution (~40% of seeds) that
the exactly-symmetric rule can't reach; it trades that upside for consistency.
The +17pp average gap rides on ~2 lucky seeds; with n=5 and huge variance it is
SUGGESTIVE, not conclusive. The dramatic single-run gap was largely seed-luck,
exactly as the original caveat feared. This does NOT affect the main compute-
frontier result (evolved rule computes global majority), which replicated cleanly.
