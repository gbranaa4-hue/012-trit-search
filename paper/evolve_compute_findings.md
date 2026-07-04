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

## RESOLVED: the bimodality is a MECHANISM (signal velocity), not seed-luck

The "seed luck" above was unsatisfying -- WHY do ~1/5 seeds generalize to N=149
and the rest plateau? CA computational-mechanics theory (Crutchfield/Hanson/Das)
says good density classifiers compute by propagating signals/particles (domain
walls) across the ring; a rule can only integrate global info on a big ring if
those signals travel fast enough to cross it in ~2N steps. So a rule's SIGNAL
VELOCITY -- measurable at SMALL N without ever seeing N=149 -- should predict its
N=149 generalization. Tested it (trit_evolve_predict.py).

Probe (independent of the task): seed a MINORITY block in a majority background;
measure how fast the rule closes it by expanding the CORRECT-majority domain
(steps to reach 0.9N at the true majority, both polarities averaged). A rule that
collapses to 0 or to noise never reaches correct consensus -> velocity 0 (the
measure had to be hardened twice: single-cell damage heals in a convergent rule,
and "block disappeared" was fooled by collapse-to-zero -- only correct-majority
takeover counts).

Evolved 16 unconstrained seeds; velocity vs N=149 hard-solved:

| | velocity | N=149 hard |
|---|---|---|
| seeds 0, 4, 13 | 0.12 | 59.8, 51.1, 51.7 % |
| other 13 seeds | 0.00 | 13-21 % |

- **corr(velocity, N149 hard) = +0.981** -- but honestly it is a near-PERFECT
  BINARY SEPARATOR: all 3 nonzero-velocity seeds are the top-3 generalizers
  (51-60%), all 13 zero-velocity seeds cluster at 13-21%, no overlap.
- **Robust to probe scale:** measuring velocity at N=35 or N=49 (a quarter of the
  test ring) flags the SAME 3 seeds, same +0.981. The predictor is genuinely a
  small-N measurement predicting large-N behavior it never saw.
- The 3 generalizers all landed at the SAME velocity (0.12): looks like a
  discrete solution basin evolution either finds (~3/16 ~ 19%) or misses.
- **asymmetry corr = -0.142** -- the 5-seed note's guess that "a little asymmetry
  is load-bearing" is NOT supported. Asymmetry is unrelated; velocity is the
  variable. **N=21 skill corr = +0.659** -- trained-size accuracy is a WEAK
  predictor (most rules solve N=21 regardless).

Result: the bimodality is EXPLAINED. Generalization to a big ring is not luck --
it is whether evolution discovered a rule with propagating domain walls (nonzero
signal velocity) vs one doing local smoothing that cannot cross the ring. This
matches the classic particle-based-computation picture, now shown in an evolved
TERNARY CA and made predictive from a small-N probe.

Honest limits: velocity=0 means "does not resolve the block probe within its
budget," not literally zero propagation; base rate of the good basin here is
~19% (3/16), below the earlier small-sample 2/5. The main compute result is
unaffected -- this ADDS the mechanism behind its scale-generalization.

---
*Script: trit_evolve_predict.py   Run: trit_evolve_predict_run.txt*

## Confirmed OUT-OF-SAMPLE + the winners are a distinct rule FAMILY

The velocity->generalization link was calibrated on N=149. Two falsifiable
follow-ups (trit_evolve_scale.py), reusing the 16 saved rules, no re-evolution:

**(A) Does small-N velocity predict at rings never looked at?** Evaluated all 16
rules at N=149, 299, 499:

| ring | corr(vel, hard) | winners | others | gap |
|---|---|---|---|---|
| 149 (calibrated) | +0.985 | 56.1% | 16.3% | +39.8pp |
| 299 (unseen) | +0.971 | 48.9% | 10.6% | +38.3pp |
| 499 (unseen) | +0.986 | 55.9% |  7.2% | +48.7pp |

Velocity (probed at N=49) predicts generalization at 2x-3x the calibration ring
at corr ~0.97-0.99, and the winner/other gap WIDENS with scale. Out-of-sample:
the mechanism is not a fit to N=149.

**Surprise (a worry refuted):** I expected the winners (velocity 0.12) to COLLAPSE
at large N -- 0.12 cells/step seemed too slow to cross a 499-ring in ~2N steps.
They did NOT. Per-winner hard-solve is essentially SCALE-INVARIANT:

| seed | N149 | N299 | N499 |
|---|---|---|---|
| 0 | 55 | 50 | 52 |
| 4 | 59 | 52 | 62 |
| 13 | 54 | 45 | 53 |

Flat ~50-60% (within sampling noise) to N=499 while others crater to ~7%. The
naive "velocity budget" arithmetic was too literal; these rules found a genuinely
SCALE-FREE majority computation. Stronger than expected, not weaker.

**(B) Are the 3 winners one solution?** The rule is a pure function of a 7-cell
ternary neighborhood, so its COMPLETE behavior is a 3^7 = 2187-entry table
(permutation-invariant, exact). Pairwise agreement:

- winner-winner: 0.817   winner-loser: 0.202   loser-loser: 0.716

The winners agree 82% with each other but only 20% with losers (below the ~33%
equal-density-random baseline -- the two groups give OPPOSITE outputs on many
neighborhoods). Evolution finds one of (at least) TWO qualitatively different
rule FAMILIES; only the ~19% "winner" family has propagating signals and
generalizes. Honest calibration: 0.817 != 1.0, so the winners are a shared
BASIN/FAMILY, not the identical CA; and the table counts all 2187 neighborhoods
incl. ones rarely visited in real dynamics (structural fingerprint, not
behavioral identity). The winner/loser split is stark under any reading.

Net: the crown result's scale-generalization is now (1) mechanistically explained
by signal velocity, (2) confirmed OUT-OF-SAMPLE at N=299/499, (3) scale-invariant
for the winner family, (4) traced to a structurally distinct, reproducibly-found
solution basin. What was "seed luck" is a named, measurable, predictive mechanism.

[!] Points (2) and (3) are PARTLY RETRACTED below -- a stricter per-polarity
metric shows the "winners" were sign-biased, so their scale-invariance was on
the FAVORED polarity only. Read the correction.

---
*Scripts: trit_evolve_predict.py, trit_evolve_scale.py   Runs: *_run.txt*

## CORRECTION (polarity) + CAUSAL TEST: what velocity actually bought

A stricter, gaming-proof metric did two things: it PARTLY RETRACTED the
out-of-sample section above, and it delivered a genuine causal result -- with an
honest price tag. (trit_evolve_causal.py)

### Correction: the "winners" were POLARITY-BIASED, not balanced

Density classification is sign-symmetric: +1-majority and -1-majority inputs are
mirror problems, and a rule that truly computes majority should be equally good
on both. The out-of-sample section reported winners at "~55% N=149 hard" -- but
that was the MEAN over both polarities, and a mean HIDES imbalance. Per-polarity:

- prior "winner" rules at N=149: ~100% on +1-majority near-ties, ~9-18% on
  -1-majority near-ties. The celebrated ~55% = (100 + ~15) / 2.

So on genuinely hard (near-tie) large-ring cases they do NOT compute the
majority -- they DEFAULT to a preferred sign and get credit whenever that sign
happens to win. They ARE sign-equivariant at the trained size N=21 (measured
correctly earlier); the small ~7% asymmetry DOMINATES at N=149. "Scale-invariant
majority computation / out-of-sample confirmed" was OVERSTATED: scale-invariant
on the FAVORED polarity only. Honest metric = min(hard+, hard-), the worse side.

### Causal test: does SELECTING for velocity PRODUCE balanced generalizers?

Correlation (+0.98) and out-of-sample prediction showed velocity PREDICTS
generalization. Causal question: does OPTIMIZING it PRODUCE generalization?
Paired ("twins") design: for each of 12 seeds, two rules from IDENTICAL
initialization seeing IDENTICAL configs; the ONLY difference is the fitness --
baseline = consensus; treatment = consensus + 2.0 * signal_velocity(reduce=min).
The 'min' over polarities is gaming-proof (a sign-biased flooder scores 0; a
smoke test caught exactly that flooder under mean-reduction and rejected it).
The OUTCOME scored is the real N=149 per-polarity hard (min), not the probe, so
a rule cannot win by gaming the probe.

| | balanced generalizers (min-hard>30%) | mean min-hard | N=21 solved |
|---|---|---|---|
| baseline  | 0/12  | 13.7% | 89-93% |
| treatment | 11/12 | 40.7% | 50-91% (several ~50%) |

Hit rate 0% -> 91.7% (+91.7pp). Under the twins design nothing else differs, so
the velocity reward CAUSED the balanced generalizers. Real causal effect.

### Honest fences -- real, but NOT a clean win

1. TRADE, not upgrade. Treatment bought balanced large-ring behavior by SELLING
   trained-size accuracy: several treatment rules are near coin-flip (~50%) at
   N=21, the size they were bred on. Goodhart -- optimize a proxy hard and you
   get it by robbing what you did not protect.
2. The reward had to DOMINATE. Treatment fitness reached 1.6-2.0 (consensus max
   ~0.97), so velocity was the PRIMARY objective, not a nudge. What is shown is
   "making velocity the main goal reshapes the rules," NOT "a gentle bonus helps."
3. Probe/outcome mismatch. The post-hoc velocity probe (min, N=49) reads ~0.01
   for the treatment rules even though they generalize balanced at N=149. So
   SOMETHING velocity-adjacent was steered, but 'velocity' is not cleanly
   confirmed as the exact causal quantity. Loose thread, stated not hidden.
4. Not perfect: 1/12 treatment rule stayed biased (gen=False); some generalizers
   still lean (e.g. +33/-93).

Defensible claim: rewarding signal velocity CAUSALLY shifts evolved ternary rules
toward BALANCED both-polarity large-ring behavior that plain consensus never
finds (0/12 -> 11/12, controlled) -- the mechanism can be STEERED, not just
observed -- at a measured cost to trained-size accuracy. Celebrate the 11/12 and
name the coin-flip price in the same breath, or it is the "it's just luck" doctor.

---
*Script: trit_evolve_causal.py   Run: trit_evolve_causal_run.txt*
