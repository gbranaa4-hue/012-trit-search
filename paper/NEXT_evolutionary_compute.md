# NEXT: cross the compute frontier with evolutionary search (fresh-session spec)

Scoped handoff so this starts cleanly, not from scratch. The morphogenesis
side is done and works (grow_ca_findings.md). The computation side does NOT
work with gradient descent (compute_ca_findings.md) -- and that failure is
*expected*, which is the whole reason for this plan.

## Why gradient descent failed (don't repeat it)

Backprop through tens of unrolled CA steps has vanishing/exploding gradients
-- the loss literally went UP and diverged in every compute attempt (density
classification and conditional output alike). This is a known property, not a
tuning miss. The field solves CA computation with EVOLUTIONARY search, not
backprop (the classic GKL rule for density classification was hand-designed;
good rules since are evolved).

## The plan: evolve the rule, don't backprop it

- Keep the exact same architecture idea (a small shared local rule, ternary
  messages between cells) but make the rule's parameters SMALL enough to evolve
  (a few hundred to low-thousands of weights -- e.g. a tiny MLP, not the 128-hidden
  conv stack, which has too many params for CMA-ES).
- Optimizer: CMA-ES (cma package) or a simple (mu, lambda) evolution strategy /
  genetic algorithm. No gradients at all -- fitness is measured by running the CA.
- Fitness = the honest metric already built in trit_compute.py: consensus
  amplification (fraction of cells reaching the true majority after N steps),
  averaged over many random inputs. Start with a curriculum: easy skewed
  densities first, then harder near-50/50.

## Concrete first target (density classification)

- Grid: start SMALL (e.g. 1D ring of ~21 cells -- the classic setup -- or a
  small 2D grid). 1D is the canonical density-classification testbed and far
  cheaper to evolve than 2D.
- Steps: ~2x grid size (info must cross).
- Success bar, honest: the literature ceiling is ~75-85% on random inputs; the
  GKL rule ~81.6%. Beating 50% chance meaningfully (say >65%) with an EVOLVED
  ternary rule is already a real "the cells compute a global property" result.
  Do NOT expect ~100% -- exact majority is provably unsolvable by a uniform CA.

## What "success" would prove

That the talking ternary cells can compute a genuine global property from local
interaction once the rule is found by the RIGHT method (evolution) rather than
the wrong one (backprop). That's the honest crossing of the frontier this
session mapped but did not cross.

## Reuse

- trit_compute.py: the task, the honest consensus metric, make_batch().
- trit_grow.py: ternary_ste, the CA step structure (adapt to a smaller
  evolvable rule; drop the alive-mask -- compute uses a full active grid).
