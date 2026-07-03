"""
Does the SYMMETRY idea from the phononic selection-rule work apply to the
evolved ternary density-classification rule? Made empirical.

Density classification has an exact +/-1 (sign) symmetry: flip every cell's
sign and a majority-(+1) input becomes a majority-(-1) input, with the correct
answer flipped too. A rule that genuinely "computes majority" should be
SIGN-EQUIVARIANT: f(-x) = -f(x), and its accuracy should be identical for both
polarities. If it is, symmetry genuinely governs what the rule computes
(echoing "symmetry as a selection rule for computation"). If it broke the
symmetry to work, that's its own finding. This measures it.
"""
import numpy as np
import trit_evolve as te

theta = np.load("trit_evolve_best.npy")


def solved(final, maj):
    return (np.sign(final) == maj[:, None]).mean(axis=1) > 0.9


def main():
    # 1. single-step rule oddness: does f(-x) == -f(x) exactly?
    te.rng = np.random.default_rng(0)
    x, _ = te.make_configs(1000)
    fx = te.ca_run(theta, x.copy(), steps=1)
    fnx = te.ca_run(theta, (-x).copy(), steps=1)
    step_equiv = (fnx == -fx).mean()

    # 2. full-run behavioral: accuracy on x vs sign-flipped x, and per polarity
    x, maj = te.make_configs(4000)
    fin = te.ca_run(theta, x.copy())
    fin_neg = te.ca_run(theta, (-x).copy())

    acc_x = solved(fin, maj).mean()
    acc_negx = solved(fin_neg, -maj).mean()               # flipped input -> flipped majority
    pos = maj > 0
    acc_pos = solved(fin, maj)[pos].mean()                # accuracy on +1-majority inputs
    acc_neg = solved(fin, maj)[~pos].mean()               # accuracy on -1-majority inputs

    # 3. full-run equivariance: is the whole trajectory's output f_T(-x) == -f_T(x)?
    full_equiv = (np.sign(fin_neg) == -np.sign(fin)).mean()

    print("=" * 62)
    print("  SYMMETRY-EQUIVARIANCE OF THE EVOLVED DENSITY-CLASSIFICATION RULE")
    print("=" * 62)
    print(f"  single-step  f(-x) == -f(x)  : {step_equiv*100:.1f}% of cells")
    print(f"  full-run     f_T(-x) == -f_T(x): {full_equiv*100:.1f}% of cells")
    print(f"\n  accuracy (solved rate):")
    print(f"    original inputs x        : {acc_x*100:.1f}%")
    print(f"    sign-flipped inputs -x   : {acc_negx*100:.1f}%   (should match if symmetric)")
    print(f"    on +1-majority inputs    : {acc_pos*100:.1f}%")
    print(f"    on -1-majority inputs    : {acc_neg*100:.1f}%   (gap = broken symmetry / polarity bias)")
    print(f"\n  polarity accuracy gap: {abs(acc_pos-acc_neg)*100:.1f}pp")
    print("\n  Honest read: high equivariance + equal polarity accuracy = the rule")
    print("  RESPECTS the task's sign symmetry (symmetry governs the computation).")
    print("  Large polarity gap / low equivariance = it broke the symmetry to solve.")


if __name__ == "__main__":
    main()
