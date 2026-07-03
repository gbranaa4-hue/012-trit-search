"""
Multi-seed confirmation of the surprising single-run finding: forcing EXACT
sign-equivariance ties the unconstrained rule at the trained size (N=21) but
HURTS generalization to larger rings (N=149). Single runs showed a large gap;
this replicates it across 5 seeds per mode to see if it's real or seed-luck.

For each seed x each mode: evolve at N=21, then evaluate solved-rate at N=21
(trained size) and N=149 (classic hard benchmark, unseen). Report mean +/- std.
"""
import numpy as np
import trit_evolve as te


def set_mode(equiv):
    te.EQUIVARIANT = equiv
    te.DIM = (te.IN * te.HID + te.HID) if equiv else (te.IN * te.HID + te.HID + te.HID + 1)


def solved_at(theta, Nv, seed):
    te.N = Nv
    te.rng = np.random.default_rng(5000 + seed)
    x, maj = te.make_configs(1200)
    fin = te.ca_run(theta, x.copy(), steps=2 * Nv)
    solved = ((np.sign(fin) == maj[:, None]).mean(axis=1) > 0.9).mean()
    skew = np.abs((x > 0).mean(axis=1) - 0.5)
    hard = ~(skew >= 0.12)
    sh = ((np.sign(fin[hard]) == maj[hard, None]).mean(axis=1) > 0.9).mean()
    return solved, sh


def run_one(equiv, seed):
    set_mode(equiv)
    te.N = 21
    te.rng = np.random.default_rng(seed)
    theta = te.evolve(gens=180, lam=40, mu=10)
    s21, h21 = solved_at(theta, 21, seed)
    s149, h149 = solved_at(theta, 149, seed)
    return s21, h21, s149, h149


def main():
    results = {"unconstrained": [], "equivariant": []}
    for seed in range(5):
        for equiv, name in [(False, "unconstrained"), (True, "equivariant")]:
            r = run_one(equiv, seed)
            results[name].append(r)
            print(f"seed {seed} {name:13s}: N21 solved {r[0]*100:4.1f} hard {r[1]*100:4.1f}  |  "
                  f"N149 solved {r[2]*100:4.1f} hard {r[3]*100:4.1f}", flush=True)

    print("\n" + "=" * 66)
    print("  SUMMARY (mean +/- std over 5 seeds)")
    print("=" * 66)
    for name in ("unconstrained", "equivariant"):
        a = np.array(results[name])
        print(f"  {name:14s}  N21 solved {a[:,0].mean()*100:4.1f}+/-{a[:,0].std()*100:3.1f}   "
              f"N149 solved {a[:,2].mean()*100:4.1f}+/-{a[:,2].std()*100:3.1f}   "
              f"N149 hard {a[:,3].mean()*100:4.1f}+/-{a[:,3].std()*100:3.1f}")
    u, e = np.array(results["unconstrained"]), np.array(results["equivariant"])
    print(f"\n  generalization gap (unconstrained - equivariant) at N=149:")
    print(f"    solved: {(u[:,2].mean()-e[:,2].mean())*100:+.1f}pp   hard: {(u[:,3].mean()-e[:,3].mean())*100:+.1f}pp")
    print("\n  Honest read: if unconstrained consistently > equivariant at N=149,")
    print("  the 'exact symmetry hurts generalization' finding replicates. If the")
    print("  spread overlaps, the single-run gap was seed-luck.")


if __name__ == "__main__":
    main()
