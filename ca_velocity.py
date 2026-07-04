"""
Closing the from-scratch loop: no cheap lever made the rule robust (ca_levers).
The lever that worked in the main investigation was REWARDING SIGNAL VELOCITY.
Reproduce that here, in the from-scratch framework.

velocity(): a gaming-proof domain-wall probe -- seed a minority block, time how
fast the CORRECT majority takes it over, BOTH polarities, take the MIN (a sign-
biased flooder scores 0). Add lambda_v * velocity to the fitness and see if it
produces BALANCED generalizers (small +1/-1 gap, high min-hard) at N=149, where
plain training gave biased flooders.
"""
import numpy as np
import ca_levers as cl


def velocity(rule, theta, Nsz=35, trials=8):
    step = rule(theta)
    W, c = Nsz // 3, Nsz // 2
    target, init_maj = 0.9 * Nsz, Nsz - Nsz // 3
    conv = (target - init_maj) / 2.0
    speeds = []
    for minority in (-1.0, 1.0):
        maj_sign = -minority
        s = np.full((trials, Nsz), maj_sign)
        s[:, (np.arange(W) + (c - W // 2)) % Nsz] = minority
        reached = np.full(trials, np.nan)
        cur = s.copy()
        for t in range(1, Nsz + 1):
            cur = step(cur)
            mc = (np.sign(cur) == maj_sign).sum(axis=1)
            reached[np.isnan(reached) & (mc >= target)] = t
        sp = np.where(np.isnan(reached), 0.0, conv / np.maximum(reached, 1))
        speeds.append(sp.mean())
    return min(speeds)                                   # worse polarity -> anti-gaming


def velocity_dense(rule, theta, Nsz=35):
    """DENSE version: reward partial both-polarity progress, not binary resolution.
    Per polarity: how much closer to correct consensus the block got (0 = did
    nothing, 1 = fully resolved, <0 = made it worse). min over polarities keeps
    it anti-flooder, but it gives a gradient to half-working rules."""
    step = rule(theta)
    W, c = Nsz // 3, Nsz // 2
    init = (Nsz - W) / Nsz                               # correct-fraction of the start
    imps = []
    for minority in (-1.0, 1.0):
        maj_sign = -minority
        s = np.full((6, Nsz), maj_sign)
        s[:, (np.arange(W) + (c - W // 2)) % Nsz] = minority
        cur = s.copy()
        for _ in range(Nsz):
            cur = step(cur)
        final = (np.sign(cur) == maj_sign).mean()
        imps.append((final - init) / (1 - init))        # normalized improvement
    return min(imps)


def evolve_vel(rule, DIM, lam_v, seed, gens=40, lam=25, mu=7, sigma=0.5, batch=120, Ntrain=21):
    rng = np.random.default_rng(seed)
    center = rng.standard_normal(DIM) * 0.3
    best, best_s = center.copy(), -1.0
    for g in range(gens):
        x, maj = cl.configs(batch, Ntrain, rng)
        pop = center + sigma * rng.standard_normal((lam, DIM))
        sc = np.empty(lam)
        for i, p in enumerate(pop):
            f = cl.score(rule, p, x, maj, Ntrain)
            if lam_v != 0:
                f += lam_v * velocity_dense(rule, p)     # dense both-polarity progress reward
            sc[i] = f
        top = np.argsort(-sc)[:mu]
        center = pop[top].mean(axis=0)
        if sc[top[0]] > best_s:
            best_s, best = sc[top[0]], pop[top[0]].copy()
        sigma *= 0.995
    return best


if __name__ == "__main__":
    print("does rewarding VELOCITY produce BALANCED generalizers in the from-scratch build?")
    print("2-layer rule, trained at N=21 only; measured on hard@149 split by polarity.\n")
    for name, lam_v in [("baseline (no velocity)", 0.0), ("+ dense velocity (lam=0.3)", 0.3)]:
        print(name)
        for seed in range(2):
            th = evolve_vel(cl.rule2, cl.DIM2, lam_v, seed)
            h, hp, hn = cl.hard_at(cl.rule2, th, 149, np.random.default_rng(9), batch=1500)
            print(f"  seed {seed}: hard@149 {h:5.1f}%  +1 {hp:5.1f}  -1 {hn:5.1f}  "
                  f"min {min(hp, hn):5.1f}  gap {abs(hp - hn):5.1f}", flush=True)
    print("\nWin = velocity rows have HIGHER min (both polarities solved) and SMALLER gap")
    print("than baseline. That reproduces the causal result in the from-scratch code.")
