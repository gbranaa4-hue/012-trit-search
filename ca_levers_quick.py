"""Lean version of the follow-up: same question, far less compute."""
import numpy as np
import ca_levers as cl


def evolve_lean(rule, DIM, sizes, rng, gens=50, lam=30, mu=8, sigma=0.5, batch=100):
    center = rng.standard_normal(DIM) * 0.3
    best, best_s = center.copy(), -1.0
    for g in range(gens):
        Nsz = int(sizes[rng.integers(len(sizes))])
        x, maj = cl.configs(batch, Nsz, rng)
        pop = center + sigma * rng.standard_normal((lam, DIM))
        sc = np.array([cl.score(rule, p, x, maj, Nsz) for p in pop])
        top = np.argsort(-sc)[:mu]
        center = pop[top].mean(axis=0)
        if sc[top[0]] > best_s:
            best_s, best = sc[top[0]], pop[top[0]].copy()
        sigma *= 0.995
    return best


for name, rule, DIM in [("2L @ 21,49,99", cl.rule2, cl.DIM2), ("1N @ 21,49,99", cl.rule1, cl.DIM1)]:
    theta = evolve_lean(rule, DIM, [21, 49, 99], np.random.default_rng(0))
    print(f"\n{name}  (trained on 21,49,99)")
    print(f"   {'N':>4} {'hard':>7} {'+1':>7} {'-1':>7} {'gap':>6}")
    for Nt in (21, 49, 99, 149):
        h, hp, hn = cl.hard_at(rule, theta, Nt, np.random.default_rng(9), batch=1200)
        tag = " <-trained" if Nt in (21, 49, 99) else ""
        print(f"   {Nt:>4} {h:6.1f}% {hp:6.1f}% {hn:6.1f}% {abs(hp - hn):5.1f}{tag}", flush=True)
