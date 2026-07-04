"""
Sharper follow-up: are the 'flooders' actually balanced-and-strong at the sizes
they TRAINED on, and only collapse to a polarity default when extrapolating past
their range? Measure each informative rule at 21/49/99/149/299, per-polarity.
Same seed -> reproduces the exact rules from the ca_levers scouting run.
"""
import numpy as np
import ca_levers as cl

RULES = [
    ("2L  @ 21",          cl.rule2, cl.DIM2, [21]),
    ("2L  @ 21,49,99",    cl.rule2, cl.DIM2, [21, 49, 99]),
    ("1N  @ 21,49,99",    cl.rule1, cl.DIM1, [21, 49, 99]),
]

for name, rule, DIM, sizes in RULES:
    theta = cl.evolve(rule, DIM, sizes, np.random.default_rng(0), gens=100)
    trained = set(sizes)
    print(f"\n{name}   (trained on {sizes})")
    print(f"   {'N':>4} {'hard':>7} {'+1':>7} {'-1':>7} {'gap':>6}")
    for Nt in (21, 49, 99, 149, 299):
        h, hp, hn = cl.hard_at(rule, theta, Nt, np.random.default_rng(9))
        tag = " <-trained" if Nt in trained else ""
        print(f"   {Nt:>4} {h:6.1f}% {hp:6.1f}% {hn:6.1f}% {abs(hp-hn):5.1f}{tag}", flush=True)

print("\nRead: if the +1/-1 GAP is small AT trained sizes and only explodes past")
print("the training range, the flooder is an EXTRAPOLATION limit. If the gap is")
print("large even at trained sizes, training genuinely failed to balance it.")
