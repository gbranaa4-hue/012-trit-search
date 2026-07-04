"""
VELOCITY SPECTRUM -- the "list of velocities," not one blurry average.

The single-scalar velocity probe (trit_evolve_predict.py) collapses a rule to one
number. But CA computation (Crutchfield/Hanson/Das computational mechanics) runs
on a CAST of moving fronts ("particles"), each with its OWN velocity, and the
computation lives in their collisions. A single average is blind to that -- and
that lossiness is the likely reason the scalar probe read ~0 on rules that
clearly transport information.

Method (dependency-free): run the rule from random init, record the space-time
field S(t,x), take its 2D FFT. A feature moving v cells/step concentrates power
along temporal_freq = v * spatial_freq, i.e. v = m/n in fft-frequency units. Bin
the power by implied velocity -> a velocity spectrum. Peaks = characteristic
front speeds. Max causal speed is the radius R=3, so real particles have |v|<=3.

Validate on KNOWN cases first (Q4): do-nothing -> all static; random -> broad/no
peak; a genuine computing rule -> sharp nonzero-velocity peak(s).
"""
import numpy as np
import trit_evolve as te

te.EQUIVARIANT = False
te.DIM = te.IN * te.HID + te.HID + te.HID + 1


def velocity_spectrum(theta, N=149, T=None, burn=0, trials=6, seed=0, vmax=3.0, nbins=61):
    """Record the TRANSIENT (the task-length approach to consensus, where the
    particles actually move) from random inits, and bin its space-time power by
    implied velocity. A converged rule freezes -> that shows as static (v~0)
    energy AFTER its moving transient, which is itself an honest fingerprint."""
    te.N = N
    T = T or 2 * N
    rng = np.random.default_rng(seed)
    vgrid = np.linspace(-vmax, vmax, nbins)
    spec = np.zeros(nbins)
    m = np.fft.fftfreq(T)[:, None]            # temporal freq, cycles/step
    n = np.fft.fftfreq(N)[None, :]            # spatial freq, cycles/cell
    with np.errstate(divide="ignore", invalid="ignore"):
        v = m / n                            # implied velocity, cells/step (v=inf where n=0)
    valid = np.isfinite(v) & (np.abs(v) <= vmax)
    valid &= (np.abs(m) < 0.499) & (np.abs(n) < 0.499)   # drop period-2 flicker, not transport
    v_safe = np.where(np.isfinite(v), v, 0.0)
    idx = np.clip(np.round((v_safe - (-vmax)) / (2 * vmax) * (nbins - 1)).astype(int), 0, nbins - 1)
    for tr in range(trials):
        d = rng.uniform(0.35, 0.65)
        s = (rng.random((1, N)) < d).astype(np.float64) * 2 - 1
        if burn:
            s = te.ca_run(theta, s, steps=burn)
        rows = []
        for _ in range(T):
            s = te.ca_run(theta, s, steps=1)
            rows.append(s[0].copy())
        S = np.array(rows)
        if S.std() < 1e-9:                    # dead / instantly frozen -> no spectrum
            continue
        S = S - S.mean()
        P = np.abs(np.fft.fft2(S)) ** 2
        np.add.at(spec, idx[valid], P[valid])
    tot = spec.sum()
    return vgrid, (spec / tot if tot > 0 else spec)


def peaks(vgrid, spec, k=3, min_sep=0.35):
    cand = [i for i in range(1, len(spec) - 1) if spec[i] > spec[i - 1] and spec[i] >= spec[i + 1]]
    cand.sort(key=lambda i: -spec[i])
    chosen = []
    for i in cand:
        if all(abs(vgrid[i] - vgrid[j]) >= min_sep for j in chosen):
            chosen.append(i)
        if len(chosen) >= k:
            break
    return [(round(float(vgrid[i]), 2), round(float(spec[i]), 3)) for i in chosen]


def bands(vgrid, spec):
    static = spec[np.abs(vgrid) < 0.25].sum()
    moving = spec[(np.abs(vgrid) >= 0.25)].sum()
    return round(float(static), 3), round(float(moving), 3)


def describe(name, theta, scalar=None):
    vg, sp = velocity_spectrum(theta)
    st, mv = bands(vg, sp)
    pk = peaks(vg, sp)
    extra = f"  [scalar-probe said {scalar}]" if scalar is not None else ""
    print(f"  {name:16s}: static={st:.2f} moving={mv:.2f}   peaks(v,power)={pk}{extra}")


def main():
    print("SANITY (known cases) -- validate the spectrum before trusting it:\n")
    rng = np.random.default_rng(42)
    describe("do-nothing", np.zeros(te.DIM))
    describe("random", rng.standard_normal(te.DIM) * 0.3)

    print("\nEVOLVED rules (predict_theta_*.npy from the 16-seed run):\n")
    # seeds 0,13 were generalizers (scalar-mean vel 0.12; scalar-MIN read 0.00);
    # seeds 1,2 were non-generalizers (scalar 0.00).
    for s, tag, sc in [(0, "winner s0", "0.12 / min 0"), (13, "winner s13", "0.12 / min 0"),
                       (1, "loser s1", "0.00"), (2, "loser s2", "0.00")]:
        describe(tag, np.load(f"predict_theta_{s}.npy"), scalar=sc)

    print("\n  Read: winners should show a SHARP nonzero-velocity peak (a moving")
    print("  particle) that the scalar-min probe called ~0. Losers should pile")
    print("  energy at v~0 (static) or show no coherent peak. If so, the spectrum")
    print("  sees the transport the single number missed.")


if __name__ == "__main__":
    main()
