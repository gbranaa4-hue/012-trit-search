"""
trit_compute -- does the grown/talking ternary CA actually COMPUTE, or just
look coherent? The honest frontier step from morphogenesis toward "a brain
that decides something."

Task: density / majority classification -- a classic, decades-studied
cellular-automaton computation benchmark. The whole grid starts as a random
+/-1 pattern; every cell must converge to whichever value was in the MAJORITY.
That requires GLOBAL information (the overall count) to emerge from purely
LOCAL neighbor talk -- no cell can see the whole grid. It's genuinely "cells
talk to reach one collective decision," and it's measurable.

Honest expectation, stated before running: uniform CAs provably cannot solve
exact majority perfectly; even good evolved CAs hit only ~75-85% on random
inputs, with near-50/50 splits the hard case. Expect partial success. Results
are reported SPLIT by input skew (easy skewed vs hard near-tie) vs the 50%
chance baseline -- "solves the easy ones, struggles at the tie" is the honest
likely outcome, not magic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from trit_grow import ternary_ste


class ComputeCA(nn.Module):
    def __init__(self, channels=12, hidden=96):
        super().__init__()
        self.channels = channels
        ident = torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]])
        sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        sy = sx.t()
        k = torch.zeros(channels * 3, 1, 3, 3)
        for c in range(channels):
            k[3 * c + 0, 0] = ident
            k[3 * c + 1, 0] = sx
            k[3 * c + 2, 0] = sy
        self.register_buffer("perc_kernel", k)
        self.update = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def step(self, x):
        comm = ternary_ste(x)                                   # cells talk in trits
        y = F.conv2d(comm, self.perc_kernel, padding=1, groups=self.channels)
        x = x + self.update(y)
        x = torch.cat([torch.tanh(x[:, :1]), x[:, 1:]], dim=1)  # bound the readout channel
        return x


def make_batch(batch, h, w, channels):
    """Random +/-1 pattern in channel 0 (density uniform in [0.25,0.75],
    excluding exact ties); hidden channels start at 0. Returns (x, majority)."""
    x = torch.zeros(batch, channels, h, w)
    dens = torch.empty(batch).uniform_(0.25, 0.75)
    for i in range(batch):
        x[i, 0] = (torch.rand(h, w) < dens[i]).float() * 2 - 1
    s = x[:, 0].sum(dim=(1, 2))
    maj = torch.sign(s)
    maj[maj == 0] = 1
    return x, maj


def train(steps=1000, h=14, w=14, run=40, channels=12):
    """Fixed recipe after the first run diverged (loss went UP): standard
    global-norm gradient clipping instead of per-parameter normalization
    (which destabilized this task), lower LR, more CA steps so global info can
    actually cross the grid (a 14-wide grid needs ~14+ steps for a signal to
    traverse; 40 gives room to integrate), and a slightly smaller grid to
    shorten the propagation distance."""
    torch.manual_seed(0)
    ca = ComputeCA(channels)
    opt = torch.optim.Adam(ca.parameters(), lr=1e-3)
    for step in range(steps):
        x, maj = make_batch(16, h, w, channels)
        for _ in range(run):
            x = ca.step(x)
        target = maj.view(-1, 1, 1, 1).expand(-1, 1, h, w)
        loss = F.mse_loss(x[:, :1], target)                    # push every cell to the majority
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ca.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss={loss.item():.4f}")
    return ca


@torch.no_grad()
def evaluate(ca, n=600, h=16, w=16, run=24, channels=12):
    """Real density-classification metric: CONSENSUS. After running, what
    fraction of cells agree with the true majority? Doing nothing leaves this
    at the initial majority density; genuine computation amplifies it toward
    1.0 (all cells agree). We report initial vs final consensus (the amount of
    amplification is the real signal), the 'solved' rate (>90% agree AND
    correct sign), split by input skew."""
    init_cons, final_cons, solved = [], [], 0
    buckets = {"easy (>=60/40)": [[], 0, 0], "hard (<60/40)": [[], 0, 0]}
    for _ in range(n):
        x, maj = make_batch(1, h, w, channels)
        skew = abs((x[:, 0] > 0).float().mean().item() - 0.5)
        init_frac = (torch.sign(x[:, 0]) == maj).float().mean().item()   # baseline: majority density
        for _ in range(run):
            x = ca.step(x)
        final_sign = torch.sign(x[:, 0])
        final_frac = (final_sign == maj).float().mean().item()           # how many cells reached majority
        ok = int(final_frac > 0.9)                                       # solved = near-unanimous & correct
        init_cons.append(init_frac); final_cons.append(final_frac); solved += ok
        b = "easy (>=60/40)" if skew >= 0.10 else "hard (<60/40)"
        buckets[b][0].append(final_frac); buckets[b][1] += ok; buckets[b][2] += 1

    import statistics as st
    print(f"\ninitial consensus (do-nothing baseline): {st.mean(init_cons)*100:.1f}% of cells at majority")
    print(f"final   consensus (after local ternary talk): {st.mean(final_cons)*100:.1f}% of cells at majority")
    print(f"solved (>90% agree, correct): {solved/n*100:.1f}%  ({solved}/{n})")
    for name, (fracs, s, t) in buckets.items():
        if t:
            print(f"  {name:16s}: final consensus {st.mean(fracs)*100:5.1f}%,  solved {s/t*100:5.1f}%  ({s}/{t})")


def main():
    print("Density/majority classification -- do the talking ternary cells compute a GLOBAL property?\n")
    H = W = 14
    RUN = 40
    ca = train(h=H, w=W, run=RUN)
    evaluate(ca, h=H, w=W, run=RUN)
    print("\nHonest read: >50% overall = better than chance = real (partial) global")
    print("computation from local ternary talk. Easy>>hard is expected (near-tie is")
    print("the provably-hard case). ~chance everywhere = looks coherent but doesn't compute.")


if __name__ == "__main__":
    main()
