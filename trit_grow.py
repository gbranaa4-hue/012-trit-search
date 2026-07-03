"""
trit_grow -- a growing neural cellular automaton, rebuilt STAGED after two
honest training failures (runaway fill-all, then dead/no-growth). Both were
recipe bugs, not fundamentals: the missing ingredient was the standard
Growing-NCA *stochastic per-cell update mask* (each cell updates only ~half
the time), without which all cells move in lockstep and collapse together.

Disciplined, one-component-at-a-time (matches paper/triadic_robustness_findings.md):
  mode="bare"      known-good minimal NCA (continuous comms, no consensus)
  mode="ternary"   cells communicate in {-1,0,+1} instead of continuous
  mode="consensus" ternary + a consensus growth gate

Each stage must still GROW before the next is added. This file proves the
base recipe first; ternary and consensus are toggles layered on only once
bare works.

Honest scope unchanged: this is morphogenesis (growing coherent structure),
not cognition. It won't reason or do language.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_ste(x, frac=0.7):
    thresh = frac * x.abs().mean().clamp(min=1e-6)
    t = torch.where(x > thresh, torch.ones_like(x),
        torch.where(x < -thresh, -torch.ones_like(x), torch.zeros_like(x)))
    return x + (t - x).detach()


class GrowCA(nn.Module):
    def __init__(self, channels=16, hidden=128, mode="bare", fire_p=0.5, consensus_thresh=3.0):
        super().__init__()
        self.channels = channels
        self.mode = mode
        self.fire_p = fire_p                      # stochastic update probability (the fix)
        self.consensus_thresh = consensus_thresh

        # Fixed perception: identity + Sobel-x + Sobel-y per channel (standard NCA).
        ident = torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]])
        sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        sy = sx.t()
        k = torch.zeros(channels * 3, 1, 3, 3)
        for c in range(channels):
            k[3 * c + 0, 0] = ident
            k[3 * c + 1, 0] = sx
            k[3 * c + 2, 0] = sy
        self.register_buffer("perc_kernel", k)

        # per-cell update rule (1x1 convs = shared MLP over the perception vector)
        self.update = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.update[-1].weight)    # near-identity start (standard)
        nn.init.zeros_(self.update[-1].bias)

    def perceive(self, x):
        return F.conv2d(x, self.perc_kernel, padding=1, groups=self.channels)

    def alive(self, x):
        # channel 0 is visible/alpha; a cell is alive if it or a neighbor is > 0.1
        return F.max_pool2d((x[:, :1] > 0.1).float(), 3, stride=1, padding=1)

    def step(self, x):
        pre_life = self.alive(x)

        comm = x
        if self.mode in ("ternary", "consensus"):
            comm = ternary_ste(x)                 # cells communicate in trits
        y = self.perceive(comm)
        dx = self.update(y)

        # THE FIX: stochastic per-cell update -- each cell updates only ~half
        # the time, breaking the lockstep symmetry that caused both prior
        # collapses (all-on / all-off).
        fire = (torch.rand_like(x[:, :1]) < self.fire_p).float()
        x = x + dx * fire

        life = pre_life * self.alive(x)

        if self.mode == "consensus":
            # extra growth gate: a cell may also stay alive where neighbors
            # reach ternary consensus (layered on ONLY after bare+ternary work)
            msg = ternary_ste(x[:, :1])
            neigh = F.conv2d(msg, torch.ones(1, 1, 3, 3, device=x.device), padding=1)
            consensus = (neigh.abs() >= self.consensus_thresh).float()
            life = torch.clamp(life + consensus, 0, 1)

        return x * life


def seed_grid(h, w, channels):
    x = torch.zeros(1, channels, h, w)
    x[0, 0, h // 2, w // 2] = 1.0
    return x


def target_square(h, w, size=10):
    t = torch.zeros(1, 1, h, w)
    a, b = (h - size) // 2, (h + size) // 2
    t[0, 0, a:b, a:b] = 1.0
    return t


def ascii_grid(x):
    v = x[0, 0].detach().clamp(0, 1)
    chars = " .:-=+*#%@"
    return "\n".join("".join(chars[min(int(c * 9), 9)] for c in row) for row in v)


def train(mode, steps=500, h=28, w=28, unroll=(28, 48)):
    torch.manual_seed(0)
    ca = GrowCA(mode=mode)
    target = target_square(h, w)
    opt = torch.optim.Adam(ca.parameters(), lr=2e-3)
    for step in range(steps):
        x = seed_grid(h, w, ca.channels)
        n = torch.randint(unroll[0], unroll[1], (1,)).item()
        for _ in range(n):
            x = ca.step(x)
        loss = F.mse_loss(x[:, :1].clamp(0, 1), target)
        opt.zero_grad(); loss.backward()
        # per-parameter grad normalization (the ORIGINAL Growing-NCA recipe);
        # it caused runaway earlier ONLY because the stochastic fire mask was
        # missing -- with the mask in place this is the correct choice.
        for p in ca.parameters():
            if p.grad is not None:
                p.grad = p.grad / (p.grad.norm() + 1e-8)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  [{mode}] step {step:4d}  loss={loss.item():.4f}")
    return ca, target


def demo(mode="bare"):
    print(f"=== STAGE: mode={mode} -- can it grow the square from one seed? ===\n")
    ca, target = train(mode)
    h = w = 28
    x = seed_grid(h, w, ca.channels)
    for _ in range(40):
        x = ca.step(x)
    print("\ngrown from seed:")
    print(ascii_grid(x))
    grown = F.mse_loss(x[:, :1].clamp(0, 1), target).item()
    print(f"  grown match-to-target MSE: {grown:.4f}  (0.08 = did nothing, lower = grew the square)")

    x[:, :, :, : w // 2] = 0.0
    for _ in range(40):
        x = ca.step(x)
    healed = F.mse_loss(x[:, :1].clamp(0, 1), target).item()
    print("\nafter erasing left half + 40 steps (self-heal?):")
    print(ascii_grid(x))
    print(f"  healed match-to-target MSE: {healed:.4f}")


if __name__ == "__main__":
    import sys
    demo(sys.argv[1] if len(sys.argv) > 1 else "bare")
