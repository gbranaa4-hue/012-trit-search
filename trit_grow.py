"""
trit_grow -- a growing neural cellular automaton. Neurons spawn from one
seed, communicate with neighbors only in trits {-1,0,+1}, and self-organize
into one coherent whole.

History (honest): two early failures (runaway fill-all, then dead/no-growth)
were a missing stochastic per-cell update mask. Rebuilt staged; bare and
ternary communication both grow coherent shapes (see paper/grow_ca_findings.md).
Two things were then fixed:

  1. Consensus gate. The first consensus rule ADDED life (life += consensus),
     pure positive feedback -> runaway fill-all. Redesigned: the standard
     alive-mask stays the growth BRAKE (it's what bounds bare/ternary), and
     the neighbor-consensus signal is fed as an INPUT to the update rule --
     agreement shapes how a cell grows, it never unilaterally keeps a cell alive.

  2. Self-heal. Training only from the seed taught growth but not repair.
     Added persistent-pool training with damage: train from a pool of grown
     (and randomly damaged) states so the CA learns to HOLD and REGENERATE a
     pattern, not just grow it once.

modes: bare / ternary / consensus.  trainers: from-seed (train) / pool (train_pool).
Honest scope: morphogenesis (growing structure), not cognition.
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
    def __init__(self, channels=16, hidden=128, mode="bare", fire_p=0.5):
        super().__init__()
        self.channels = channels
        self.mode = mode
        self.fire_p = fire_p

        ident = torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]])
        sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        sy = sx.t()
        k = torch.zeros(channels * 3, 1, 3, 3)
        for c in range(channels):
            k[3 * c + 0, 0] = ident
            k[3 * c + 1, 0] = sx
            k[3 * c + 2, 0] = sy
        self.register_buffer("perc_kernel", k)
        self.register_buffer("neigh_kernel", torch.ones(1, 1, 3, 3))

        # consensus mode feeds one extra input channel (neighbor consensus)
        in_ch = channels * 3 + (1 if mode == "consensus" else 0)
        self.update = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def perceive(self, x):
        return F.conv2d(x, self.perc_kernel, padding=1, groups=self.channels)

    def alive(self, x):
        return F.max_pool2d((x[:, :1] > 0.1).float(), 3, stride=1, padding=1)

    def step(self, x):
        pre_life = self.alive(x)

        comm = ternary_ste(x) if self.mode in ("ternary", "consensus") else x
        y = self.perceive(comm)

        if self.mode == "consensus":
            # signed neighbor consensus of the visible channel, as an INPUT
            # feature only. It informs the update; it does NOT touch the life
            # mask (which stays the growth brake -- this is the runaway fix).
            msg = ternary_ste(x[:, :1])
            neigh = F.conv2d(msg, self.neigh_kernel, padding=1)   # signed, ~[-9,9]
            y = torch.cat([y, neigh], dim=1)

        dx = self.update(y)
        fire = (torch.rand_like(x[:, :1]) < self.fire_p).float()   # stochastic update (the fix)
        x = x + dx * fire

        life = pre_life * self.alive(x)   # standard bounded growth -- unchanged brake
        return x * life


def seed_batch(h, w, channels, batch=1):
    x = torch.zeros(batch, channels, h, w)
    x[:, 0, h // 2, w // 2] = 1.0
    return x


def target_square(h, w, size=10):
    t = torch.zeros(1, 1, h, w)
    a, b = (h - size) // 2, (h + size) // 2
    t[0, 0, a:b, a:b] = 1.0
    return t


def target_shape(name, h, w):
    """Non-trivial targets -- the point is to show the ternary CA
    self-organizes real STRUCTURE, not just a filled square."""
    t = torch.zeros(1, 1, h, w)
    cy, cx = h / 2, w / 2
    for i in range(h):
        for j in range(w):
            if name == "square":
                if abs(i - cy) < 5 and abs(j - cx) < 5:
                    t[0, 0, i, j] = 1.0
            elif name == "ring":          # hollow square -- must NOT fill center
                dy, dx = abs(i - cy), abs(j - cx)
                if 5 <= max(dy, dx) <= 8:
                    t[0, 0, i, j] = 1.0
            elif name == "cross":
                if (abs(i - cy) < 2 and abs(j - cx) < 9) or (abs(j - cx) < 2 and abs(i - cy) < 9):
                    t[0, 0, i, j] = 1.0
            elif name == "heart":
                x = (j - cx) / (w * 0.30)
                y = -(i - cy) / (h * 0.30) + 0.35
                if (x * x + y * y - 1) ** 3 - x * x * y * y * y <= 0:
                    t[0, 0, i, j] = 1.0
    return t


def damage(x):
    """Zero a random half of each sample -- teaches repair."""
    x = x.clone()
    B, _, h, w = x.shape
    for i in range(B):
        side = torch.randint(0, 4, (1,)).item()
        if side == 0:   x[i, :, : h // 2, :] = 0
        elif side == 1: x[i, :, h // 2:, :] = 0
        elif side == 2: x[i, :, :, : w // 2] = 0
        else:           x[i, :, :, w // 2:] = 0
    return x


def ascii_grid(x):
    v = x[0, 0].detach().clamp(0, 1)
    chars = " .:-=+*#%@"
    return "\n".join("".join(chars[min(int(c * 9), 9)] for c in row) for row in v)


def _grad_normalize(ca):
    for p in ca.parameters():
        if p.grad is not None:
            p.grad = p.grad / (p.grad.norm() + 1e-8)


def train_seed(mode, steps=500, h=28, w=28, unroll=(28, 48), shape="square"):
    """From-seed trainer -- the one that verifiably grows (bare/ternary).
    Kept so the consensus REDESIGN can be tested in isolation from the
    (separately-finicky) pool trainer."""
    torch.manual_seed(0)
    ca = GrowCA(mode=mode)
    target = target_shape(shape, h, w)
    opt = torch.optim.Adam(ca.parameters(), lr=2e-3)
    for step in range(steps):
        x = seed_batch(h, w, ca.channels, 1)
        n = torch.randint(unroll[0], unroll[1], (1,)).item()
        for _ in range(n):
            x = ca.step(x)
        loss = F.mse_loss(x[:, :1].clamp(0, 1), target)
        opt.zero_grad(); loss.backward()
        _grad_normalize(ca)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  [{mode}/seed] step {step:4d}  loss={loss.item():.4f}")
    return ca, target


def train_seed_repair(mode, steps=600, h=28, w=28, unroll=(36, 56), dmg_prob=0.5, shape="square"):
    """Diagnosis-driven self-heal: the pool trainer failed because persisted
    states accumulate into long horizons my model can't hold (degrade-loop,
    isolated via the no-damage diagnostic). This sidesteps the pool entirely:
    grow from a fresh seed, but with probability dmg_prob erase half the state
    at a random point PARTWAY through, and require the model to have recovered
    the target by the end. Teaches repair directly, with no state persistence
    across training steps -- so no long-horizon feedback loop to destabilize."""
    torch.manual_seed(0)
    ca = GrowCA(mode=mode)
    target = target_shape(shape, h, w)
    opt = torch.optim.Adam(ca.parameters(), lr=2e-3)
    for step in range(steps):
        x = seed_batch(h, w, ca.channels, 1)
        n = torch.randint(unroll[0], unroll[1], (1,)).item()
        dmg_at = torch.randint(n // 3, 2 * n // 3, (1,)).item() if torch.rand(1).item() < dmg_prob else -1
        for t in range(n):
            x = ca.step(x)
            if t == dmg_at:
                x = damage(x)                      # erase half mid-growth; must recover by step n
        loss = F.mse_loss(x[:, :1].clamp(0, 1), target)
        opt.zero_grad(); loss.backward(); _grad_normalize(ca); opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  [{mode}/seed-repair] step {step:4d}  loss={loss.item():.4f}")
    return ca, target


def train_pool(mode, steps=600, warmup=250, h=28, w=28, unroll=(28, 48),
               pool_size=32, batch=8, use_damage=True, shape="square"):
    """Warm-started persistent-pool training. The first attempt failed by
    throwing damage at an untrained model from step 0 -> unstable, never grew.
    Fix: PHASE 1 grows from the seed (no pool, no damage) until the model can
    actually form the square; PHASE 2 then seeds the pool from grown states
    and introduces damage, so repair is learned on top of a model that already
    grows rather than instead of it."""
    torch.manual_seed(0)
    ca = GrowCA(mode=mode)
    target = target_shape(shape, h, w)
    tgt_b = target.expand(batch, 1, h, w)
    opt = torch.optim.Adam(ca.parameters(), lr=2e-3)

    # PHASE 1: from-seed growth warmup
    for step in range(warmup):
        x = seed_batch(h, w, ca.channels, 1)
        n = torch.randint(unroll[0], unroll[1], (1,)).item()
        for _ in range(n):
            x = ca.step(x)
        loss = F.mse_loss(x[:, :1].clamp(0, 1), target)
        opt.zero_grad(); loss.backward(); _grad_normalize(ca); opt.step()
        if step % 50 == 0:
            print(f"  [{mode}/warmup] step {step:4d}  loss={loss.item():.4f}")

    # PHASE 2: seed the pool with GROWN states, then pool + damage for repair
    with torch.no_grad():
        pool = seed_batch(h, w, ca.channels, pool_size)
        for _ in range(40):
            pool = ca.step(pool)
    for step in range(warmup, steps):
        idx = torch.randperm(pool_size)[:batch]
        x = pool[idx].clone()
        with torch.no_grad():
            losses = ((x[:, :1].clamp(0, 1) - tgt_b) ** 2).mean(dim=(1, 2, 3))
        x[losses.argmax().item()] = seed_batch(h, w, ca.channels, 1)[0]   # keep from-seed growth fresh
        if use_damage and batch > 3:
            x[[1, 2]] = damage(x[[1, 2]])                                  # damage a couple for repair
        n = torch.randint(unroll[0], unroll[1], (1,)).item()
        for _ in range(n):
            x = ca.step(x)
        loss = F.mse_loss(x[:, :1].clamp(0, 1), tgt_b)
        opt.zero_grad(); loss.backward(); _grad_normalize(ca); opt.step()
        pool[idx] = x.detach()
        if step % 50 == 0 or step == steps - 1:
            print(f"  [{mode}/pool] step {step:4d}  loss={loss.item():.4f}")
    return ca, target


def demo(mode="ternary", trainer="seed", shape="square"):
    print(f"=== mode={mode}, trainer={trainer}, shape={shape} ===\n")
    trainers = {"seed": train_seed, "pool": train_pool, "seed-repair": train_seed_repair}
    ca, target = trainers[trainer](mode, shape=shape)
    h = w = 28

    x = seed_batch(h, w, ca.channels, 1)
    for _ in range(48):
        x = ca.step(x)
    print("\ngrown from seed:")
    print(ascii_grid(x))
    grown = F.mse_loss(x[:, :1].clamp(0, 1), target).item()
    print(f"  grown MSE: {grown:.4f}  (0.08 = did nothing)")

    x[:, :, :, : w // 2] = 0.0     # erase left half
    for _ in range(48):
        x = ca.step(x)
    healed = F.mse_loss(x[:, :1].clamp(0, 1), target).item()
    print("\nafter erasing left half + 48 steps (self-heal):")
    print(ascii_grid(x))
    print(f"  healed MSE: {healed:.4f}  (compare to grown; close = clean repair)")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "ternary"
    trainer = sys.argv[2] if len(sys.argv) > 2 else "seed"
    shape = sys.argv[3] if len(sys.argv) > 3 else "square"
    demo(mode, trainer, shape)
