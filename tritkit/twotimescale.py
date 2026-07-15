"""Two-timescale ternary: topology (slow gate) x polarity (fast sign), factorized.

W_eff = alpha * G * sign(u)
  G       in {0,1}  -- TOPOLOGY (ternary's zero as a connectivity gate), re-selected
                       on the slow clock from structural EVIDENCE (see below).
  sign(u) in {-1,+1} -- POLARITY, trained every step through the STE.
  alpha    -- per-layer scale = mean |u| over the live set (tritkit convention).

Two lessons this file encodes (both measured, not assumed):
 * STRUCTURAL MEMORY (two_timescale_ternary_v2.py): a gated-off weight receives no
   sign gradient -- its shadow u_i is frozen while dormant and reused on reconnection.
 * SEPARATE SENSING CHANNEL (v1 confound + this file's first smoke-test failure):
   gating by |u| is self-reinforcing -- only live weights get gradient, so a wrong
   initial gate locks in forever. The gate instead integrates an EMA of the FULL
   UNGATED gradient magnitude |dL/dW| (cheap: one outer product already computed in
   backward), which every weight sees whether live or dormant.

DATASHEET (network_tongues_step1b.py / phase_datasheet_step2.py):
 1. NOISE: average gradients until the kick K*sigma/sqrt(B) is below the capture
    half-width; single-sample updates destroy locking at moderate K.
 2. ENTRAINMENT: keep independent clocks at incommensurate (golden) ratios.
 3. RE-ENTRY: if the demand ROTATES with period P, make dormancy a multiple of P.
"""
import math
import torch
import torch.nn as nn

PHI = (1 + 5 ** 0.5) / 2


class _GatedSignLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u, G, alpha, evidence, beta):
        s = torch.where(u >= 0, 1.0, -1.0)
        w = alpha * G * s
        ctx.save_for_backward(x, G, w, evidence)
        ctx.alpha, ctx.beta = alpha, beta
        return x @ w.t()

    @staticmethod
    def backward(ctx, gout):
        x, G, w, ev = ctx.saved_tensors
        gw = gout.t() @ x                                  # FULL ungated weight grad
        with torch.no_grad():                              # structural sensing channel:
            ev.mul_(ctx.beta).add_(gw.abs(), alpha=1 - ctx.beta)
        gu = ctx.alpha * G * gw                            # STE, gated -> dormant u frozen
        gx = gout @ w
        return gx, gu, None, None, None, None


class TwoTimescaleLinear(nn.Module):
    """Linear layer with factorized ternary weight alpha * G * sign(u).

    Fast clock: ordinary optimizer steps train sign(u) on live entries (STE).
    Slow clock: call `step_gate()` every gate_period steps -- reconnects the
    top-evidence fraction; dormant signs stay frozen (structural memory).
    """

    def __init__(self, in_features, out_features, density=0.5, bias=True,
                 evidence_beta=0.99):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.density, self.evidence_beta = density, evidence_beta
        self.u = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        self.register_buffer("G", torch.ones(out_features, in_features))
        self.register_buffer("evidence", torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.step_gate()

    @torch.no_grad()
    def step_gate(self):
        """SLOW update: connect the top-evidence fraction (falls back to |u| before
        any evidence has accumulated)."""
        score = self.evidence if self.evidence.abs().sum() > 0 else self.u.abs()
        k = max(1, int(self.density * score.numel()))
        thresh = score.flatten().kthvalue(score.numel() - k + 1).values
        self.G.copy_((score >= thresh).float())

    def forward(self, x):
        with torch.no_grad():
            live = self.G > 0
            alpha = self.u.abs()[live].mean() if live.any() else self.u.abs().mean()
        out = _GatedSignLinear.apply(x, self.u, self.G, alpha,
                                     self.evidence, self.evidence_beta)
        return out + self.bias if self.bias is not None else out

    def sign_transfer(self, u_before):
        """Fraction of currently-live signs unchanged since the snapshot -- the
        memory-survival metric from the v2 experiment."""
        live = self.G > 0
        return ((self.u[live] >= 0) == (u_before[live] >= 0)).float().mean().item()


def golden_period(fast_tau: int) -> int:
    """Gate period incommensurate with the fast timescale (anti-entrainment)."""
    p = max(2, round(PHI * PHI * fast_tau))
    while math.gcd(p, max(fast_tau, 1)) > 1:
        p += 1
    return p


def commensurate_period(demand_period: float, min_steps: int) -> int:
    """Dormancy snapped to a multiple of a ROTATING demand's period (re-entry rule)."""
    k = max(1, math.ceil(min_steps / demand_period))
    return int(round(k * demand_period))


if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)
    N, K_SUP, R, T_REG, STEPS = 24, 8, 3, 300, 9000
    s_true = torch.randint(0, 2, (N,)).float() * 2 - 1
    sups = [torch.randperm(N)[:K_SUP] for _ in range(R)]

    lyr = TwoTimescaleLinear(N, 1, density=K_SUP / N, bias=False, evidence_beta=0.95)
    opt = torch.optim.SGD(lyr.parameters(), lr=0.02)
    B, T_GATE = 16, 50            # gate clock: slower than sign convergence (~20-30
    losses, transfers = [], []    # steps), FASTER than the regime -- never in sync
    u_snap = lyr.u.detach().clone()
    for t in range(STEPS):
        A = sups[(t // T_REG) % R]
        x = torch.randn(B, N)
        mask = torch.zeros(N); mask[A] = 1
        y = (x * (s_true * mask)).sum(1, keepdim=True)
        loss = F.mse_loss(lyr(x * mask), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if t % T_GATE == T_GATE - 1:
            lyr.step_gate()
        if t % T_REG == T_REG - 1:                    # measure memory per regime
            transfers.append(lyr.sign_transfer(u_snap))
            u_snap = lyr.u.detach().clone()
        if t > STEPS - 1000:
            losses.append(loss.item() / max(y.var().item(), 1e-8))
    nmse = sum(losses) / len(losses)
    tr = sum(transfers[-6:]) / 6
    print(f"smoke: tail NMSE {nmse:.3f} (want <0.35), sign transfer {tr:.2f} (want >0.85)")
    assert nmse < 0.35 and tr > 0.85, "SMOKE TEST FAILED"
    print("smoke test PASSED -- evidence-gated topology + frozen-sign memory works")
