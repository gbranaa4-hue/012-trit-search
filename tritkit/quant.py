"""tritkit.quant -- alpha-scaled ternary/binary quantizers with straight-through gradients.

The alpha scale is the piece naive ternary code leaves out (including this repo's
own earlier 7B artifact, which stored raw {-1,0,+1} with no scale and collapsed):
a ternary weight must be  alpha * {-1, 0, +1},  not raw {-1,0,+1}, or the weight
magnitudes are wrong by ~1/alpha and activations blow up. Here alpha is the TWN
analytic optimum -- the mean magnitude of the *kept* (non-zero) weights.

Forward: alpha * ternary(W). Backward: straight-through estimator (identity,
clipped at |W|<=1) so gradients reach the full-precision shadow weight.
"""
import torch

THRESH = 0.7  # Delta = THRESH * mean(|W|) per weight tensor (TWN / this-repo rule)


class _TernaryAlpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, thresh):
        delta = thresh * w.abs().mean()
        mask = w.abs() > delta
        codes = torch.sign(w) * mask                       # {-1, 0, +1}
        n = mask.sum().clamp(min=1)
        alpha = (w.abs() * mask).sum() / n                 # analytic TWN scale
        ctx.save_for_backward(w)
        return alpha * codes
    @staticmethod
    def backward(ctx, g):
        (w,) = ctx.saved_tensors
        return g * (w.abs() <= 1.0).to(g.dtype), None      # STE, clipped; None for `thresh`


class _BinaryAlpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        alpha = w.abs().mean()
        ctx.save_for_backward(w)
        signs = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        return alpha * signs
    @staticmethod
    def backward(ctx, g):
        (w,) = ctx.saved_tensors
        return g * (w.abs() <= 1.0).to(g.dtype)


def quantize(w, binary=False, thresh=THRESH):
    """alpha-scaled ternary (default) or binary quantization of a weight tensor."""
    return _BinaryAlpha.apply(w) if binary else _TernaryAlpha.apply(w, thresh)


def ternary_codes(w, thresh=THRESH):
    """For export/bit-packing: return (int8 codes in {-1,0,+1}, alpha float)."""
    delta = thresh * w.abs().mean()
    mask = w.abs() > delta
    codes = (torch.sign(w) * mask).to(torch.int8)
    n = mask.sum().clamp(min=1)
    alpha = ((w.abs() * mask).sum() / n).item()
    return codes, alpha
