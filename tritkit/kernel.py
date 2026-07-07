"""tritkit.kernel -- a multiply-free reference ternary inference kernel (numpy).

Runs a packed tritkit model end-to-end WITHOUT PyTorch and WITHOUT float
multiplies in the ternary layers: a ternary MAC is add / subtract / skip.
For codes in {-1,0,+1},  codes @ x  =  (codes==+1)@x  -  (codes==-1)@x,
and a 0/1-matrix times x is pure accumulation -- no multipliers. The only
multiply is one alpha-scale per output channel (negligible).

This is a REFERENCE kernel: it proves the compute model and counts the ops the
energy story rests on. A production kernel ports this to C/CUDA/MCU (XNOR-popcount
or LUT); it consumes the same packed format. numpy here = correctness + op audit.

    from tritkit.kernel import run_tiny_cnn
    logits, stats = run_tiny_cnn("model.tt", x_numpy)   # x: (N,3,H,W) float32
"""
import numpy as np
import torch

from .export import unpack_trits

EPS = 1e-5  # torch BatchNorm2d default


class _Ops:
    def __init__(self):
        self.mults_avoided = 0   # float multiplies a dense kernel would have done
        self.add_sub = 0         # real ternary ops (nonzero codes)
        self.skipped = 0         # zero codes skipped


def _weight(entry):
    """Return ('t'|'b'|'f', array, alpha) from a packed blob entry."""
    kind = entry[0]
    if kind == "f":
        return "f", entry[1].float().numpy(), None
    _, shape, alpha, packed = entry
    n = int(np.prod(shape))
    if kind == "t":
        codes = unpack_trits(packed, n).reshape(shape).astype(np.int8)
    else:  # binary
        bits = np.unpackbits(packed)[:n].astype(np.int8)
        codes = (bits * 2 - 1).reshape(shape)
    return kind, codes, alpha


def _trit_matmul(codes, cols, alpha, ops):
    """codes (O,K) in {-1,0,+1}, cols (M,K) float -> (M,O), multiply-free."""
    pos = (codes == 1).astype(np.float32)
    neg = (codes == -1).astype(np.float32)
    out = cols @ pos.T - cols @ neg.T          # accumulation only (0/1 matmul)
    M, K = cols.shape
    O = codes.shape[0]
    ops.mults_avoided += M * O * K
    nz = int((codes != 0).sum())
    ops.add_sub += M * nz
    ops.skipped += M * (O * K - nz)
    return alpha * out


def _im2col(x, kh, kw, stride, pad):
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    cols = np.empty((N, C, kh, kw, Ho, Wo), np.float32)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = xp[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * Ho * Wo, C * kh * kw), Ho, Wo


def _conv(x, wentry, bias, ops, stride=1, pad=1):
    kind, w, alpha = wentry
    O, C, kh, kw = w.shape
    cols, Ho, Wo = _im2col(x, kh, kw, stride, pad)
    if kind == "f":
        out = cols @ w.reshape(O, -1).T            # float conv (kept-float layer)
        ops.mults_avoided += 0
    else:
        out = _trit_matmul(w.reshape(O, -1), cols, alpha, ops)
    N = x.shape[0]
    out = out.reshape(N, Ho, Wo, O).transpose(0, 3, 1, 2)
    if bias is not None:
        out += bias[None, :, None, None]
    return out


def _bn(x, w, b, rm, rv):
    return (x - rm[None, :, None, None]) / np.sqrt(rv[None, :, None, None] + EPS) \
        * w[None, :, None, None] + b[None, :, None, None]


def _linear(x, wentry, bias, ops):
    kind, w, alpha = wentry           # w: (O, in)
    if kind == "f":
        out = x @ w.T
    else:
        out = _trit_matmul(w, x, alpha, ops)
    if bias is not None:
        out += bias[None, :]
    return out


def run_tiny_cnn(packed_path, x):
    """Reference forward for the tritkit bench model:
    3x [conv -> BN -> ReLU -> MaxPool2d(2)] -> GAP -> Linear.
    Returns (logits ndarray, op-stats dict)."""
    blob = torch.load(packed_path, weights_only=False)
    g = lambda k: blob[k]
    npf = lambda k: g(k)[1].float().numpy() if g(k)[0] == "f" else None
    ops = _Ops()
    h = np.asarray(x, np.float32)
    for i in (0, 1, 2):
        cb = npf(f"{i}.0.bias") if f"{i}.0.bias" in blob else None
        h = _conv(h, _weight(g(f"{i}.0.weight")), cb, ops, stride=1, pad=1)
        h = _bn(h, npf(f"{i}.1.weight"), npf(f"{i}.1.bias"),
                npf(f"{i}.1.running_mean"), npf(f"{i}.1.running_var"))
        h = np.maximum(h, 0.0)                                  # ReLU
        N, C, H, W = h.shape
        h = h.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))   # MaxPool2d(2)
    h = h.mean(axis=(2, 3))                                     # GAP -> (N, C)
    lb = npf("5.bias") if "5.bias" in blob else None
    logits = _linear(h, _weight(g("5.weight")), lb, ops)
    stats = dict(mults_avoided=ops.mults_avoided, add_sub=ops.add_sub,
                 skipped=ops.skipped,
                 skip_frac=ops.skipped / max(1, ops.add_sub + ops.skipped))
    return logits, stats
