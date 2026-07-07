"""tritkit.export -- bit-pack a ternarized model to a compact deployable file.

This is what turns the profiler's size *target* into a real artifact on disk:
  * ternary weights -> 5 trits per byte (base-3, 3^5=243<256) = 1.6 bits/weight
    (near the log2(3)=1.585 optimum), plus one float32 alpha per layer;
  * binary weights  -> 8 per byte (1 bit) + alpha;
  * kept-float layers / BatchNorm / bias -> float16.

    tk.save_packed(model, "m.tt")        # writes the packed file, returns bytes
    tk.load_packed(fresh_model, "m.tt")  # reconstructs a runnable model

HONEST SCOPE: this delivers the real DISK size and a functional load-and-run
(the loaded model reproduces the trained model's predictions). It does NOT yet
deliver the runtime SPEED -- the reconstructed weights run as alpha*codes through
normal matmuls. A native ternary kernel (XNOR/popcount or LUT, hardware-specific)
consumes this packed format and is the v0.3 step. Packing first, kernel next.
"""
import os

import numpy as np
import torch

from .layers import TernaryConv2d, TernaryLinear
from .quant import ternary_codes


def pack_trits(codes):
    """codes: int array in {-1,0,1} -> uint8 array, 5 trits/byte (base-3)."""
    flat = (np.asarray(codes).reshape(-1) + 1).astype(np.uint8)   # -> {0,1,2}
    pad = (-len(flat)) % 5
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.uint8)])
    g = flat.reshape(-1, 5)
    return (g[:, 0] + 3 * g[:, 1] + 9 * g[:, 2] + 27 * g[:, 3] + 81 * g[:, 4]).astype(np.uint8)


def unpack_trits(packed, n):
    """uint8 array -> first n trits in {-1,0,1}."""
    out = np.zeros((len(packed), 5), np.uint8)
    v = packed.astype(np.uint16).copy()
    for i in range(5):
        out[:, i] = v % 3
        v //= 3
    return out.reshape(-1)[:n].astype(np.int8) - 1


def _quant_weight_modules(model):
    return {name + ".weight": m for name, m in model.named_modules()
            if isinstance(m, (TernaryConv2d, TernaryLinear)) and m.quantizable}


def save_packed(model, path):
    """Write model to a bit-packed file. Returns the file size in bytes."""
    qmods = _quant_weight_modules(model)
    blob = {}
    for name, tensor in model.state_dict().items():
        m = qmods.get(name)
        if m is None:
            t = tensor.detach().cpu()
            blob[name] = ("f", t.half() if t.is_floating_point() else t.clone())
            continue
        w = tensor.detach().cpu()
        if m.binary:
            alpha = w.abs().mean().item()
            bits = (w.reshape(-1) >= 0).to(torch.uint8).numpy()      # {-1,+1} -> {0,1}
            blob[name] = ("b", tuple(w.shape), alpha, np.packbits(bits))
        else:
            codes, alpha = ternary_codes(w)
            blob[name] = ("t", tuple(w.shape), alpha, pack_trits(codes.numpy()))
    torch.save(blob, path)
    return os.path.getsize(path)


def load_packed(model, path):
    """Reconstruct weights into `model` (same architecture, already ternarized).
    Sets do_quantize off -- the loaded weights ARE the effective alpha*codes."""
    blob = torch.load(path, weights_only=False)
    sd = {}
    for name, entry in blob.items():
        kind = entry[0]
        if kind == "t":
            _, shape, alpha, packed = entry
            n = int(np.prod(shape))
            codes = unpack_trits(packed, n).reshape(shape).astype(np.float32)
            sd[name] = torch.from_numpy(alpha * codes)
        elif kind == "b":
            _, shape, alpha, packed = entry
            n = int(np.prod(shape))
            bits = np.unpackbits(packed)[:n].astype(np.float32)      # {0,1}
            sd[name] = torch.from_numpy(alpha * (bits * 2 - 1)).reshape(shape)
        else:
            sd[name] = entry[1].float() if entry[1].is_floating_point() else entry[1]
    model.load_state_dict(sd)
    for m in model.modules():
        if isinstance(m, (TernaryConv2d, TernaryLinear)):
            m.do_quantize = False      # loaded weight already = alpha*codes
    return model
