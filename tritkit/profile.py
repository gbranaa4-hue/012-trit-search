"""tritkit.profile -- the honest numbers: params, size, FLOPs, latency.

Size counts quantized layers at their real bit cost (log2(3)=1.585 bits/weight for
ternary, 1 bit for binary; layers pinned to float count at 32). This is the
deployed size assuming bit-packing (see tritkit.export, v0.2) -- the profiler
reports the *target*, the exporter delivers it.
"""
import math
import time

import torch
import torch.nn as nn

from .layers import TernaryConv2d, TernaryLinear

LOG2_3 = math.log2(3)


def _bits_per_weight(m):
    if isinstance(m, (TernaryConv2d, TernaryLinear)) and m.quantizable:
        return 1.0 if m.binary else LOG2_3
    return 32.0


def size_kb(model):
    total = sum(p.numel() for p in model.parameters())
    bits = 0.0
    counted = 0
    for m in model.modules():
        if isinstance(m, (TernaryConv2d, TernaryLinear)):
            bits += m.weight.numel() * _bits_per_weight(m)
            counted += m.weight.numel()
            if m.bias is not None:
                bits += m.bias.numel() * 32
                counted += m.bias.numel()
    bits += (total - counted) * 32          # BatchNorm, PReLU, un-swapped params
    return total, bits / 8 / 1024


def flops(model, input_size=(1, 3, 32, 32), device="cpu"):
    macs = [0]

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            macs[0] += o.shape[1] * o.shape[2] * o.shape[3] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            macs[0] += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_size).to(device))
    for h in hs:
        h.remove()
    return macs[0]


def latency_ms(model, input_size=(1, 3, 32, 32), device="cpu", iters=50, warmup=10):
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def profile(model, input_size=(1, 3, 32, 32), device="cpu", show=True):
    """Return {params, size_kb, mmacs, latency_ms} and (optionally) print it."""
    params, kb = size_kb(model)
    mmacs = flops(model, input_size, device) / 1e6
    lat = latency_ms(model, input_size, device)
    out = dict(params=params, size_kb=kb, mmacs=mmacs, latency_ms=lat)
    if show:
        print(f"  params      : {params:,}")
        print(f"  size        : {kb:.1f} KB  (bit-packed target)")
        print(f"  compute     : {mmacs:.1f} MMACs/inference")
        print(f"  latency     : {lat:.2f} ms  ({1000/lat:.0f}/s) on {device}")
    return out
