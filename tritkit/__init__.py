"""tritkit -- a focused, honest ternary-QAT toolkit for tiny edge models.

    import tritkit as tk

    model  = MyTinyCNN()                                    # any small CNN
    tmodel = tk.ternarize(model, keep_first_last=True)      # alpha-scaled ternary layers
    tk.qat_fit(tmodel, train_loader, epochs=20, warmup=4)   # QAT (float warmup -> quantized)
    tk.profile(tmodel)                                      # params / size / FLOPs / latency

Scope, stated up front (this is the whole point):
  * WORKS: coarse / redundant tasks on tiny CNNs, trained with QAT. Ternary ~= float.
  * FAILS (do not ship): fine identity/verification, and PTQ of any kind. Measured.
  * The alpha scale is built into every layer (raw {-1,0,+1} without it collapses).

v0.1: quant + layers + convert + qat + profile. v0.2 roadmap: export.py
(bit-packing + a reference ternary inference kernel, to make the size/energy win
real on hardware), threshold/temperature annealing for harder tasks, a bench/ suite.
"""
from .quant import quantize, ternary_codes, THRESH
from .layers import TernaryConv2d, TernaryLinear
from .convert import ternarize, set_quant
from .qat import qat_fit
from .profile import profile, size_kb, flops, latency_ms
from .export import save_packed, load_packed, pack_trits, unpack_trits
from .kernel import run_tiny_cnn

__version__ = "0.3.0"
__all__ = [
    "quantize", "ternary_codes", "THRESH",
    "TernaryConv2d", "TernaryLinear",
    "ternarize", "set_quant",
    "qat_fit",
    "profile", "size_kb", "flops", "latency_ms",
    "save_packed", "load_packed", "pack_trits", "unpack_trits",
    "run_tiny_cnn",
]
