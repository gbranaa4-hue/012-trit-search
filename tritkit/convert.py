"""tritkit.convert -- swap a model's Conv2d/Linear for ternary/binary versions.

    tmodel = tritkit.ternarize(my_cnn, keep_first_last=True)

keep_first_last pins the first conv and the last linear/conv to float -- standard
low-bit practice, since those two layers are the most quantization-sensitive.
"""
import torch.nn as nn

from .layers import TernaryConv2d, TernaryLinear


def ternarize(model, keep_first_last=True, binary=False):
    """In-place swap of eligible layers. Returns the same model for chaining."""
    q = []

    def swap(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, TernaryConv2d) or isinstance(child, TernaryLinear):
                q.append(child)
            elif isinstance(child, nn.Conv2d):
                new = TernaryConv2d.from_conv(child, binary=binary)
                setattr(parent, name, new); q.append(new)
            elif isinstance(child, nn.Linear):
                new = TernaryLinear.from_linear(child, binary=binary)
                setattr(parent, name, new); q.append(new)
            else:
                swap(child)

    swap(model)
    if keep_first_last and len(q) >= 2:
        q[0].quantizable = False
        q[-1].quantizable = False
    model._trit_layers = q
    return model


def set_quant(model, active):
    """Turn quantized forward on/off (off during warmup, on during the QAT phase)."""
    for m in model.modules():
        if isinstance(m, (TernaryConv2d, TernaryLinear)):
            m.do_quantize = active and m.quantizable
