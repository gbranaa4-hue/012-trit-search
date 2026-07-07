"""tritkit.layers -- drop-in ternary/binary Conv2d and Linear.

Each layer keeps a full-precision shadow weight (what the optimizer updates) and,
when do_quantize is on, runs the forward pass through the alpha-scaled quantizer.
Warmup = do_quantize off (train in float); QAT phase = do_quantize on.
`quantizable=False` pins a layer to float (used for the first conv / final layer).
"""
import torch.nn as nn
import torch.nn.functional as F

from .quant import quantize


class TernaryConv2d(nn.Conv2d):
    def __init__(self, *args, binary=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary = binary
        self.do_quantize = False
        self.quantizable = True

    @classmethod
    def from_conv(cls, c, binary=False):
        m = cls(c.in_channels, c.out_channels, c.kernel_size, stride=c.stride,
                padding=c.padding, dilation=c.dilation, groups=c.groups,
                bias=c.bias is not None, binary=binary)
        m.weight.data.copy_(c.weight.data)
        if c.bias is not None:
            m.bias.data.copy_(c.bias.data)
        return m

    def forward(self, x):
        w = quantize(self.weight, self.binary) if (self.do_quantize and self.quantizable) else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class TernaryLinear(nn.Linear):
    def __init__(self, *args, binary=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary = binary
        self.do_quantize = False
        self.quantizable = True

    @classmethod
    def from_linear(cls, lin, binary=False):
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None, binary=binary)
        m.weight.data.copy_(lin.weight.data)
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.data)
        return m

    def forward(self, x):
        w = quantize(self.weight, self.binary) if (self.do_quantize and self.quantizable) else self.weight
        return F.linear(x, w, self.bias)
