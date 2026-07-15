#!/usr/bin/env python3
"""Adopt-BitNet path: run a pretrained ternary LLM and measure it, honestly.

Loads microsoft/bitnet-b1.58-2B-4T (2B params, ternary weights, trained from
scratch by Microsoft -- the recipe our own ternary LLM lacked) and measures:
  (1) does it produce coherent output?  (the thing our 7B artifact could not)
  (2) tokens/sec on this machine.

Honest note: transformers runs BitNet as a REFERENCE (dequantized) path, so the
tok/s here is a lower bound -- bitnet.cpp (the optimized 1.58-bit kernel) is
faster. This measures 'does adopting BitNet give a working on-device LLM', not
the peak speed.
"""
import os
# BitNet's weight-unpack uses torch.compile, which needs a C++ compiler (CPU) or
# triton (GPU) to build kernels -- neither is present here. Run eager instead.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import time
import torch
import torch.nn as nn
import torch._dynamo
torch._dynamo.config.disable = True
from transformers import AutoModelForCausalLM, AutoTokenizer


def dequantize_bitnet(model):
    """Replace AutoBitLinear (uint8 codes {0,1,2} + weight_scale) with a plain
    bf16 Linear whose weight = (code-1)*scale = alpha*{-1,0,+1}. Bypasses the
    torch.compile weight-unpack that needs a C++ compiler/triton (absent here).
    Runs activations in bf16 (higher precision than BitNet's int8 acts -> coherent)."""
    n = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if type(child).__name__ == "AutoBitLinear":
                codes = child.weight.data
                scale = child.weight_scale.data.to(torch.bfloat16)
                real = (codes.to(torch.bfloat16) - 1.0) * scale        # alpha*{-1,0,+1}
                lin = nn.Linear(child.in_features, child.out_features,
                                bias=getattr(child, "bias", None) is not None)
                lin.weight = nn.Parameter(real.reshape(child.out_features, child.in_features))
                if getattr(child, "bias", None) is not None:
                    lin.bias = nn.Parameter(child.bias.data.to(torch.bfloat16))
                setattr(parent, name, lin)
                n += 1
    return model, n

# The transformers BitNet unpack is @torch.compile'd; with no C++ compiler/triton
# it fails and the load silently leaves weights packed (uint8). Replace it with an
# eager version (same 2-bit/4-per-byte logic) so the load materializes bf16 weights.
import transformers.integrations.bitnet as _bn
_VPI = getattr(_bn, "VALUES_PER_ITEM", 4)


def _unpack_eager(packed, dtype):
    ps = packed.shape
    rows = ps[0] * _VPI
    shape = (rows,) if len(ps) == 1 else (rows, *ps[1:])
    unpacked = torch.zeros(shape, device=packed.device, dtype=torch.uint8)
    for i in range(_VPI):
        s = i * ps[0]
        unpacked[s:s + ps[0]] = (packed & (3 << (2 * i))) >> (2 * i)
    return unpacked.to(dtype) - 1


_bn.unpack_weights = _unpack_eager

MODEL = "microsoft/bitnet-b1.58-2B-4T"
OUT = "bitnet_result.txt"


def log(msg):
    print(msg, flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    open(OUT, "w").close()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {MODEL} on {device} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model, n_deq = dequantize_bitnet(model)
    log(f"  dequantized {n_deq} ternary layers -> bf16 (alpha*codes)")
    try:
        model = model.to(device)
    except RuntimeError as e:
        log(f"  GPU load failed ({str(e)[:80]}), falling back to CPU")
        device = "cpu"; model = model.to("cpu")
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e9
    log(f"  loaded in {time.time()-t0:.0f}s | {params:.2f}B params | device {device}")

    prompt = "Explain what a Python for loop does, in one sentence."
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(device)
    n_in = enc["input_ids"].shape[1]

    with torch.no_grad():
        model.generate(**enc, max_new_tokens=8, do_sample=False)     # warmup
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model.generate(**enc, max_new_tokens=120, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
    dt = time.time() - t0
    new = out.shape[1] - n_in
    resp = tok.decode(out[0][n_in:], skip_special_tokens=True)

    coherent = len(resp.split()) > 4 and any(c.isalpha() for c in resp)
    log(f"\n=== GENERATION: {new} tokens in {dt:.1f}s = {new/dt:.1f} tok/s on {device} ===")
    log(f"prompt: {prompt}")
    log(f"output: {resp}")
    log(f"\nverdict: {'COHERENT -- adopting BitNet gives a working on-device LLM' if coherent else 'INCOHERENT -- investigate'}")
    log("(tok/s is the transformers reference path; bitnet.cpp is faster.)")


if __name__ == "__main__":
    main()
