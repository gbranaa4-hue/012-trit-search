#!/usr/bin/env python3
"""#3 step 1 -- YOUR tritkit ternary, PTQ'd onto a real 0.5B LLM (Qwen2.5-0.5B).

No retraining: swap every attention/MLP Linear for tritkit's alpha-scaled
TernaryLinear and generate. This is the instrument check before the QAT+distill
build -- honest prior: PTQ degrades a lot (it always has this session). Measure it.
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tritkit as tk

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "qwen_trit_ptq_out.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(m):
    with open(OUT, "a", encoding="utf-8") as f:   # file first (utf-8) so it's always captured
        f.write(m + "\n")
    try:
        print(m, flush=True)
    except UnicodeEncodeError:
        print(m.encode("ascii", "replace").decode(), flush=True)


@torch.no_grad()
def gen(model, tok, prompt, n=60):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(DEV)
    out = model.generate(**enc, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    open(OUT, "w").close()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    log(f"loaded {MODEL}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, {n_lin} Linear layers")

    prompt = "In one sentence, what is a neural network?"
    log("\n--- FLOAT baseline ---")
    log(gen(model, tok, prompt))

    # YOUR ternary, applied as PTQ (no retraining)
    tk.ternarize(model, keep_first_last=True)
    tk.set_quant(model, True)
    model = model.to(DEV)
    # zero fraction of the ternarized weights
    zt = zc = 0
    for m in model.modules():
        if isinstance(m, tk.TernaryLinear) and m.quantizable:
            q = tk.quantize(m.weight, binary=False)
            zt += q.numel(); zc += (q == 0).sum().item()
    log(f"\n--- tritkit TERNARY-PTQ (your method, {100*zc/zt:.0f}% weights zeroed) ---")
    log(gen(model, tok, prompt))

    log("\n[read] if float is coherent and ternary is not -> PTQ insufficient (expected),")
    log("       QAT + distillation from the float teacher is step 2 (the real 'yours').")


if __name__ == "__main__":
    main()
