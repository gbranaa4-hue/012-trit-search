#!/usr/bin/env python3
"""BitNet b1.58 2B via the REAL transformers path (no hacks) -- now that MSVC
lets torch.compile unpack the ternary weights properly. Measures coherent
output + tokens/sec."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/bitnet-b1.58-2B-4T"
OUT = "bitnet_result.txt"


def log(m):
    print(m, flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(m + "\n")


def main():
    open(OUT, "w").close()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {MODEL} (real path, MSVC-compiled unpack) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    log(f"  loaded/converted in {time.time()-t0:.0f}s | "
        f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    try:
        model = model.to(dev)
    except Exception as e:
        log(f"  GPU move failed ({str(e)[:70]}); CPU"); dev = "cpu"; model = model.to("cpu")
    model.eval()

    prompt = "Explain what a Python for loop does, in one sentence."
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(dev)
    n_in = enc["input_ids"].shape[1]

    def gen(nt, e):
        with torch.no_grad():
            return model.generate(**e, max_new_tokens=nt, do_sample=False)

    try:
        gen(8, enc)                                  # warmup (triggers compile)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time(); out = gen(120, enc)
        if dev == "cuda":
            torch.cuda.synchronize()
    except Exception as e:
        log(f"  {dev} generate failed ({str(e)[:90]}); retrying on CPU")
        dev = "cpu"; model = model.to("cpu"); enc = enc.to("cpu")
        gen(8, enc); t0 = time.time(); out = gen(120, enc)

    dt = time.time() - t0
    new = out.shape[1] - n_in
    resp = tok.decode(out[0][n_in:], skip_special_tokens=True)
    coherent = len(set(resp.split())) > 8      # not a repeated single token
    log(f"\n=== GENERATION: {new} tokens in {dt:.1f}s = {new/dt:.1f} tok/s on {dev} ===")
    log(f"prompt: {prompt}")
    log(f"output: {resp}")
    log(f"\nverdict: {'COHERENT (distinct vocabulary)' if coherent else 'STILL DEGENERATE'}")


if __name__ == "__main__":
    main()
