#!/usr/bin/env python3
"""#3 step 2 -- recover a COHERENT ternary Qwen-0.5B with QAT + distillation.

The plan:
  * STUDENT = Qwen2.5-0.5B, tritkit-ternarized (your alpha-scaled TernaryLinear),
    quantization ON from the start; the full-precision shadow weights are what
    we train, quantized in every forward (QAT).
  * TEACHER = the frozen float Qwen. It hands the student soft targets so the
    ternary student learns to reproduce the float model's behaviour despite the
    {-1,0,+1} weights. This is the piece PTQ lacked -- and why PTQ collapsed.
  * DATA = general text (wikitext) so it recovers general coherence, not just style.

Memory-tuned for 8 GB: teacher frozen bf16 (no grad/opt), student with gradient
checkpointing + Adafactor (low-memory optimizer), small batch/seq.

Honest expectation: weight-only ternary (no A8/SubLN) + a few-hour fine-tune
recovers PARTIAL coherence -- from garbage toward readable, not full float
quality. We MEASURE how far it gets. Set MAX_STEPS small first (feasibility),
then large for the real run.
"""
import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import tritkit as tk

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "150"))   # feasibility default; raise for the real run
BATCH = int(os.environ.get("BATCH", "1"))             # fp32 student -> batch 1 to fit 8GB
SEQ, T = 256, 2.0
LR = float(os.environ.get("LR", "2e-4"))
OUT = "qwen_qat_distill_out.txt"


def log(m):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    try:
        print(m, flush=True)
    except UnicodeEncodeError:
        print(m.encode("ascii", "replace").decode(), flush=True)


def vram():
    if DEV == "cuda":
        return f"{torch.cuda.memory_allocated()/1e9:.2f}GB alloc / {torch.cuda.max_memory_allocated()/1e9:.2f}GB peak"
    return "cpu"


@torch.no_grad()
def sample(model, tok, prompt="In one sentence, what is a neural network?", n=40):
    tk.set_quant(model, True); model.eval()
    uc = model.config.use_cache; model.config.use_cache = True
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(DEV)
    out = model.generate(**enc, max_new_tokens=n, do_sample=False)
    model.config.use_cache = uc; model.train()
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def make_batches(tok, n_blocks):
    # INSTRUCT distillation: teach the ternary student Qwen's *assistant* behaviour
    # (chat-formatted instruction -> response), not encyclopedic wikitext style.
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train[:15000]")
    parts = []
    for ex in ds:
        instr = ex["instruction"] + (("\n" + ex["input"]) if ex.get("input") else "")
        parts.append(tok.apply_chat_template(
            [{"role": "user", "content": instr},
             {"role": "assistant", "content": ex["output"]}], tokenize=False))
    ids = tok("\n".join(parts), return_tensors="pt").input_ids[0]
    blocks = ids[: (len(ids) // SEQ) * SEQ].view(-1, SEQ)
    return blocks[:n_blocks] if n_blocks else blocks


def main():
    open(OUT, "w").close()
    log(f"MAX_STEPS={MAX_STEPS}, batch={BATCH}, seq={SEQ}, T={T}, lr={LR}")
    tok = AutoTokenizer.from_pretrained(MODEL)

    log("loading teacher (frozen float) + student (ternary QAT)...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    tk.ternarize(student, keep_first_last=True)
    student.float()                        # fp32 MASTER weights -> QAT can accumulate tiny updates
    student.to(DEV)
    tk.set_quant(student, True)
    student.gradient_checkpointing_enable()
    student.config.use_cache = False
    log(f"  loaded | VRAM {vram()}")

    from transformers.optimization import Adafactor
    opt = Adafactor([p for p in student.parameters() if p.requires_grad],
                    lr=LR, scale_parameter=False, relative_step=False, warmup_init=False)

    blocks = make_batches(tok, n_blocks=None).to("cpu")
    log(f"  corpus: {blocks.shape[0]} blocks of {SEQ} tokens")
    log(f"\n[before] ternary sample: {sample(student, tok)!r}")

    log("\ntraining (distill ternary student <- float teacher)...")
    t0 = time.time()
    step = 0
    while step < MAX_STEPS:
        perm = torch.randperm(blocks.shape[0])
        for bi in range(0, blocks.shape[0] - BATCH, BATCH):
            if step >= MAX_STEPS:
                break
            x = blocks[perm[bi:bi + BATCH]].to(DEV)
            with torch.no_grad():
                t_logits = teacher(x).logits
            s_logits = student(x).logits                       # fp32
            # distillation KL on soft targets (the recovery signal)
            loss = F.kl_div(F.log_softmax(s_logits / T, -1),
                            F.softmax(t_logits.float() / T, -1),
                            reduction="batchmean") * (T * T)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0:
                log(f"  step {step:>4}/{MAX_STEPS}  distill_loss {loss.item():.3f}  | VRAM {vram()}")
            if step > 0 and step % 500 == 0:                 # watch it climb + checkpoint
                log(f"  [sample @{step}] {sample(student, tok)!r}")
                os.makedirs("results", exist_ok=True)
                tk.save_packed(student, "results/qwen_ternary_qat.tt")
            step += 1
    log(f"trained {step} steps in {time.time()-t0:.0f}s")

    log(f"\n[after] ternary sample: {sample(student, tok)!r}")
    os.makedirs("results", exist_ok=True)
    tk.save_packed(student, "results/qwen_ternary_qat.tt")
    log("saved -> results/qwen_ternary_qat.tt")


if __name__ == "__main__":
    main()
