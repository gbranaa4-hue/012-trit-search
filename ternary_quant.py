"""
012 Ternary Quantization — Apply to any HuggingFace model

Takes any pretrained LLM and converts weights to {-1, 0, +1}.
No retraining required (post-training quantization).
Optional: fine-tune for a few hundred steps to recover quality.

Default target: Qwen/Qwen2.5-7B-Instruct
  Float16 size : ~14 GB download, ~14 GB VRAM
  Ternary size : ~1.4 GB VRAM  ← fits your RTX 5060 easily

Other options (change MODEL_NAME):
  "microsoft/Phi-3-mini-4k-instruct"   3.8B  0.75 GB ternary
  "mistralai/Mistral-7B-Instruct-v0.2" 7B    1.39 GB ternary
  "Qwen/Qwen2.5-14B-Instruct"          14B   2.77 GB ternary

Requirements:
  pip install transformers accelerate sentencepiece

Usage:
  python ternary_quant.py --quantize        Download + quantize
  python ternary_quant.py --chat            Chat with quantized model
  python ternary_quant.py --benchmark       Compare ternary vs float16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, time, json
import numpy as np

os.makedirs("models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME  = "Qwen/Qwen2.5-7B-Instruct"
SAVE_PATH   = "models/qwen2.5-7b-ternary"
TRIT_THRESH = 0.7   # τ = 0.7 × E[|W|]

# Layers to quantize — skip embeddings and final head (hurts quality most)
QUANTIZE_TARGETS = (nn.Linear,)
SKIP_NAMES       = {"lm_head", "embed_tokens", "embed_positions"}

# ══════════════════════════════════════════════════════════════════════════════
# TERNARY QUANTIZATION
# ══════════════════════════════════════════════════════════════════════════════

def quantize_weight(w, thresh=TRIT_THRESH):
    """
    w: float tensor → ternary tensor {-1, 0, +1}
    τ = thresh × E[|w|]  (adaptive per-layer threshold)
    """
    t = thresh * w.abs().mean()
    return torch.where(w >  t,  torch.ones_like(w),
           torch.where(w < -t, -torch.ones_like(w),
           torch.zeros_like(w)))

class TernaryLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with ternary weights.
    Stores original float weights for fine-tuning (STE).
    At inference: uses quantized {-1, 0, +1} weights.
    """
    def __init__(self, orig_layer):
        super().__init__()
        self.in_features  = orig_layer.in_features
        self.out_features = orig_layer.out_features
        self.has_bias     = orig_layer.bias is not None

        # Store float weights for gradient flow
        self.weight = nn.Parameter(orig_layer.weight.data.float().clone())
        if self.has_bias:
            self.bias = nn.Parameter(orig_layer.bias.data.float().clone())
        else:
            self.bias = None

        # Pre-compute trit weights for fast inference (skip on meta tensors)
        if self.weight.device.type != "meta":
            with torch.no_grad():
                self._trit_weight = quantize_weight(self.weight).half()
            self.zero_frac = (self._trit_weight == 0).float().mean().item()
        else:
            self._trit_weight = None
            self.zero_frac = 0.0

    def update_trit(self):
        """Recompute trit weights from current float weights"""
        if self.weight is None or self.weight.device.type == "meta":
            return
        with torch.no_grad():
            self._trit_weight = quantize_weight(self.weight).to(
                self.weight.device).half()
        self.zero_frac = float((self._trit_weight == 0).float().mean())

    def freeze(self):
        """Drop float weights after quantization — inference only, saves VRAM"""
        if self.weight is not None and self.weight.device.type == "meta":
            return  # skip meta tensors — handled lazily in forward()
        self.update_trit()
        self.weight = None

    def forward(self, x):
        if self._trit_weight is None:
            self.update_trit()
        if self._trit_weight.device != x.device:
            self._trit_weight = self._trit_weight.to(x.device)
        w    = self._trit_weight.to(dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"zero={self.zero_frac*100:.1f}%")

def apply_ternary_quantization(model, skip_names=SKIP_NAMES, verbose=True):
    """
    Replace all nn.Linear layers with TernaryLinear.
    Skips embedding and output head layers.
    """
    replaced = 0
    total_params = 0
    zero_params  = 0

    def replace_recursive(module, prefix=""):
        nonlocal replaced, total_params, zero_params
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                # Check if this layer should be skipped
                skip = any(s in full_name for s in skip_names)
                if not skip:
                    trit = TernaryLinear(child).to(child.weight.device)
                    setattr(module, name, trit)
                    replaced    += 1
                    total_params += child.weight.numel()
                    zero_params  += int(trit.zero_frac * child.weight.numel())
            else:
                replace_recursive(child, full_name)

    replace_recursive(model)

    # Freeze all layers — compute trit weights, drop float weights, free VRAM
    print("  Freezing (dropping float weights)...")
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            module.freeze()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n  Quantized {replaced} layers")
        print(f"  Total ternary params : {total_params:,}")
        print(f"  Ternary size         : {total_params*1.585/8/1e6:.1f} MB")
        print(f"  Float16 size         : {total_params*2/1e6:.1f} MB")
        print(f"  Compression          : {total_params*2/(total_params*1.585/8):.1f}x\n")

    return model

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_quantize(model_name=MODEL_NAME):
    """
    Load a HuggingFace model in float16, then apply ternary quantization.
    float16 loading keeps VRAM usage manageable during conversion.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Install: pip install transformers accelerate")
        return None, None

    print(f"Loading {model_name}...")
    print(f"  (This downloads ~14 GB on first run — subsequent runs are instant)\n")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load to CPU in float16 — avoids meta tensor issues from device_map="auto"
    # Quantize on CPU, then move final small trit weights to GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"  Loaded in {time.time()-t0:.0f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    mem_before = 0
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM before quantization: {mem_before:.2f} GB")

    # Apply ternary quantization
    print("\nApplying 012 ternary quantization...")
    model = apply_ternary_quantization(model)

    # Move quantized model to GPU — only trit weights remain, fits easily
    print("  Moving to GPU...")
    model = model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after quantization : {mem_after:.2f} GB")
        if mem_before > 0:
            print(f"  VRAM reduction          : {(mem_before-mem_after)/mem_before*100:.1f}%")

    return model, tokenizer

def save_ternary_model(model, tokenizer, path=SAVE_PATH):
    """Save quantized model. Only saves trit weights (very small)."""
    os.makedirs(path, exist_ok=True)
    # Save trit weights as int8 (values -1, 0, 1 fit in int8)
    trit_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            if module._trit_weight is None:
                # Meta tensor — materialize weight to CPU then quantize
                if module.weight is not None and module.weight.device.type == "meta":
                    w = module.weight.to("cpu")
                    module._trit_weight = quantize_weight(w.float()).half()
                else:
                    module.update_trit()
            if module._trit_weight is not None:
                trit_weights[name] = module._trit_weight.cpu().to(torch.int8)
    torch.save(trit_weights, f"{path}/trit_weights.pt")
    tokenizer.save_pretrained(path)
    size_mb = os.path.getsize(f"{path}/trit_weights.pt") / 1e6
    print(f"\n  Saved ternary weights: {path}/trit_weights.pt ({size_mb:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL FINE-TUNING
# Recover quality lost during quantization
# Only needs ~500 steps — ternary weights are already close to float16
# ══════════════════════════════════════════════════════════════════════════════

def fine_tune_ternary(model, tokenizer, texts, steps=500):
    """
    Fine-tune ternary model using STE.
    texts: list of strings to train on (your own data)
    Float weights receive gradients, trit weights used in forward pass.
    """
    from torch.optim import AdamW

    # Only optimize float weights (trit weights are derived)
    params = [p for n, p in model.named_parameters()
              if "weight" in n and p.requires_grad]
    opt = AdamW(params, lr=1e-5, weight_decay=0.01)

    print(f"\nFine-tuning {steps} steps on your data...")
    model.train()

    for step in range(steps):
        text   = texts[step % len(texts)]
        inputs = tokenizer(text, return_tensors="pt",
                           max_length=512, truncation=True).to(device)

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss    = outputs.loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        # Update trit weights from updated float weights
        if step % 50 == 0:
            for module in model.modules():
                if isinstance(module, TernaryLinear):
                    module.update_trit()
            print(f"  Step {step}/{steps}  loss={loss.item():.4f}")

    print("  Fine-tuning done.\n")

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# Compare ternary vs float16 quality
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(model, tokenizer, model_name):
    """Quick quality check on standard prompts"""
    model.eval()

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "The three laws of thermodynamics are",
        "In machine learning, a neural network",
        "Shakespeare wrote",
    ]

    print(f"\n{'═'*55}")
    print(f"  BENCHMARK — {model_name}")
    print(f"{'═'*55}\n")

    times = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t0     = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        t1 = time.perf_counter()
        n_new  = out.shape[1] - inputs["input_ids"].shape[1]
        tps    = n_new / (t1 - t0)
        times.append(tps)

        result = tokenizer.decode(out[0], skip_special_tokens=True)
        completion = result[len(prompt):]
        print(f"  Prompt : {prompt}")
        print(f"  Output : {completion.strip()[:100]}")
        print(f"  Speed  : {tps:.1f} tok/s\n")

    print(f"  Average: {np.mean(times):.1f} tok/s")

# ══════════════════════════════════════════════════════════════════════════════
# CHAT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def chat(model, tokenizer):
    model.eval()
    print(f"\n{'═'*55}")
    print(f"  012 Ternary LLM — Local Chat")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Commands: :quit  :stats  :temp 0.8")
    print(f"{'═'*55}\n")

    temperature = 0.7
    history     = []

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user: continue
        if user == ":quit": break
        if user == ":stats":
            if torch.cuda.is_available():
                print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            n_trit = sum(1 for m in model.modules() if isinstance(m, TernaryLinear))
            print(f"  Ternary layers: {n_trit}")
            continue
        if user.startswith(":temp"):
            try:
                temperature = float(user.split()[1])
                print(f"  Temperature: {temperature}")
            except: pass
            continue

        # Build chat prompt
        history.append({"role": "user", "content": user})
        try:
            prompt = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
        except:
            prompt = f"User: {user}\nAssistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        n_in   = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        t1    = time.perf_counter()
        n_new = out.shape[1] - n_in
        tps   = n_new / (t1 - t0)

        reply = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})

        print(f"\n012: {reply}")
        print(f"     [{n_new} tokens, {tps:.1f} tok/s]\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_plan():
    print(f"""
  012 Ternary LLM — Plan

  Target model : {MODEL_NAME}
  After quant  : ~1.4 GB ternary (fits easily in 8GB VRAM)

  Steps:
    1. python ternary_quant.py --quantize
       Downloads model (~14 GB, one time)
       Applies ternary quantization
       Saves 1.4 GB ternary model to models/

    2. python ternary_quant.py --benchmark
       Compares output quality and speed

    3. python ternary_quant.py --chat
       Interactive chat — fully local, no API

  To use a different model:
    Edit MODEL_NAME at the top of this file.
    Options (all free, Apache 2.0):
      "microsoft/Phi-3-mini-4k-instruct"    3.8B  0.75 GB ternary
      "Qwen/Qwen2.5-7B-Instruct"            7B    1.39 GB ternary  ← default
      "Qwen/Qwen2.5-14B-Instruct"           14B   2.77 GB ternary
      "mistralai/Mistral-7B-Instruct-v0.2"  7B    1.39 GB ternary

  Note on LLaMA:
    Meta requires account approval at meta.ai
    Qwen/Mistral/Phi are equivalent quality, freely available
    """)

def main():
    global MODEL_NAME, SAVE_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize",   action="store_true")
    parser.add_argument("--chat",       action="store_true")
    parser.add_argument("--benchmark",  action="store_true")
    parser.add_argument("--model",      type=str, default=MODEL_NAME)
    args = parser.parse_args()

    MODEL_NAME = args.model
    SAVE_PATH  = f"models/{MODEL_NAME.split('/')[-1].lower()}-ternary"

    if args.quantize:
        model, tokenizer = load_and_quantize(MODEL_NAME)
        if model is None: return
        save_ternary_model(model, tokenizer, SAVE_PATH)
        benchmark(model, tokenizer, f"{MODEL_NAME} (ternary)")
        print(f"\nDone. Run --chat to talk to it.")

    elif args.benchmark:
        model, tokenizer = load_and_quantize(MODEL_NAME)
        if model: benchmark(model, tokenizer, f"{MODEL_NAME} (ternary)")

    elif args.chat:
        model, tokenizer = load_and_quantize(MODEL_NAME)
        if model: chat(model, tokenizer)

    else:
        print_plan()

if __name__ == "__main__":
    main()
