"""
012 LoRA Merge + Quantize
Merges LoRA adapters into base Qwen model then quantizes to 4-bit GGUF.
Result: single ~4GB file, fully self-contained, no HuggingFace dependency.

Install:
  pip install transformers peft torch
  # For GGUF export (optional, smaller/faster):
  pip install llama-cpp-python

Usage:
  python trit_merge.py --merge          Merge LoRA into base weights
  python trit_merge.py --quantize       Quantize merged model to 4-bit
  python trit_merge.py --merge --quantize   Do both in one step
  python trit_merge.py --test           Test the merged model
"""

import os, argparse, json, time
from pathlib import Path
import torch

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_MODEL   = "Qwen/Qwen2.5-Coder-7B-Instruct"
LORA_PATH    = Path(__file__).parent / "lora" / "012-coder"
MERGED_PATH  = Path(__file__).parent / "models" / "012-coder-merged"
QUANT_PATH   = Path(__file__).parent / "models" / "012-coder-4bit"

# ══════════════════════════════════════════════════════════════════════════════
# MERGE
# Bakes LoRA adapter weights permanently into the base model
# Result: standard HuggingFace model, no PEFT dependency needed
# ══════════════════════════════════════════════════════════════════════════════

def merge():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        print("Install: pip install transformers peft")
        return

    if not LORA_PATH.exists():
        print(f"LoRA adapters not found at {LORA_PATH}")
        print("Run trit_lora.py --train first")
        return

    os.makedirs(MERGED_PATH, exist_ok=True)

    print(f"Loading base model: {BASE_MODEL}")
    print("(Loading in float16 for merge — ~14GB RAM needed)\n")

    # Load base in float16 on CPU for merging
    # Don't use 4-bit here — merging requires actual weights
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"Loading LoRA adapters from {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, str(LORA_PATH))

    print("Merging adapters into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_PATH}...")
    model.save_pretrained(str(MERGED_PATH), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_PATH))

    # Save model card
    card = f"""---
base_model: {BASE_MODEL}
tags:
  - 012-ternary
  - lora-merged
  - gdscript
  - code
---

# 012 Coder — Merged Model

Fine-tuned from {BASE_MODEL} using LoRA on:
- GDScript (Godot 4)
- Python, JavaScript, TypeScript, C#, Rust, Go
- 20+ programming languages

Merged: LoRA adapters baked into base weights.
No PEFT dependency required for inference.
"""
    (MERGED_PATH / "README.md").write_text(card)

    size_gb = sum(f.stat().st_size for f in MERGED_PATH.rglob("*") if f.is_file()) / 1e9
    print(f"\nMerge complete.")
    print(f"  Size   : {size_gb:.1f} GB")
    print(f"  Path   : {MERGED_PATH}")
    print(f"\nNext: python trit_merge.py --quantize")


# ══════════════════════════════════════════════════════════════════════════════
# QUANTIZE
# Converts float16 merged model → 4-bit
# Two methods:
#   1. bitsandbytes 4-bit (stays in HuggingFace format, easy)
#   2. GGUF Q4_K_M (llama.cpp format, most portable, works on CPU)
# ══════════════════════════════════════════════════════════════════════════════

def quantize(method="bnb"):
    os.makedirs(QUANT_PATH, exist_ok=True)

    if method == "bnb":
        _quantize_bnb()
    elif method == "gguf":
        _quantize_gguf()
    else:
        print(f"Unknown method: {method}. Use 'bnb' or 'gguf'")


def _quantize_bnb():
    """bitsandbytes 4-bit — stays in HuggingFace format."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("Install: pip install transformers bitsandbytes")
        return

    src = str(MERGED_PATH) if MERGED_PATH.exists() else BASE_MODEL
    print(f"Quantizing {src} → 4-bit bitsandbytes...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model in 4-bit (this IS the quantized version)...")
    model = AutoModelForCausalLM.from_pretrained(
        src,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

    print(f"Saving 4-bit model to {QUANT_PATH}...")
    model.save_pretrained(str(QUANT_PATH), safe_serialization=True)
    tokenizer.save_pretrained(str(QUANT_PATH))

    size_gb = sum(f.stat().st_size for f in QUANT_PATH.rglob("*") if f.is_file()) / 1e9
    print(f"\n4-bit quantization complete.")
    print(f"  Size   : {size_gb:.1f} GB  (was ~14GB float16)")
    print(f"  Path   : {QUANT_PATH}")
    print(f"\nTest: python trit_merge.py --test")


def _quantize_gguf():
    """
    GGUF Q4_K_M — most portable format.
    Works with llama.cpp, Ollama, LM Studio, Jan.
    Can run on CPU with no GPU.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Install: pip install transformers")
        return

    src = str(MERGED_PATH) if MERGED_PATH.exists() else BASE_MODEL

    # Step 1: convert to GGUF using llama.cpp convert script
    import subprocess, sys

    llama_cpp = Path.home() / "llama.cpp"
    convert_script = llama_cpp / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print("llama.cpp not found. Cloning...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp",
                        str(llama_cpp)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        str(llama_cpp / "requirements.txt")], check=True)

    gguf_fp16 = QUANT_PATH / "model-fp16.gguf"
    gguf_q4   = QUANT_PATH / "model-q4_k_m.gguf"
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"

    print(f"Converting {src} → GGUF float16...")
    subprocess.run([
        sys.executable, str(convert_script),
        src, "--outfile", str(gguf_fp16), "--outtype", "f16"
    ], check=True)

    if quantize_bin.exists():
        print("Quantizing GGUF float16 → Q4_K_M...")
        subprocess.run([str(quantize_bin), str(gguf_fp16), str(gguf_q4), "Q4_K_M"],
                       check=True)
        gguf_fp16.unlink()  # remove intermediate
        size_gb = gguf_q4.stat().st_size / 1e9
        print(f"\nGGUF Q4_K_M complete.")
        print(f"  Size   : {size_gb:.1f} GB")
        print(f"  Path   : {gguf_q4}")
        print(f"\nTo run with Ollama:")
        print(f"  ollama create 012-coder -f Modelfile")
    else:
        size_gb = gguf_fp16.stat().st_size / 1e9
        print(f"\nGGUF float16 saved (build llama.cpp to quantize further).")
        print(f"  Size   : {size_gb:.1f} GB")
        print(f"  Path   : {gguf_fp16}")


# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA IMPORT
# Makes your merged model available via Ollama
# ══════════════════════════════════════════════════════════════════════════════

def create_ollama_modelfile():
    """Create Ollama Modelfile for the merged model."""
    gguf_path = QUANT_PATH / "model-q4_k_m.gguf"
    if not gguf_path.exists():
        # Try bnb path
        gguf_path = QUANT_PATH

    modelfile = f"""FROM {gguf_path}

SYSTEM \"\"\"You are 012, a concise and honest coding assistant specialized in:
- GDScript and Godot 4 game development
- Python, JavaScript, TypeScript, C#, Rust, Go
- Ternary neural architecture research
- Game systems: health, combat, spawning, AI, UI

Answer directly. Show code. No fluff.\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""
    modelfile_path = Path(__file__).parent / "Modelfile"
    modelfile_path.write_text(modelfile)
    print(f"Modelfile written to {modelfile_path}")
    print("\nTo add to Ollama:")
    print(f"  ollama create 012-coder -f {modelfile_path}")
    print("  ollama run 012-coder")


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

def test():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("Install: pip install transformers")
        return

    # Use quantized if available, else merged, else base
    if QUANT_PATH.exists():
        src = str(QUANT_PATH)
        use_4bit = False  # already quantized
        print(f"Testing quantized model: {src}")
    elif MERGED_PATH.exists():
        src = str(MERGED_PATH)
        use_4bit = True
        print(f"Testing merged model (loading 4-bit): {src}")
    else:
        print("No merged model found. Run --merge first.")
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        src,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model.eval()

    test_prompts = [
        "Write a GDScript HealthComponent with take_damage and heal functions.",
        "Write a Python function to load a JSON file safely.",
        "Write a wave spawner in GDScript that spawns enemies every 30 seconds.",
    ]

    print("\n" + "="*60)
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}\n")
        messages = [
            {"role": "system", "content": "You are 012, a concise coding assistant. Show code directly."},
            {"role": "user",   "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)
        print(response)
        print("-"*60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA + quantize Qwen")
    parser.add_argument("--merge",     action="store_true", help="Merge LoRA into base")
    parser.add_argument("--quantize",  action="store_true", help="Quantize to 4-bit")
    parser.add_argument("--gguf",      action="store_true", help="Use GGUF format instead of bnb")
    parser.add_argument("--test",      action="store_true", help="Test the model")
    parser.add_argument("--ollama",    action="store_true", help="Create Ollama Modelfile")
    args = parser.parse_args()

    if args.merge:
        merge()

    if args.quantize:
        method = "gguf" if args.gguf else "bnb"
        quantize(method)

    if args.ollama:
        create_ollama_modelfile()

    if args.test:
        test()

    if not any([args.merge, args.quantize, args.test, args.ollama]):
        parser.print_help()
        print("\nTypical workflow after trit_lora.py --train completes:")
        print("  python trit_merge.py --merge --quantize")
        print("  python trit_merge.py --test")
        print("  python trit_merge.py --ollama   (optional, add to Ollama)")
