"""
012 LoRA Fine-tuning Pipeline
Fine-tune Qwen2.5-coder:7b on your data using LoRA.

LoRA = Low Rank Adaptation
  Only trains ~1% of weights (small adapter matrices)
  Base model stays frozen — you can't break it
  Adapter is ~50 MB — easy to save, share, swap
  VRAM needed: ~4-5 GB (vs 14 GB for full fine-tune)

Your RTX 5060 (8GB) handles this comfortably.

What you get:
  Qwen2.5-coder:7b quality (7/10 code)
  Fine-tuned on YOUR data (knows your project)
  Runs locally forever
  Zero API cost

Install:
  pip install transformers peft accelerate bitsandbytes tqdm

Usage:
  python trit_lora.py --prepare          Collect your training data
  python trit_lora.py --train            Fine-tune with LoRA
  python trit_lora.py --chat             Chat with fine-tuned model
  python trit_lora.py --merge            Merge adapter into base model
"""

import torch
import torch.nn as nn
import argparse, os, json, time
from pathlib import Path

os.makedirs("lora",    exist_ok=True)
os.makedirs("data",    exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM   : {vram:.1f} GB")
    print(f"GPU    : {torch.cuda.get_device_name(0)}\n")

BASE_MODEL   = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "lora/qwen_012_adapter"
MERGED_PATH  = "lora/qwen_012_merged"
DATA_PATH    = "data/lora_train.jsonl"

# LoRA config
LORA_R       = 16     # rank — higher = more capacity, more VRAM
LORA_ALPHA   = 32     # scaling factor (usually 2x rank)
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]  # which layers to adapt

# Training config
MAX_LEN      = 512    # reduced from 1024 — halves activation memory
BATCH_SIZE   = 1      # minimum batch size
GRAD_ACCUM   = 16     # effective batch = 1 × 16 = 16 (same as before)
LR           = 2e-4
EPOCHS       = 2
WARMUP_STEPS = 50

# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# Collects your files + generates Q&A pairs via Ollama
# Format: {"instruction": "...", "input": "...", "output": "..."}
# ══════════════════════════════════════════════════════════════════════════════

# All relevant languages in 2026
CODE_LANGUAGES_2026 = [
    # Systems / performance
    "Python", "Rust", "C", "C++", "Go", "Zig",
    # Web
    "JavaScript", "TypeScript", "HTML", "CSS",
    # Game dev
    "C#", "Lua",
    # Data / ML
    "SQL", "R",
    # Mobile
    "Swift", "Kotlin",
    # Scripting / shell
    "Shell", "PowerShell",
    # Other popular
    "Java", "PHP", "Ruby", "Scala", "Dart",
]

# Edit this list (or set TRIT_SCAN_DIRS env var, os.pathsep-separated) to
# point at your own codebase to fine-tune on.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
SCAN_DIRS = os.environ.get("TRIT_SCAN_DIRS", "").split(os.pathsep) if os.environ.get("TRIT_SCAN_DIRS") else [
    _PROJECT_ROOT,
]
SCAN_EXTS = {
    ".py", ".gd", ".js", ".ts", ".cs", ".md", ".txt",
    ".rs", ".go", ".c", ".cpp", ".h", ".hpp", ".java",
    ".lua", ".rb", ".php", ".swift", ".kt", ".r", ".sql",
    ".sh", ".ps1", ".dart", ".zig", ".toml", ".yaml", ".json",
}

# How many examples to stream per language from GitHub
STREAM_PER_LANG = 2000   # ~2000 code files per language

def stream_github_code(language, max_examples=STREAM_PER_LANG):
    """Stream real code from HuggingFace — tries multiple datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    examples = []

    # Dataset options in priority order
    DATASET_OPTIONS = [
        # TheStack v2 — no auth needed, large, maintained
        ("bigcode/the-stack-smol", {"split": "train",
                                     "data_dir": f"data/{language}"}),
        # Fallback: StarCoder training data subset
        ("bigcode/starcoderdata",  {"split": "train",
                                     "data_dir": language.lower()}),
        # Fallback: code_search_net for Python/Java/JS/PHP/Ruby/Go
        ("code_search_net",        {"split": "train",
                                     "trust_remote_code": True}),
    ]

    ds = None
    for dataset_name, kwargs in DATASET_OPTIONS:
        try:
            ds = load_dataset(dataset_name, streaming=True,
                              trust_remote_code=True, **kwargs)
            break
        except Exception:
            continue

    if ds is None:
        print(f"\r    {language}: no dataset available — skipping      ")
        return []

    try:
        for i, sample in enumerate(ds):
            if i >= max_examples: break
            code  = sample.get("content", sample.get("code",
                    sample.get("whole_func_string", "")))
            fname = sample.get("path", sample.get("func_name", "file"))
            if not code or len(code) < 100 or len(code) > 4000: continue

            examples.append({
                "instruction": f"Complete this {language} code:",
                "input":       code[:len(code)//2],
                "output":      code[len(code)//2:len(code)//2+1000]
            })

            if len(code) > 200:
                examples.append({
                    "instruction": f"Explain this {language} code:",
                    "input":       code[:1500],
                    "output":      f"This {language} code defines {fname} which: {code[:100]}"
                })

            if i % 200 == 0:
                print(f"\r    {language}: {len(examples)} examples",
                      end="", flush=True)

    except Exception as e:
        print(f"\n    {language} error: {e}")

    print(f"\r    {language}: {len(examples)} examples      ")
    return examples

def collect_local_data():
    """Scan local files and convert to instruction format."""
    examples = []

    for d in SCAN_DIRS:
        if not os.path.exists(d): continue
        for root, _, files in os.walk(d):
            if any(p.startswith('.') for p in Path(root).parts): continue
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in SCAN_EXTS: continue
                fpath = os.path.join(root, fname)
                try:
                    text = open(fpath, "r", encoding="utf-8", errors="ignore").read()
                    if len(text) < 50 or len(text) > 8000: continue
                except:
                    continue

                rel = os.path.relpath(fpath, SCAN_DIRS[0] if SCAN_DIRS else _PROJECT_ROOT)

                # Format 1: explain this file
                examples.append({
                    "instruction": f"Explain what this file does: {rel}",
                    "input":       text[:2000],
                    "output":      f"This is {rel}. " + text[:500]
                })

                # Format 2: complete this code (use first half as prompt)
                if ext in {".py", ".gd", ".js", ".cs", ".rs"}:
                    mid = len(text) // 2
                    examples.append({
                        "instruction": "Complete this code:",
                        "input":       text[:mid],
                        "output":      text[mid:mid+1000]
                    })

    print(f"  Local files: {len(examples)} examples")
    return examples

def generate_ollama_examples():
    """Generate Q&A pairs from Ollama covering your specific topics."""
    import urllib.request, urllib.error

    PROMPTS = [
        # Response style examples — direct, honest, code-first
        ("What is a signal in Godot and when should I use it?",
         "A signal is Godot's event system. Use it when one node needs to notify others without knowing who they are.\n\n```gdscript\nsignal enemy_died(enemy_id)\n\nfunc die():\n    emit_signal(\"enemy_died\", id)\n```\n\nConnect in the parent:\n```gdscript\nenemy.enemy_died.connect(_on_enemy_died)\n```\n\nUse signals for: UI updates, game events, decoupling systems. Don't use them for: direct function calls between tightly coupled nodes."),
        ("My GDScript code is slow, what should I check first?",
         "In order:\n1. Move heavy work out of _process — use _physics_process or timers\n2. Profile with Godot's built-in profiler (Debugger > Profiler)\n3. Avoid get_node() calls in loops — cache references in _ready()\n4. Use object pools for frequently spawned objects\n5. Check if you're doing string concatenation in loops\n\nMost horde game slowdowns are from too many individual draw calls or uncached node lookups."),
        ("Explain the difference between @export and a regular variable in GDScript",
         "@export makes a variable visible and editable in the Godot Inspector.\n\n```gdscript\n@export var speed: float = 200.0  # editable per-instance in editor\nvar internal_timer: float = 0.0   # code-only, not in inspector\n```\n\nUse @export for anything a designer or you might want to tweak per-scene. Keep implementation details as regular vars."),
        # Godot / GDScript
        ("Write a GDScript state machine for an enemy with patrol, chase, attack states",
         None),
        ("Write a GDScript signal system for a horde defense game wave manager",
         None),
        ("Write a Godot CharacterBody2D movement script with acceleration and friction",
         None),
        ("Write a GDScript object pool for bullets in a horde game",
         None),
        ("Explain Godot's scene instancing with a code example",
         None),
        ("Write a GDScript resource for a card in a deck-building game",
         None),
        ("How do you implement a creep spawner in Godot?",
         None),
        # 012 Architecture
        ("Explain ternary neural networks with code examples in Python",
         None),
        ("Write a PyTorch ternary linear layer with straight-through estimator",
         None),
        ("What is the difference between ternary and binary neural networks?",
         None),
        ("Write a simple Hopfield network in Python",
         None),
        ("Explain continual learning and catastrophic forgetting",
         None),
        ("What is knowledge distillation in machine learning?",
         None),
        ("Write a simple transformer attention mechanism in PyTorch",
         None),
        ("Explain LoRA fine-tuning with a code example",
         None),
        # General coding
        ("Write a Python class for a replay buffer used in continual learning",
         None),
        ("Write an efficient Python LRU cache implementation",
         None),
        ("Explain and implement A* pathfinding in Python",
         None),
        ("Write a Python implementation of a binary search tree",
         None),
        ("Write a simple neural network training loop in PyTorch",
         None),
        ("How do you implement a streaming data loader in Python?",
         None),
        ("Write a Python script to chunk a large text file for processing",
         None),
        ("Implement a simple tokenizer in Python",
         None),
    ]

    examples = []
    url      = "http://localhost:11434/api/generate"

    # Check Ollama
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models",[])]
    except:
        print("  Ollama not running — skipping generated examples")
        return examples

    # Find best available model
    model = None
    for preferred in ["qwen2.5-coder:7b", "deepseek-r1:7b", "llama3.2:latest"]:
        if any(preferred in m for m in models):
            model = preferred
            break
    if not model:
        print(f"  No suitable model found in Ollama")
        return examples

    print(f"  Generating {len(PROMPTS)} examples via {model}...")

    for i, (prompt, _) in enumerate(PROMPTS):
        payload = json.dumps({
            "model": model, "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 600, "temperature": 0.3}
        }).encode()

        try:
            req  = urllib.request.Request(url, data=payload,
                                          headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                resp = json.loads(r.read()).get("response","")
            examples.append({
                "instruction": prompt,
                "input":       "",
                "output":      resp
            })
            print(f"\r  [{i+1}/{len(PROMPTS)}] {prompt[:50]}...", end="", flush=True)
        except Exception as e:
            print(f"\n  Failed: {e}")

    print(f"\n  Generated: {len(examples)} examples")
    return examples

def prepare_data():
    """Collect all training data and save as JSONL."""
    print("Preparing training data...\n")

    all_examples = []

    # 1. Your local files — highest priority, goes in first so it's seen often
    print("Scanning local files...")
    local = collect_local_data()
    # Repeat local files 3x so model weights them more heavily
    all_examples += local * 3
    print(f"  Local total (×3): {len(local)*3} examples\n")

    # 2. GitHub code — all 2026 relevant languages
    print("Streaming GitHub code (this takes a while)...")
    for lang in CODE_LANGUAGES_2026:
        print(f"  {lang}...")
        examples = stream_github_code(lang, max_examples=STREAM_PER_LANG)
        all_examples += examples

    # 3. Ollama generated Q&A — targeted to your work
    print("\nGenerating targeted Q&A via Ollama...")
    all_examples += generate_ollama_examples()

    if not all_examples:
        print("No training data found.")
        return 0

    # Shuffle
    import random
    random.shuffle(all_examples)

    # Save as JSONL
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    size_kb = os.path.getsize(DATA_PATH) / 1024
    print(f"\n  Saved {len(all_examples)} examples → {DATA_PATH} ({size_kb:.1f} KB)")
    return len(all_examples)

# ══════════════════════════════════════════════════════════════════════════════
# LORA TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def format_prompt(example, tokenizer):
    """Format instruction/input/output into chat template."""
    instruction = example.get("instruction","")
    inp         = example.get("input","")
    output      = example.get("output","")

    if inp:
        user_msg = f"{instruction}\n\n{inp}"
    else:
        user_msg = instruction

    messages = [
        {"role": "system",    "content": "You are a helpful coding assistant with expertise in Godot, GDScript, Python, and AI research."},
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": output},
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    except:
        text = f"User: {user_msg}\nAssistant: {output}"

    return text

class LoRADataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_len=MAX_LEN):
        self.examples  = []
        self.tokenizer = tokenizer
        self.max_len   = max_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line.strip())
                text = format_prompt(ex, tokenizer)
                tok  = tokenizer(text, truncation=True, max_length=max_len,
                                 return_tensors="pt")
                ids  = tok["input_ids"][0]
                if len(ids) > 10:
                    self.examples.append(ids)

        print(f"  Dataset: {len(self.examples)} examples loaded")

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return {"input_ids": ids, "labels": ids.clone()}

def collate_fn(batch, pad_id=0):
    ids    = [b["input_ids"] for b in batch]
    labels = [b["labels"]    for b in batch]
    max_l  = max(x.shape[0] for x in ids)

    ids_pad    = torch.stack([
        torch.cat([x, torch.full((max_l - x.shape[0],), pad_id)]) for x in ids])
    labels_pad = torch.stack([
        torch.cat([x, torch.full((max_l - x.shape[0],), -100)]) for x in labels])

    return {"input_ids": ids_pad, "labels": labels_pad}

def train_lora():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("Install: pip install transformers peft accelerate bitsandbytes")
        return

    if not os.path.exists(DATA_PATH):
        print(f"No data found. Run: python trit_lora.py --prepare")
        return

    print(f"Loading {BASE_MODEL}...")
    print(f"Using 4-bit base + LoRA adapters — needs ~4-5 GB VRAM\n")

    # Load in 4-bit to save VRAM (base model frozen, only LoRA trains)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,}  ({trainable/total*100:.2f}% of total)")
    print(f"  Adapter size     : ~{trainable*4/1e6:.1f} MB\n")

    # Dataset
    dataset  = LoRADataset(DATA_PATH, tokenizer)
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn
    )

    opt      = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )
    total_steps = len(loader) * EPOCHS // GRAD_ACCUM
    sch         = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    print(f"Training {EPOCHS} epochs  ({total_steps} steps)...\n")

    global_step = 0
    best_loss   = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        opt.zero_grad()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            out  = model(input_ids=input_ids, labels=labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step+1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
                global_step += 1

                # Progress bar
                pct    = (step+1) / len(loader)
                bar_w  = 25
                filled = int(bar_w * pct)
                bar    = "█" * filled + "░" * (bar_w - filled)
                avg    = epoch_loss / (step+1)

                print(f"\r  Epoch {epoch+1}/{EPOCHS} [{bar}] "
                      f"step={global_step}  loss={avg:.3f}  "
                      f"lr={sch.get_last_lr()[0]:.2e}",
                      end="", flush=True)

        avg_loss = epoch_loss / len(loader)
        print(f"\r  Epoch {epoch+1}/{EPOCHS} [{'█'*bar_w}] "
              f"loss={avg_loss:.3f}  lr={sch.get_last_lr()[0]:.2e}   ")

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(ADAPTER_PATH)
            tokenizer.save_pretrained(ADAPTER_PATH)
            print(f"  ✓ Adapter saved → {ADAPTER_PATH}\n")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Run --chat to talk to it, --merge to create standalone model\n")

# ══════════════════════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are 012, a concise and honest coding assistant.
You were built using the 012 ternary triadic architecture and fine-tuned on the user's own codebase.
Your personality:
- Direct. Give the answer first, explanation after.
- Honest. Say when you don't know or when something won't work.
- Concise. No filler. No unnecessary hedging.
- Code-first. Show working code, not pseudocode.
- Aware of the user's project: Godot 4 horde defense game, 012 ternary research, Python ML code.
When shown code, understand it in the context of the user's project."""

def load_chat_model():
    """Load base + adapter. Returns (model, tokenizer) or (None, None)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError:
        print("Install: pip install transformers peft accelerate bitsandbytes")
        return None, None

    if not os.path.exists(ADAPTER_PATH):
        print(f"No adapter found. Run: python trit_lora.py --train")
        return None, None

    print("Loading model + adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    base      = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True)
    model     = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    vram = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    print(f"  VRAM: {vram:.2f} GB\n")
    return model, tokenizer

def do_retrain(model, tokenizer):
    """Re-prepare data and re-train adapter. Called from chat via :retrain."""
    print("\n  Retraining on current project files...\n")
    # Unload model to free VRAM
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Re-prepare local files only (fast — skip GitHub streaming)
    examples = collect_local_data() * 3
    examples += generate_ollama_examples()
    import random; random.shuffle(examples)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Prepared {len(examples)} examples")
    train_lora()
    # Reload
    return load_chat_model()

def generate_reply(model, tokenizer, history, temperature=0.7):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except:
        prompt = SYSTEM_PROMPT + "\n" + "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history
        ) + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_in   = inputs["input_ids"].shape[1]
    t0     = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    t1    = time.perf_counter()
    n_new = out.shape[1] - n_in
    reply = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
    return reply, n_new, t1 - t0

def chat():
    model, tokenizer = load_chat_model()
    if model is None: return

    print("012 Assistant — Qwen2.5-Coder 7B + your codebase")
    print("Commands:")
    print("  :retrain     Re-scan your project files and retrain adapter")
    print("  :clear       Clear conversation history")
    print("  :temp 0.7    Set temperature")
    print("  :paste       Paste multiline code (end with :::)")
    print("  :quit        Exit\n")

    history     = []
    temperature = 0.7

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not q: continue

        if q == ":quit":
            break

        if q == ":clear":
            history = []
            print("  Cleared.\n")
            continue

        if q.startswith(":temp"):
            try:
                temperature = float(q.split()[1])
                print(f"  Temperature: {temperature}")
            except: pass
            continue

        if q == ":paste":
            print("  Paste code, end with ::: on its own line:")
            lines = []
            while True:
                line = input()
                if line.strip() == ":::": break
                lines.append(line)
            q = "\n".join(lines)
            print(f"  ({len(q)} chars received)\n")

        if q == ":retrain":
            print("  Retraining on your current project files...")
            print("  This will take 1-2 hours. Continue? (y/n)")
            if input("  > ").strip().lower() == "y":
                model, tokenizer = do_retrain(model, tokenizer)
                if model is None: break
                history = []
                print("  Retrain complete. Ready.\n")
            else:
                print("  Cancelled.\n")
            continue

        history.append({"role": "user", "content": q})

        print("012: ", end="", flush=True)
        reply, n_new, elapsed = generate_reply(model, tokenizer, history, temperature)
        tps = n_new / elapsed if elapsed > 0 else 0

        print(reply)
        print(f"     [{n_new} tok  {tps:.1f} tok/s]\n")

        history.append({"role": "assistant", "content": reply})

        # Trim history to last 10 turns to avoid context overflow
        if len(history) > 20:
            history = history[-20:]
        n_new = out.shape[1] - n_in
        tps   = n_new / (t1 - t0)

        reply = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})

        print(f"\n012: {reply}")
        print(f"     [{n_new} tok  {tps:.1f} tok/s]\n")

# ══════════════════════════════════════════════════════════════════════════════
# MERGE — optional, creates standalone model without needing base
# ══════════════════════════════════════════════════════════════════════════════

def merge():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        print("Install: pip install transformers peft accelerate")
        return

    if not os.path.exists(ADAPTER_PATH):
        print("No adapter found.")
        return

    print("Merging LoRA adapter into base model...")
    print("(Needs ~14 GB RAM — loads full float16 model)\n")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    base      = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True)
    model     = PeftModel.from_pretrained(base, ADAPTER_PATH)
    merged    = model.merge_and_unload()

    merged.save_pretrained(MERGED_PATH)
    tokenizer.save_pretrained(MERGED_PATH)
    size = sum(f.stat().st_size for f in Path(MERGED_PATH).rglob("*")) / 1e9
    print(f"  Merged model saved: {MERGED_PATH}  ({size:.1f} GB)")
    print(f"  Run with: ollama create 012-assistant -f Modelfile")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Collect training data")
    parser.add_argument("--train",   action="store_true", help="Fine-tune with LoRA")
    parser.add_argument("--chat",    action="store_true", help="Chat with fine-tuned model")
    parser.add_argument("--merge",   action="store_true", help="Merge adapter into base")
    parser.add_argument("--all",     action="store_true", help="Prepare + train")
    args = parser.parse_args()

    if args.prepare or args.all:
        n = prepare_data()
        if n == 0: return

    if args.train or args.all:
        train_lora()

    if args.chat:
        chat()

    if args.merge:
        merge()

    if not any([args.prepare, args.train, args.chat, args.merge, args.all]):
        print(__doc__)
        print(f"  Data exists    : {os.path.exists(DATA_PATH)}")
        print(f"  Adapter exists : {os.path.exists(ADAPTER_PATH)}")
        print(f"  Merged exists  : {os.path.exists(MERGED_PATH)}")

if __name__ == "__main__":
    main()
