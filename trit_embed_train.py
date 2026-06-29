"""
012 Ternary Embedding Fine-Tuner
Fine-tunes all-MiniLM-L6-v2 on GitHub code so trit_search understands
code semantics across all languages including GDScript.

Streams from HuggingFace — never downloads the full dataset.
Generates training pairs automatically from code structure.

Install:
  pip install sentence-transformers datasets torch

Usage:
  python trit_embed_train.py --train        Train from GitHub code
  python trit_embed_train.py --train --local Also include your local .gd files
  python trit_embed_train.py --test         Test the trained model
  python trit_embed_train.py --install      Copy model into trit_search
"""

import os, re, argparse, random, json, time
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME   = "all-MiniLM-L6-v2"
SAVE_PATH    = Path(__file__).parent / "models" / "code-minilm"
STATUS_PATH  = Path(__file__).parent / "models" / "embed_status.json"

# Languages to stream — includes GDScript via gdscript filter
CODE_LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "C", "C++", "C#",
    "Go", "Rust", "Java", "Kotlin", "Swift", "Ruby", "PHP",
    "Lua", "R", "Dart", "Zig", "Shell", "PowerShell",
    "GDScript",   # Godot — rare in public datasets but exists
]

PAIRS_PER_LANG  = 50_000   # pairs to extract per language
BATCH_SIZE      = 32       # training batch size
EPOCHS          = 3
WARMUP_STEPS    = 100
LR              = 2e-5

LOCAL_DIRS = [
    r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1",
    r"C:\Users\gbran\OneDrive\Documents\tribe",
    r"C:\Users\gbran\OneDrive\Documents\012-ternary",
]

# ══════════════════════════════════════════════════════════════════════════════
# PAIR EXTRACTION
# Takes raw code → generates (anchor, positive) pairs for contrastive training
# anchor: natural language description / function name
# positive: the actual code that matches
# ══════════════════════════════════════════════════════════════════════════════

def extract_pairs_from_code(code: str, lang: str) -> list:
    """
    Extract (anchor, positive) training pairs from a code snippet.
    Works across all languages by pattern matching.
    """
    pairs = []
    lines = code.splitlines()

    # 1. Function name → function body
    func_patterns = [
        r'(?:def|func|fn|function|fun|sub|proc)\s+(\w+)\s*\(',   # Python/GDScript/Rust/JS/Kotlin
        r'(?:public|private|protected|static)?\s*\w+\s+(\w+)\s*\(',  # Java/C#/C++
    ]
    for pat in func_patterns:
        for i, line in enumerate(lines):
            m = re.search(pat, line)
            if m:
                fname = m.group(1)
                if len(fname) < 3 or fname in ('if', 'for', 'while', 'return'):
                    continue
                # grab up to 15 lines of body
                body_lines = lines[i:i+15]
                body = "\n".join(body_lines).strip()
                if len(body) > 50:
                    # Convert snake_case/camelCase to words
                    readable = re.sub(r'([A-Z])', r' \1', fname)
                    readable = readable.replace('_', ' ').lower().strip()
                    pairs.append((readable, body))
                    # Also add raw name as anchor
                    pairs.append((fname, body))

    # 2. Comment → code below it
    comment_patterns = [
        r'^\s*#\s*(.+)$',       # Python / GDScript / Shell
        r'^\s*//\s*(.+)$',      # JS / C / C++ / Rust / Go
        r'^\s*--\s*(.+)$',      # Lua / SQL
    ]
    for i, line in enumerate(lines):
        for pat in comment_patterns:
            m = re.match(pat, line)
            if m:
                comment = m.group(1).strip()
                if len(comment) < 10 or len(comment) > 200:
                    continue
                # get next non-empty code block
                rest = "\n".join(lines[i+1:i+10]).strip()
                if len(rest) > 30:
                    pairs.append((comment, rest))
                break

    # 3. Class name → class body
    class_patterns = [
        r'class(?:_name)?\s+(\w+)',        # GDScript / Python
        r'(?:class|struct|interface)\s+(\w+)',  # C#/Java/C++/Rust
    ]
    for i, line in enumerate(lines):
        for pat in class_patterns:
            m = re.search(pat, line)
            if m:
                cname = m.group(1)
                body = "\n".join(lines[i:i+20]).strip()
                if len(body) > 60:
                    readable = re.sub(r'([A-Z])', r' \1', cname).lower().strip()
                    pairs.append((f"{readable} class", body))
                break

    # 4. Signal / event declarations (GDScript specific)
    for line in lines:
        m = re.match(r'^\s*signal\s+(\w+)', line)
        if m:
            sname = m.group(1).replace('_', ' ')
            pairs.append((f"signal {sname}", line.strip()))

    # Deduplicate and filter garbage
    seen   = set()
    result = []
    for anchor, positive in pairs:
        key = anchor[:50]
        if key in seen:
            continue
        seen.add(key)
        # Skip if anchor is too short or too long
        if len(anchor) < 4 or len(anchor) > 300:
            continue
        if len(positive) < 20:
            continue
        result.append((anchor, positive))

    return result


def extract_local_pairs() -> list:
    """Extract pairs from your local project files."""
    pairs = []
    ext_map = {'.gd': 'GDScript', '.py': 'Python', '.js': 'JavaScript',
               '.ts': 'TypeScript', '.cs': 'C#', '.rs': 'Rust'}

    for base in LOCAL_DIRS:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in
                       {'.git', '__pycache__', 'addons', 'node_modules',
                        '.venv', 'models', 'search_index', 'lora'}]
            for f in files:
                ext = Path(f).suffix
                if ext not in ext_map:
                    continue
                path = os.path.join(root, f)
                try:
                    code = open(path, encoding='utf-8', errors='replace').read()
                    lang = ext_map[ext]
                    new_pairs = extract_pairs_from_code(code, lang)
                    pairs.extend(new_pairs)
                except Exception:
                    pass

    print(f"  Local files: {len(pairs)} pairs extracted")
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# GITHUB STREAMING
# ══════════════════════════════════════════════════════════════════════════════

def stream_github_pairs(lang: str, target: int) -> list:
    """Stream code from GitHub, extract pairs until target reached."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return []

    pairs   = []
    scanned = 0

    # Dataset options in priority order
    datasets_to_try = [
        ("bigcode/the-stack-smol", {"data_files": f"data/{lang.lower()}/*",
                                     "split": "train", "streaming": True}),
        ("codeparrot/github-code",  {"streaming": True, "split": "train",
                                     "languages": [lang]}),
    ]

    for ds_name, ds_kwargs in datasets_to_try:
        try:
            ds = load_dataset(ds_name, trust_remote_code=True, **ds_kwargs)
            for item in ds:
                scanned += 1
                code = item.get("content", item.get("code", ""))
                if not code or len(code) < 200:
                    continue
                new_pairs = extract_pairs_from_code(code, lang)
                pairs.extend(new_pairs)

                if len(pairs) >= target:
                    break
                if scanned % 500 == 0:
                    print(f"\r    {lang}: {len(pairs)}/{target} pairs  "
                          f"({scanned} files scanned)", end="", flush=True)

            if pairs:
                break   # got enough from this dataset
        except Exception as e:
            continue    # try next dataset

    print(f"\r    {lang}: {len(pairs)} pairs  ({scanned} files scanned)     ")
    return pairs[:target]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(use_local=True):
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
        from torch.utils.data import DataLoader
    except ImportError:
        print("Install: pip install sentence-transformers")
        return

    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(SAVE_PATH.parent, exist_ok=True)

    # Load status to allow resume
    status = {}
    if STATUS_PATH.exists():
        status = json.loads(STATUS_PATH.read_text())

    all_pairs = []

    # --- Local files first (highest priority — your actual code) ---
    if use_local:
        print("\n[1/2] Extracting pairs from your local projects...")
        local = extract_local_pairs()
        # Repeat local pairs 5x — they're gold, weight them heavily
        all_pairs.extend(local * 5)
        print(f"  Local pairs (×5): {len(local)*5}")

    # --- GitHub streaming ---
    print("\n[2/2] Streaming from GitHub...")
    for lang in CODE_LANGUAGES:
        if status.get(f"lang_{lang}"):
            print(f"  {lang}: already done (resume)")
            cached = json.loads(
                (SAVE_PATH / f"pairs_{lang}.json").read_text()
            ) if (SAVE_PATH / f"pairs_{lang}.json").exists() else []
            all_pairs.extend(cached)
            continue

        print(f"  Streaming {lang}...")
        pairs = stream_github_pairs(lang, PAIRS_PER_LANG)

        # Cache to disk so we can resume
        (SAVE_PATH / f"pairs_{lang}.json").write_text(
            json.dumps(pairs[:2000])  # save sample for resume
        )
        status[f"lang_{lang}"] = True
        STATUS_PATH.write_text(json.dumps(status))

        all_pairs.extend(pairs)

    print(f"\nTotal pairs collected: {len(all_pairs):,}")

    if len(all_pairs) < 100:
        print("Not enough pairs. Check dataset access.")
        return

    # Shuffle
    random.shuffle(all_pairs)

    # Convert to InputExample format
    examples = [InputExample(texts=[a, p]) for a, p in all_pairs]

    # Split: 95% train, 5% eval
    split    = int(len(examples) * 0.95)
    train_ex = examples[:split]
    eval_ex  = examples[split:]

    print(f"  Train: {len(train_ex):,}  |  Eval: {len(eval_ex):,}")

    # Load base model
    print(f"\nLoading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # DataLoader
    loader = DataLoader(train_ex, shuffle=True, batch_size=BATCH_SIZE)

    # MultipleNegativesRankingLoss — best for (anchor, positive) pairs
    # For each anchor in a batch, all other positives become negatives
    # This is what sentence-transformers uses for production embedding training
    loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluator on held-out pairs
    eval_anchors   = [e.texts[0] for e in eval_ex[:200]]
    eval_positives = [e.texts[1] for e in eval_ex[:200]]
    # Give dummy scores of 1.0 (all pairs are positive)
    eval_scores    = [1.0] * len(eval_anchors)
    evaluator = EmbeddingSimilarityEvaluator(
        eval_anchors, eval_positives, eval_scores,
        name="code-eval"
    )

    total_steps = len(loader) * EPOCHS

    print(f"Training {EPOCHS} epochs  ({total_steps:,} steps)...")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print(f"  Save path   : {SAVE_PATH}\n")

    model.fit(
        train_objectives=[(loader, loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        optimizer_params={"lr": LR},
        output_path=str(SAVE_PATH),
        show_progress_bar=True,
        evaluation_steps=500,
        save_best_model=True,
    )

    print(f"\nModel saved to {SAVE_PATH}")
    print("Run: python trit_embed_train.py --install")


# ══════════════════════════════════════════════════════════════════════════════
# INSTALL INTO trit_search
# ══════════════════════════════════════════════════════════════════════════════

def install():
    """Patch trit_search.py to use the fine-tuned model."""
    search_path = Path(__file__).parent / "trit_search.py"
    if not search_path.exists():
        print("trit_search.py not found")
        return
    if not SAVE_PATH.exists():
        print(f"Model not found at {SAVE_PATH} — run --train first")
        return

    code = search_path.read_text(encoding='utf-8')
    old  = '"all-MiniLM-L6-v2"'
    new  = f'r"{SAVE_PATH}"'

    if new in code:
        print("Already using fine-tuned model.")
        return

    code = code.replace(old, new)
    search_path.write_text(code, encoding='utf-8')
    print(f"trit_search.py updated to use fine-tuned model.")
    print("Now rebuild the index:")
    print("  rmdir /s /q search_index")
    print("  python trit_search.py --index")


# ══════════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════════

def test():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("Install: pip install sentence-transformers numpy")
        return

    model_path = str(SAVE_PATH) if SAVE_PATH.exists() else MODEL_NAME
    print(f"Loading: {model_path}")
    model = SentenceTransformer(model_path)

    test_pairs = [
        ("player health",       "var health: float = 100.0"),
        ("take damage",         "func take_damage(amount: float): health -= amount"),
        ("wave spawner",        "func spawn_wave(): for i in range(count): spawn_enemy()"),
        ("ternary weights",     "def quantize(w): t = 0.7 * w.abs().mean()"),
        ("zombie attack",       "func _attack(): if target: target.take_damage(damage)"),
        ("neural network loss", "loss = F.cross_entropy(logits, targets)"),
        ("pathfinding",         "NavigationAgent3D get_next_path_position()"),
        ("card draw",           "func draw_card(): return deck.pop_back()"),
    ]

    print("\nSimilarity scores (higher = better match):\n")
    for query, code in test_pairs:
        q_vec = model.encode([query], normalize_embeddings=True)
        c_vec = model.encode([code],  normalize_embeddings=True)
        score = float(np.dot(q_vec[0], c_vec[0]))
        bar   = "█" * int(score * 30)
        print(f"  {score:.3f} [{bar:<30}] {query!r}")

    print("\nDone.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MiniLM on code")
    parser.add_argument("--train",   action="store_true", help="Train on GitHub code")
    parser.add_argument("--local",   action="store_true", help="Include local files in training")
    parser.add_argument("--test",    action="store_true", help="Test similarity scores")
    parser.add_argument("--install", action="store_true", help="Install into trit_search.py")
    parser.add_argument("--reset",   action="store_true", help="Clear resume status and retrain")
    args = parser.parse_args()

    if args.reset and STATUS_PATH.exists():
        STATUS_PATH.unlink()
        print("Status cleared.")

    if args.train:
        train(use_local=args.local or True)
    elif args.test:
        test()
    elif args.install:
        install()
    else:
        parser.print_help()
