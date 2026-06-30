"""
012 TritLM Mega Training Pipeline
Continual learning across Wikipedia + all major code languages.

Key: uses HuggingFace STREAMING — downloads chunk by chunk.
Never needs more than ~500 MB free on disk at once.
Each cycle trains, saves replay, then moves to next chunk.
Model grows smarter with every cycle, never forgets old cycles.

Cycle order:
  1. Wikipedia (general knowledge)
  2. Python
  3. JavaScript
  4. C / C++
  5. C#
  6. GDScript (via Godot repos)
  7. Rust
  8. Go
  9. Java
  10. TypeScript
  11. SQL
  12. Shell/Bash
  13. More Wikipedia (science, math, history)
  14. Your local files (last — highest priority)

Install:
  pip install datasets tqdm

Usage:
  python trit_mega_train.py                  Show plan + status
  python trit_mega_train.py --start          Run full pipeline (days)
  python trit_mega_train.py --cycle wiki     Run only Wikipedia cycles
  python trit_mega_train.py --cycle code     Run only code cycles
  python trit_mega_train.py --cycle local    Ingest your local files
  python trit_mega_train.py --resume         Resume from last checkpoint
  python trit_mega_train.py --chat           Chat with current model
  python trit_mega_train.py --status         Show what's been trained
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, json, time, random
from pathlib import Path

# Import from trit_distill.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from trit_distill import (
    TritLM, TernaryLinear, TritMemCell, TritBlock,
    ReplayBuffer, set_quant, encode, decode,
    build_vocab, get_batch, get_mixed_batch,
    CTX, D_MODEL, N_HEADS, N_LAYER,
    MODEL_PATH, VOCAB_PATH, REPLAY_PATH, REPLAY_MIX,
    chat
)

os.makedirs("results", exist_ok=True)
os.makedirs("data",    exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

STATUS_PATH = "results/mega_train_status.json"
CHUNK_CHARS = 500_000      # 500 KB per training chunk — fits in RAM easily
ITERS_WIKI  = 1000         # iterations per Wikipedia chunk
ITERS_CODE  = 800          # iterations per code chunk
WARMUP      = 200          # ternary warmup per cycle

# ══════════════════════════════════════════════════════════════════════════════
# STATUS TRACKING
# Remembers which cycles completed so --resume works
# ══════════════════════════════════════════════════════════════════════════════

def load_status():
    if os.path.exists(STATUS_PATH):
        return json.load(open(STATUS_PATH))
    return {"completed": [], "total_chars": 0, "cycles": 0}

def save_status(status):
    json.dump(status, open(STATUS_PATH, "w"), indent=2)

def mark_done(status, cycle_id, chars):
    status["completed"].append(cycle_id)
    status["total_chars"] += chars
    status["cycles"]      += 1
    save_status(status)

def is_done(status, cycle_id):
    return cycle_id in status["completed"]

# ══════════════════════════════════════════════════════════════════════════════
# STREAMING DATA LOADER
# Downloads and yields text chunks without storing full dataset
# ══════════════════════════════════════════════════════════════════════════════

def stream_wikipedia(language="en", max_chars=50_000_000):
    """
    Stream Wikipedia articles.
    Yields chunks of ~CHUNK_CHARS characters.
    Never stores more than one chunk at a time.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return

    print(f"  Streaming Wikipedia ({language})...")
    ds = load_dataset("wikipedia", "20220301.en",
                      split="train", streaming=True,
                      trust_remote_code=True)

    chunk      = ""
    total      = 0
    chunk_num  = 0

    for article in ds:
        text   = article.get("text", "")
        title  = article.get("title", "")
        chunk += f"\n# {title}\n{text}\n"

        if len(chunk) >= CHUNK_CHARS:
            chunk_num += 1
            yield chunk_num, chunk
            total += len(chunk)
            chunk  = ""
            if total >= max_chars:
                break

    if chunk:
        chunk_num += 1
        yield chunk_num, chunk

def stream_code(language, max_chars=20_000_000):
    """
    Stream code from HuggingFace datasets.
    Uses codeparrot/github-code which has 115 languages.
    Yields chunks of ~CHUNK_CHARS characters.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return

    # Map common names to dataset language names
    lang_map = {
        "python":     "Python",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "c":          "C",
        "cpp":        "C++",
        "csharp":     "C#",
        "rust":       "Rust",
        "go":         "Go",
        "java":       "Java",
        "sql":        "SQL",
        "bash":       "Shell",
        "php":        "PHP",
        "swift":      "Swift",
        "kotlin":     "Kotlin",
        "ruby":       "Ruby",
        "scala":      "Scala",
        "lua":        "Lua",
    }

    ds_lang = lang_map.get(language.lower(), language)
    print(f"  Streaming {ds_lang} code...")

    try:
        ds = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            split="train",
            trust_remote_code=True,
            languages=[ds_lang],
        )
    except Exception as e:
        print(f"  Could not load {ds_lang}: {e}")
        return

    chunk     = ""
    total     = 0
    chunk_num = 0

    for sample in ds:
        code   = sample.get("code", "")
        fname  = sample.get("path", "file")
        chunk += f"\n# {fname}\n{code}\n"

        if len(chunk) >= CHUNK_CHARS:
            chunk_num += 1
            yield chunk_num, chunk
            total += len(chunk)
            chunk  = ""
            if total >= max_chars:
                break

    if chunk:
        chunk_num += 1
        yield chunk_num, chunk

def stream_local_files(dirs=None):
    """
    Yield text from your local files.
    Scans for .py, .gd, .js, .cs, .txt, .md files.
    """
    if dirs is None:
        # Edit this default (or pass dirs= explicitly) to point at your own
        # codebase for local streaming.
        dirs = [str(Path(__file__).resolve().parent)]

    EXTS = {".py", ".gd", ".js", ".ts", ".cs", ".txt", ".md",
            ".json", ".cfg", ".toml", ".rs", ".go", ".java"}
    chunk = ""

    for d in dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in EXTS:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    text   = open(fpath, "r", encoding="utf-8", errors="ignore").read()
                    chunk += f"\n# {fpath}\n{text}\n"
                except:
                    continue

                if len(chunk) >= CHUNK_CHARS:
                    yield 0, chunk
                    chunk = ""

    if chunk:
        yield 0, chunk

# ══════════════════════════════════════════════════════════════════════════════
# CORE TRAINING CYCLE
# One cycle = one chunk of data + replay buffer mixing
# ══════════════════════════════════════════════════════════════════════════════

def run_cycle(text, cycle_id, iters=1000, warmup=WARMUP):
    """
    Train one cycle on text.
    Loads existing model, trains with replay buffer, saves.
    """
    if len(text) < 1000:
        print(f"  Skipping {cycle_id} — too short ({len(text)} chars)")
        return 0

    # ── Vocab ─────────────────────────────────────────────────────────────────
    if os.path.exists(VOCAB_PATH):
        saved     = json.load(open(VOCAB_PATH))
        old_chars = set(saved["stoi"].keys())
        all_chars = sorted(old_chars | set(text))
    else:
        all_chars = sorted(set(text))

    stoi = {c:i for i,c in enumerate(all_chars)}
    itos = {i:c for i,c in enumerate(all_chars)}
    json.dump({"stoi": stoi, "itos": itos}, open(VOCAB_PATH, "w"))

    # ── Encode ────────────────────────────────────────────────────────────────
    data  = torch.tensor(encode(text, stoi), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data   = data[split:]

    # ── Replay ────────────────────────────────────────────────────────────────
    replay      = ReplayBuffer()
    replay_data = None
    if not replay.empty():
        replay_data = torch.tensor(encode(replay.get_text(), stoi), dtype=torch.long)

    # ── Model ─────────────────────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        ckpt  = torch.load(MODEL_PATH, map_location=device)
        model = TritLM(vocab_size=len(stoi)).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        lr    = 5e-5   # low LR for continual learning — preserve old knowledge
    else:
        model = TritLM(vocab_size=len(stoi)).to(device)
        lr    = 3e-4

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    # ── Train ─────────────────────────────────────────────────────────────────
    best_val  = float('inf')
    t0        = time.time()
    loss_hist = []

    for it in range(iters):
        set_quant(model, it >= warmup)
        if it % 100 == 0:
            model.reset_memory()

        xb, yb  = get_mixed_batch(train_data, replay_data)
        _, loss = model(xb, yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()

        loss_hist.append(loss.item())

        # ── Live progress bar ─────────────────────────────────────────────────
        elapsed  = time.time() - t0
        eta      = elapsed / (it+1) * (iters - it - 1)
        pct      = (it+1) / iters
        bar_w    = 30
        filled   = int(bar_w * pct)
        bar      = "█" * filled + "░" * (bar_w - filled)
        avg_loss = sum(loss_hist[-50:]) / len(loss_hist[-50:])
        phase    = "TRIT" if it >= warmup else "WARM"

        print(f"\r  [{phase}] [{bar}] {pct*100:5.1f}%  "
              f"loss={avg_loss:.3f}  "
              f"eta={int(eta//60):02d}:{int(eta%60):02d}  "
              f"cycle={cycle_id}",
              end="", flush=True)

        if (it+1) % 200 == 0 or it == iters-1:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data)
                if xv is not None:
                    _, vl = model(xv, yv)
                    val_loss = vl.item()
                else:
                    val_loss = loss.item()

            print(f"\r  [{phase}] [{bar}] {pct*100:5.1f}%  "
                  f"loss={avg_loss:.3f}  val={val_loss:.3f}  "
                  f"eta={int(eta//60):02d}:{int(eta%60):02d}  "
                  f"cycle={cycle_id}   ")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(),
                            "stoi": stoi, "itos": itos,
                            "cycle": cycle_id}, MODEL_PATH)
                print(f"  ✓ Saved  (val={best_val:.3f})")

            model.train()

    print()  # newline after progress bar

    # ── Update replay ─────────────────────────────────────────────────────────
    replay.add(text, sample_rate=0.15)
    elapsed = time.time() - t0
    print(f"  Cycle {cycle_id} done — val={best_val:.3f}  "
          f"{len(text)//1024} KB  {elapsed:.0f}s\n")
    return len(text)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE DEFINITION
# Ordered list of all training cycles
# ══════════════════════════════════════════════════════════════════════════════

# Wikipedia: how many chunks (each ~500 KB = ~1 article batch)
WIKI_CHUNKS  = 100    # 50 MB of Wikipedia
CODE_CHUNKS  = 20     # 10 MB per language

CODE_LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "c",
    "cpp",
    "csharp",
    "rust",
    "go",
    "java",
    "sql",
    "bash",
    "lua",
    "swift",
    "kotlin",
    "ruby",
]

def run_wikipedia_cycles(status, max_chunks=WIKI_CHUNKS, resume=True):
    print(f"\n{'═'*55}")
    print(f"  WIKIPEDIA CYCLES  (target: {max_chunks} chunks)")
    print(f"{'═'*55}\n")

    count = 0
    for chunk_num, chunk in stream_wikipedia(max_chars=max_chunks*CHUNK_CHARS):
        cycle_id = f"wiki_{chunk_num:04d}"
        if resume and is_done(status, cycle_id):
            print(f"  Skipping {cycle_id} (already done)")
            continue

        print(f"  [{cycle_id}] {len(chunk)//1024} KB")
        chars = run_cycle(chunk, cycle_id, iters=ITERS_WIKI)
        mark_done(status, cycle_id, chars)
        count += 1

        if count >= max_chunks:
            break

    print(f"  Wikipedia complete: {count} chunks\n")

def run_code_cycles(status, languages=None, max_chunks=CODE_CHUNKS, resume=True):
    if languages is None:
        languages = CODE_LANGUAGES

    print(f"\n{'═'*55}")
    print(f"  CODE CYCLES  ({len(languages)} languages)")
    print(f"{'═'*55}\n")

    for lang in languages:
        print(f"\n  Language: {lang.upper()}")
        count = 0
        for chunk_num, chunk in stream_code(lang, max_chars=max_chunks*CHUNK_CHARS):
            cycle_id = f"code_{lang}_{chunk_num:04d}"
            if resume and is_done(status, cycle_id):
                print(f"  Skipping {cycle_id} (already done)")
                continue

            print(f"  [{cycle_id}] {len(chunk)//1024} KB")
            chars = run_cycle(chunk, cycle_id, iters=ITERS_CODE)
            mark_done(status, cycle_id, chars)
            count += 1

            if count >= max_chunks:
                break

        if count == 0:
            print(f"  No new chunks for {lang}")

def run_local_cycles(status, resume=True):
    print(f"\n{'═'*55}")
    print(f"  LOCAL FILES CYCLE")
    print(f"{'═'*55}\n")

    for chunk_num, chunk in stream_local_files():
        cycle_id = f"local_{chunk_num:04d}"
        if resume and is_done(status, cycle_id):
            continue
        print(f"  [{cycle_id}] {len(chunk)//1024} KB")
        chars = run_cycle(chunk, cycle_id, iters=1500)
        mark_done(status, cycle_id, chars)

# ══════════════════════════════════════════════════════════════════════════════
# STATUS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def print_status():
    status = load_status()
    completed = status["completed"]

    wiki_done  = [c for c in completed if c.startswith("wiki_")]
    code_done  = [c for c in completed if c.startswith("code_")]
    local_done = [c for c in completed if c.startswith("local_")]

    langs_done = {}
    for c in code_done:
        parts = c.split("_")
        if len(parts) >= 2:
            lang = parts[1]
            langs_done[lang] = langs_done.get(lang, 0) + 1

    print(f"\n{'═'*55}")
    print(f"  MEGA TRAINING STATUS")
    print(f"{'═'*55}")
    print(f"  Total cycles    : {status['cycles']}")
    print(f"  Total data      : {status['total_chars']//1_000_000:.1f} MB")
    print(f"  Wikipedia chunks: {len(wiki_done)}")
    print(f"  Code chunks     : {len(code_done)}")
    print(f"    " + ", ".join(f"{l}:{n}" for l,n in sorted(langs_done.items())))
    print(f"  Local chunks    : {len(local_done)}")
    print(f"  Model exists    : {os.path.exists(MODEL_PATH)}")
    print(f"  Replay size     : ", end="")
    if os.path.exists(REPLAY_PATH):
        print(f"{os.path.getsize(REPLAY_PATH)//1024} KB")
    else:
        print("none yet")

    print(f"\n  Estimated time remaining:")
    wiki_left  = max(0, WIKI_CHUNKS - len(wiki_done))
    code_left  = max(0, CODE_CHUNKS * len(CODE_LANGUAGES) - len(code_done))
    wiki_hrs   = wiki_left  * ITERS_WIKI  / 3600 * 0.05
    code_hrs   = code_left  * ITERS_CODE  / 3600 * 0.05
    print(f"    Wikipedia : {wiki_left} chunks  (~{wiki_hrs:.1f} hrs)")
    print(f"    Code      : {code_left} chunks  (~{code_hrs:.1f} hrs)")
    print(f"    Total     : ~{wiki_hrs+code_hrs:.1f} hours\n")

def print_plan():
    print(f"""
  MEGA TRAINING PLAN
  ══════════════════════════════════════════════════════

  Phase 1 — Wikipedia ({WIKI_CHUNKS} chunks × 500 KB)
    General knowledge, science, history, math, culture
    ~{WIKI_CHUNKS * ITERS_WIKI * 0.05 / 3600:.1f} hours

  Phase 2 — Code ({len(CODE_LANGUAGES)} languages × {CODE_CHUNKS} chunks)
    {', '.join(CODE_LANGUAGES)}
    ~{len(CODE_LANGUAGES) * CODE_CHUNKS * ITERS_CODE * 0.05 / 3600:.1f} hours

  Phase 3 — Your local files
    Godot project, 012 research, notes, scripts
    ~0.5 hours

  Total estimated: ~{(WIKI_CHUNKS*ITERS_WIKI + len(CODE_LANGUAGES)*CODE_CHUNKS*ITERS_CODE)*0.05/3600:.0f} hours

  Storage needed at any time: ~500 MB (streaming)
  Final model size: ~5 MB
  Replay buffer: ~1 MB (permanent)

  You can stop and resume at any time with --resume.
  Every completed cycle is checkpointed.
  """)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  action="store_true", help="Run full pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--cycle",  type=str, default=None,
                        help="Run specific phase: wiki / code / local / LANGUAGE")
    parser.add_argument("--chat",   action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--chunks", type=int, default=None,
                        help="Override number of chunks per source")
    parser.add_argument("--reset",  action="store_true",
                        help="Clear status and start fresh")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(STATUS_PATH):
            os.remove(STATUS_PATH)
            print("Status cleared.")

    status = load_status()
    resume = args.resume or args.start

    if args.status:
        print_status()
        return

    if args.chat:
        chat()
        return

    if args.start or args.resume:
        print_plan()
        print("Starting in 3 seconds... (Ctrl+C to stop anytime)\n")
        time.sleep(3)

        max_chunks = args.chunks or None

        # Phase 1: Wikipedia
        run_wikipedia_cycles(status,
                             max_chunks=max_chunks or WIKI_CHUNKS,
                             resume=resume)

        # Phase 2: All code languages
        run_code_cycles(status,
                        max_chunks=max_chunks or CODE_CHUNKS,
                        resume=resume)

        # Phase 3: Local files (last — highest priority)
        run_local_cycles(status, resume=resume)

        print("\n  MEGA TRAINING COMPLETE")
        print_status()
        return

    if args.cycle:
        c = args.cycle.lower()
        max_c = args.chunks or CODE_CHUNKS

        if c == "wiki":
            run_wikipedia_cycles(status, max_chunks=args.chunks or WIKI_CHUNKS,
                                 resume=resume)
        elif c == "code":
            run_code_cycles(status, max_chunks=max_c, resume=resume)
        elif c == "local":
            run_local_cycles(status, resume=resume)
        elif c in [l.lower() for l in CODE_LANGUAGES]:
            # Single language
            run_code_cycles(status, languages=[c],
                            max_chunks=max_c, resume=resume)
        else:
            print(f"Unknown cycle: {c}")
            print(f"Options: wiki, code, local, {', '.join(CODE_LANGUAGES)}")
        return

    # Default: show plan and status
    print_plan()
    print_status()

if __name__ == "__main__":
    main()
