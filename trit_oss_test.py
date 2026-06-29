"""
012 OSS Benchmark
Downloads real open source projects, indexes them, runs searches,
compares baseline MiniLM vs fine-tuned model.

Projects (diverse languages):
  - fastapi      Python web framework      ~50k lines
  - godot        GDScript demo scripts     ~20k lines
  - redis        C systems code            ~100k lines
  - typescript   TS compiler               ~300k lines
  - tokio        Rust async runtime        ~80k lines
  - gin          Go web framework          ~30k lines
  - odin-lang    Odin language examples    ~40k lines

Usage:
  python trit_oss_test.py --download     Clone all projects
  python trit_oss_test.py --index        Index with both models
  python trit_oss_test.py --benchmark    Run search benchmark
  python trit_oss_test.py --all          Do everything
"""

import os, subprocess, sys, time, json, random
from pathlib import Path
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

OSS_DIR      = Path(__file__).parent / "oss_projects"
RESULTS_DIR  = Path(__file__).parent / "oss_results"
FINE_TUNED   = str(Path(__file__).parent / "models" / "code-minilm")
BASELINE     = "all-MiniLM-L6-v2"

PROJECTS = [
    # (name, git_url, primary_language, file_extensions)
    ("fastapi",    "https://github.com/tiangolo/fastapi",           "Python",     [".py"]),
    ("godot-demo", "https://github.com/godotengine/godot-demo-projects", "GDScript", [".gd"]),
    ("redis",      "https://github.com/redis/redis",                "C",          [".c", ".h"]),
    ("tokio",      "https://github.com/tokio-rs/tokio",             "Rust",       [".rs"]),
    ("gin",        "https://github.com/gin-gonic/gin",              "Go",         [".go"]),
    ("deno",       "https://github.com/denoland/deno_std",           "TypeScript", [".ts"]),
    ("godot-gdext","https://github.com/godot-rust/gdext",           "Rust+GDScript",[".rs",".gd"]),
]

# Search queries with expected relevant files/concepts
# Format: (query, [keywords that should appear in top results])
SEARCH_QUERIES = [
    # Python / FastAPI
    ("http route handler",           ["router", "route", "endpoint", "app"]),
    ("request validation middleware",["middleware", "depends", "validator"]),
    ("async database query",         ["async", "await", "db", "session"]),
    ("authentication token",         ["auth", "token", "jwt", "bearer"]),
    ("pydantic model schema",        ["BaseModel", "Field", "schema", "model"]),

    # GDScript / Godot
    ("enemy spawner wave",           ["spawn", "wave", "enemy", "instantiate"]),
    ("player movement physics",      ["velocity", "move_and_slide", "CharacterBody"]),
    ("health damage system",         ["health", "damage", "take_damage", "die"]),
    ("animation state machine",      ["AnimationPlayer", "animation", "state"]),
    ("signal connection",            ["signal", "connect", "emit"]),

    # C / Redis
    ("memory allocation",            ["malloc", "zmalloc", "alloc", "free"]),
    ("hash table lookup",            ["hash", "dict", "lookup", "find"]),
    ("network socket connection",    ["socket", "connect", "bind", "listen"]),
    ("command parser",               ["parse", "command", "argc", "argv"]),
    ("event loop",                   ["event", "loop", "epoll", "select"]),

    # Rust / Tokio
    ("async task spawn",             ["spawn", "async", "await", "task"]),
    ("channel message passing",      ["channel", "send", "recv", "mpsc"]),
    ("error handling result",        ["Result", "Error", "unwrap", "?"]),
    ("mutex lock shared state",      ["Mutex", "Arc", "lock", "RwLock"]),
    ("tcp listener accept",          ["TcpListener", "accept", "bind", "stream"]),

    # Go / Gin
    ("http middleware chain",        ["middleware", "Handler", "Next", "gin"]),
    ("json response encoder",        ["json", "Marshal", "encode", "response"]),
    ("goroutine worker pool",        ["goroutine", "chan", "WaitGroup", "worker"]),
    ("context cancellation",         ["context", "cancel", "Done", "timeout"]),
    ("struct method receiver",       ["func", "receiver", "method", "struct"]),

    # TypeScript
    ("generic type constraint",      ["generic", "extends", "constraint", "type"]),
    ("async promise chain",          ["async", "await", "Promise", "then"]),
    ("interface implementation",     ["interface", "implements", "class", "type"]),
    ("error boundary handler",       ["error", "catch", "throw", "Error"]),
    ("module export import",         ["export", "import", "module", "from"]),
]

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_projects(shallow=True):
    OSS_DIR.mkdir(exist_ok=True)

    for name, url, lang, _ in PROJECTS:
        dest = OSS_DIR / name
        if dest.exists():
            print(f"  {name}: already downloaded")
            continue
        print(f"  Cloning {name} ({lang})...")
        cmd = ["git", "clone", "--depth=1", url, str(dest)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            # Count files
            count = sum(1 for _ in dest.rglob("*") if _.is_file())
            print(f"    Done: {count:,} files")
        except subprocess.CalledProcessError as e:
            print(f"    Failed: {e.stderr.decode()[:200]}")

    # Count total lines
    total_lines = 0
    total_files = 0
    for name, _, _, exts in PROJECTS:
        dest = OSS_DIR / name
        if not dest.exists():
            continue
        for ext in exts:
            for f in dest.rglob(f"*{ext}"):
                try:
                    lines = len(f.read_text(errors='ignore').splitlines())
                    total_lines += lines
                    total_files += 1
                except:
                    pass

    print(f"\nTotal: {total_files:,} source files, ~{total_lines:,} lines of code")


# ══════════════════════════════════════════════════════════════════════════════
# INDEX
# ══════════════════════════════════════════════════════════════════════════════

def load_oss_chunks(max_per_project=5000):
    """Load code chunks from all OSS projects."""
    chunks    = []   # (text, source_file, project, language)
    CHUNK_SZ  = 600
    OVERLAP   = 100

    SKIP_DIRS = {".git", "node_modules", "__pycache__", "target",
                 "build", "dist", ".cache", "vendor"}

    for name, _, lang, exts in PROJECTS:
        dest = OSS_DIR / name
        if not dest.exists():
            continue

        project_chunks = []
        for ext in exts:
            for fpath in dest.rglob(f"*{ext}"):
                # Skip unwanted dirs
                if any(p in SKIP_DIRS for p in fpath.parts):
                    continue
                try:
                    text = fpath.read_text(errors='ignore')
                    if len(text.strip()) < 100:
                        continue
                    # Chunk it
                    start = 0
                    while start < len(text):
                        chunk = text[start:start+CHUNK_SZ]
                        if len(chunk.strip()) > 50:
                            project_chunks.append((
                                chunk,
                                str(fpath.relative_to(OSS_DIR)),
                                name,
                                lang
                            ))
                        start += CHUNK_SZ - OVERLAP
                except:
                    pass

        # Cap per project
        if len(project_chunks) > max_per_project:
            project_chunks = random.sample(project_chunks, max_per_project)
        chunks.extend(project_chunks)
        print(f"  {name}: {len(project_chunks):,} chunks ({lang})")

    print(f"\nTotal chunks: {len(chunks):,}")
    return chunks


def build_index(model, chunks):
    """Encode all chunks into a FAISS index."""
    try:
        import faiss
    except ImportError:
        print("Install: pip install faiss-cpu")
        return None, None

    texts = [c[0] for c in chunks]
    meta  = [{"file": c[1], "project": c[2], "lang": c[3], "preview": c[0][:120]} for c in chunks]

    print(f"  Encoding {len(texts):,} chunks...")
    t0   = time.time()
    bs   = 256
    vecs = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        v = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vecs.append(v)
        if (i // bs) % 10 == 0:
            pct = i / len(texts) * 100
            print(f"\r    {pct:.0f}%  ({i:,}/{len(texts):,})", end="", flush=True)

    print(f"\r    100%  ({len(texts):,}/{len(texts):,})")
    vecs = np.vstack(vecs).astype("float32")

    elapsed = time.time() - t0
    print(f"  Encoded in {elapsed:.1f}s  ({len(texts)/elapsed:.0f} chunks/sec)")

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index, meta


def search_index(index, meta, model, query, k=10):
    vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "score":   float(score),
                "file":    meta[idx]["file"],
                "project": meta[idx]["project"],
                "lang":    meta[idx]["lang"],
                "preview": meta[idx]["preview"],
            })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def score_results(results, keywords):
    """
    Score search results based on keyword presence.
    Returns (hit_rate, avg_score_of_hits)
    """
    hits = 0
    for r in results[:5]:  # check top 5
        text = (r["file"] + " " + r["preview"]).lower()
        if any(kw.lower() in text for kw in keywords):
            hits += 1
    return hits / min(5, len(results)) if results else 0


def run_oss_benchmark(model, model_name, index, meta):
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {model_name}")
    print(f"{'='*60}\n")

    scores  = []
    timings = []
    details = []

    for query, keywords in SEARCH_QUERIES:
        t0      = time.time()
        results = search_index(index, meta, model, query, k=10)
        elapsed = (time.time() - t0) * 1000

        hit_rate = score_results(results, keywords)
        scores.append(hit_rate)
        timings.append(elapsed)

        top_file = results[0]["file"] if results else "none"
        icon     = "✓" if hit_rate > 0 else "✗"
        details.append({
            "query":    query,
            "hit_rate": hit_rate,
            "top_file": top_file,
            "passed":   hit_rate > 0,
        })
        print(f"  {icon} [{hit_rate:.0%}] {query[:45]:<45}  →  {top_file[:50]}")

    avg_hit  = np.mean(scores) * 100
    avg_time = np.mean(timings)

    print(f"\n{'─'*60}")
    print(f"  Hit rate  : {avg_hit:.1f}%")
    print(f"  Avg query : {avg_time:.0f}ms")
    print(f"  Model     : {model_name}")
    print(f"{'='*60}")

    return {"model": model_name, "hit_rate": avg_hit, "avg_ms": avg_time, "details": details}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download",  action="store_true")
    parser.add_argument("--index",     action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--chunks",    type=int, default=5000,
                        help="Max chunks per project (default 5000)")
    args = parser.parse_args()

    if args.all:
        args.download = args.index = args.benchmark = True

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.download:
        print("\n[1/3] Downloading OSS projects...")
        download_projects()

    if args.index or args.benchmark:
        print("\n[2/3] Loading chunks from OSS projects...")
        chunks = load_oss_chunks(max_per_project=args.chunks)

        if not chunks:
            print("No chunks loaded. Run --download first.")
            sys.exit(1)

        from sentence_transformers import SentenceTransformer

        reports = []

        # Baseline
        print(f"\nBuilding index with baseline MiniLM...")
        base_model = SentenceTransformer(BASELINE)
        base_index, meta = build_index(base_model, chunks)

        if args.benchmark and base_index is not None:
            r = run_oss_benchmark(base_model, "baseline-MiniLM", base_index, meta)
            reports.append(r)

        del base_model

        # Fine-tuned
        if Path(FINE_TUNED).exists():
            print(f"\nBuilding index with fine-tuned model...")
            ft_model = SentenceTransformer(FINE_TUNED)
            ft_index, _ = build_index(ft_model, chunks)

            if args.benchmark and ft_index is not None:
                r = run_oss_benchmark(ft_model, "fine-tuned-012", ft_index, meta)
                reports.append(r)
        else:
            print(f"Fine-tuned model not found at {FINE_TUNED}")

        # Summary
        if len(reports) == 2:
            diff = reports[1]["hit_rate"] - reports[0]["hit_rate"]
            speed_ratio = reports[0]["avg_ms"] / reports[1]["avg_ms"]
            print(f"\n{'='*60}")
            print(f"  SUMMARY")
            print(f"{'='*60}")
            print(f"  Baseline accuracy : {reports[0]['hit_rate']:.1f}%")
            print(f"  Fine-tuned accuracy: {reports[1]['hit_rate']:.1f}%")
            print(f"  Improvement       : {diff:+.1f}%")
            print(f"  Speed improvement : {speed_ratio:.1f}x faster")
            print(f"{'='*60}")

            # Save results
            out = RESULTS_DIR / "oss_benchmark.json"
            out.write_text(json.dumps(reports, indent=2))
            print(f"\n  Full results saved to {out}")
