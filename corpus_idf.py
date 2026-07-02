"""
Corpus-wide document-frequency index for distinctive tokens.

calibrate_cross_language.py found that raw token overlap fails as a
cross-language relationship signal: common English programming vocabulary
(append, array, before, between) appears in nearly every file, so large
files share MORE raw tokens with unrelated files than a genuinely related
pair shares with its real counterpart. A fixed stopword list can't capture
this -- "neuron"/"threshold" need to count for more than "append"/"array"
specifically because they're much rarer across the actual corpus, which is
a corpus-level fact, not something a hand-picked blocklist can know.

This builds that corpus-level fact once: for every distinctive token,
how many of the ~1800 indexed FILES (not chunks -- file-level presence,
so a word used many times in one file doesn't get overweighted) contain
it. Cached to disk since it requires reading every indexed file's full
text once (~1800 files) -- expensive to redo per query, cheap to
precompute and reuse.

Usage:
    python corpus_idf.py          Build and cache the document-frequency index
"""
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE_PATH = Path(__file__).resolve().parent / "corpus_df_cache.json"

_STOPWORDS = {
    "self", "return", "false", "true", "none", "null", "print", "import",
    "class", "public", "private", "static", "const", "float", "int",
    "string", "bool", "void", "function", "def", "var", "let", "extends",
    "export", "value", "index", "count", "error", "result", "data",
}


def identifier_only_tokens(text: str) -> set:
    idents = {m.lower() for m in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{4,}\b", text)}
    idents -= _STOPWORDS
    strings = {m.lower() for m in re.findall(r'"([^"]{4,40})"', text)}
    return idents | strings


def build_document_frequency(engine, base_dirs: list) -> dict:
    """df[token] = number of distinct files containing that token at least once."""
    seen_files = set()
    df = Counter()
    n_files = 0

    for p in engine.path_table:
        rel = p["rel_path"]
        key = (p["base_dir"], rel)
        if key in seen_files:
            continue
        seen_files.add(key)

        full = None
        for b in base_dirs:
            cand = Path(b) / rel
            if cand.exists() and cand.is_file():
                full = cand
                break
        if full is None:
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not text:
            continue

        n_files += 1
        for tok in identifier_only_tokens(text):
            df[tok] += 1

        if n_files % 200 == 0:
            print(f"  ...{n_files} files processed")

    return dict(df), n_files


def load_or_build():
    if CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return cached["df"], cached["n_files"]

    from trit_app import SearchEngine
    from trit_entanglement import INDEX_DIR, MODEL_PATH
    engine = SearchEngine()
    engine.load(INDEX_DIR, MODEL_PATH, lambda m: print(f"  {m}"))
    while not engine.ready:
        time.sleep(0.2)
    base_dirs = sorted({p["base_dir"] for p in engine.path_table})

    print(f"Building document-frequency index across {len(engine.path_table)} path entries...")
    df, n_files = build_document_frequency(engine, base_dirs)
    print(f"Done: {n_files} files, {len(df)} distinct tokens")

    CACHE_PATH.write_text(json.dumps({"df": df, "n_files": n_files}), encoding="utf-8")
    print(f"Cached: {CACHE_PATH}")
    return df, n_files


def idf(token: str, df: dict, n_files: int) -> float:
    d = df.get(token, 0)
    return math.log((n_files + 1) / (d + 1))


if __name__ == "__main__":
    load_or_build()
