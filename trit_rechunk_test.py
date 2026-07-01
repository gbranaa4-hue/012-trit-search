#!/usr/bin/env python3
"""
Re-chunking A/B test — does anchoring chunk offsets at function/class
signatures (instead of blind 800-char sliding windows) fix the chunk-level
recall problem found in quality_benchmark_findings.md?

Builds TWO in-memory indexes over horde-beta-version-1 only (no disk writes,
does not touch the live production index at ~/.trit-search/index):
  A) OLD chunker — exact same logic as trit_app.py's build_index():
     text[i:i+800] for i in range(0, len(text), 700)
  B) NEW chunker — anchors chunk start offsets at detected function/class
     signatures (reusing the regex from trit_embed_train.py's pair
     extraction), falling back to blind windowing for text with no
     detected functions (configs, docs, files with unmatched syntax).

Both indexes use the SAME embedding model, SAME search logic (cosine
similarity + same dedup/cutoff as query_codebase), SAME 7 test queries as
trit_quality_benchmark.py, so the only variable is chunk boundary placement.

The critical thing that changes: preview text at query time is regenerated
lazily as text[offset:offset+120] read from disk (see trit_app.py:288-292)
— so if chunker B's offsets start exactly at a function signature line,
the preview will show that signature, whereas chunker A's offsets can land
anywhere inside a function body.

Run it:
    python trit_rechunk_test.py
"""

import os
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_DIR = r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1"
MODEL_PATH = "models/code-minilm"

EXTS = {".py", ".gd", ".js", ".ts", ".cs", ".rs", ".go",
        ".c", ".cpp", ".h", ".java", ".lua", ".rb", ".php",
        ".swift", ".kt", ".dart", ".zig", ".md", ".sh", ".ps1"}

SKIP = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", "target", "models", "search_index",
    "ai_files", "addons",
    "AppData", "Temp", "Windows", "Program Files",
    "Program Files (x86)", "ProgramData",
}
SKIP_PATTERNS = ("_files", "_assets")

FUNC_PATTERNS = [
    re.compile(r'^\s*(?:func|def|fn|function|fun|sub|proc)\s+(\w+)\s*\('),
    re.compile(r'^\s*(?:public|private|protected|static)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*(?:->|:|\{)'),
    re.compile(r'^\s*(?:enum|class|struct)\s+(\w+)'),
]

CHUNK_SIZE = 800
MAX_FUNC_CHUNK = 1200   # cap for very long functions before splitting further

# ── Test queries (same as trit_quality_benchmark.py) ───────────────────────────

GROUND_TRUTH = [
    ("player health and damage handling", ["HealthComponent.gd", "player.gd"], ["take_damage"]),
    ("weapon firing and projectile logic", ["basegun.gd", "gun.gd", "flamethrower.gd", "weapon.gd"], ["shoot"]),
    ("shop UI and upgrade purchasing", ["shopui.gd", "game_manager.gd"], ["upgrade", "purchase"]),
    ("enemy AI state machine", ["zombie.gd"], ["ai_mode", "AIMode"]),
]

SECONDARY_MATCH = [
    ("heal and restore player health", "HealthComponent.gd", ["heal"]),
    ("spend gold currency", "game_manager.gd", ["spend_gold"]),
    ("apply purchased upgrade stat to player", "player.gd", ["apply_upgrade"]),
]

ALL_QUERIES = [(q, files, content) for q, files, content in GROUND_TRUTH] + \
              [(q, [f], content) for q, f, content in SECONDARY_MATCH]


# ── Chunkers ────────────────────────────────────────────────────────────────────

def chunk_old(text: str):
    """Exact reproduction of trit_app.py's blind sliding-window chunker."""
    offsets = []
    for i in range(0, len(text), 700):
        chunk = text[i:i+800]
        if len(chunk.strip()) > 50:
            offsets.append(i)
    return offsets


def chunk_new(text: str):
    """Anchor chunk offsets at detected function/class signature lines.
    Falls back to blind windowing for any gaps (text before the first
    function, or files with no matches at all)."""
    lines = text.splitlines(keepends=True)
    line_offsets = []
    pos = 0
    for line in lines:
        line_offsets.append(pos)
        pos += len(line)

    func_starts = []
    for i, line in enumerate(lines):
        for pat in FUNC_PATTERNS:
            m = pat.match(line)
            if m:
                name = m.group(1)
                if len(name) >= 3 and name not in ('if', 'for', 'while', 'return', 'else'):
                    func_starts.append(line_offsets[i])
                break

    func_starts = sorted(set(func_starts))

    if not func_starts:
        # No functions detected — fall back to blind windowing entirely
        return chunk_old(text)

    offsets = []
    # Blind-window any leading text before the first function (imports, etc)
    if func_starts[0] > 200:
        for i in range(0, func_starts[0], 700):
            if len(text[i:i+800].strip()) > 50:
                offsets.append(i)

    # One chunk anchored at each function signature
    for start in func_starts:
        offsets.append(start)
        # For very long functions, add supplementary chunks past MAX_FUNC_CHUNK
        # so the body isn't silently truncated from the embedding
        next_idx = func_starts.index(start) + 1
        next_start = func_starts[next_idx] if next_idx < len(func_starts) else len(text)
        span = next_start - start
        if span > MAX_FUNC_CHUNK:
            for extra in range(start + MAX_FUNC_CHUNK, next_start, 700):
                offsets.append(extra)

    return sorted(set(offsets))


def build_chunks_for_file(fpath: str, rel: str, chunker):
    try:
        text = open(fpath, encoding="utf-8", errors="ignore").read()
    except Exception:
        return []
    if len(text.strip()) < 100:
        return []
    offsets = chunker(text)
    out = []
    for off in offsets:
        chunk = text[off:off+CHUNK_SIZE]
        if len(chunk.strip()) > 50:
            out.append({
                "text": f"file:{rel}\n{chunk}",
                "rel_path": rel,
                "offset": off,
                "full_path": fpath,
            })
    return out


def scan_and_chunk(base_dir: str, chunker, label: str):
    chunks = []
    n_files = 0
    for root, dirs, fnames in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")
                   and not any(p in d for p in SKIP_PATTERNS)]
        for fname in fnames:
            if Path(fname).suffix.lower() not in EXTS:
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, base_dir)
            file_chunks = build_chunks_for_file(fpath, rel, chunker)
            if file_chunks:
                chunks.extend(file_chunks)
                n_files += 1
    print(f"  [{label}] {len(chunks):,} chunks from {n_files:,} files")
    return chunks


def embed_chunks(model, chunks, label):
    texts = [c["text"] for c in chunks]
    bs = 128
    vecs = []
    t0 = time.time()
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        v = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vecs.append(v)
    vecs = np.vstack(vecs).astype("float32")
    print(f"  [{label}] embedded {len(texts):,} chunks in {time.time()-t0:.1f}s")
    return vecs


def get_preview(chunk):
    """Regenerate preview exactly like trit_app.py does: read from disk at offset."""
    try:
        text = open(chunk["full_path"], encoding="utf-8", errors="ignore").read()
        return text[chunk["offset"]:chunk["offset"]+120].replace("\n", " ")
    except Exception:
        return ""


def search(model, vecs, chunks, query, k=20):
    q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = vecs @ q_vec
    order = np.argsort(-scores)[:k]
    results = []
    for idx in order:
        c = chunks[idx]
        results.append({
            "path": c["rel_path"],
            "score": float(scores[idx]),
            "preview": get_preview(c),
        })
    return results


def dedup_and_cutoff(results, k=8, threshold_frac=0.7):
    """Same logic as query_codebase in trit_mcp_server.py."""
    best_per_path = {}
    for r in results:
        if r["path"] not in best_per_path or r["score"] > best_per_path[r["path"]]["score"]:
            best_per_path[r["path"]] = r
    deduped = sorted(best_per_path.values(), key=lambda r: -r["score"])
    if not deduped:
        return []
    top = deduped[0]["score"]
    filtered = [r for r in deduped if r["score"] >= top * threshold_frac] or deduped[:1]
    return filtered[:k]


def file_and_chunk_match(hits, file_substrings, content_substrings):
    file_found = chunk_found = False
    for r in hits:
        if any(s in r["path"] for s in file_substrings):
            file_found = True
            if content_substrings and any(c.lower() in r["preview"].lower() for c in content_substrings):
                chunk_found = True
    return file_found, chunk_found


def run_recall_test(model, vecs, chunks, label):
    file_pass = chunk_pass = total = 0
    per_query = []
    for query, files, content in ALL_QUERIES:
        raw = search(model, vecs, chunks, query, k=20)
        hits_search_code_style = raw[:10]
        hits_query_codebase_style = dedup_and_cutoff(raw, k=8, threshold_frac=0.7)

        f10, c10 = file_and_chunk_match(hits_search_code_style, files, content)
        f_qc, c_qc = file_and_chunk_match(hits_query_codebase_style, files, content)

        total += 1
        # Report the more permissive (top-10 style) as the primary "file/chunk" numbers,
        # matching what trit_quality_benchmark.py measured for search_code.
        file_pass += f10
        chunk_pass += c10
        per_query.append((query, f10, c10, f_qc, c_qc))

    print(f"\n  [{label}] top-10-style file recall:  {file_pass}/{total} ({file_pass/total*100:.0f}%)")
    print(f"  [{label}] top-10-style chunk recall: {chunk_pass}/{total} ({chunk_pass/total*100:.0f}%)")
    return file_pass, chunk_pass, total, per_query


def main():
    print("Loading model...")
    model = SentenceTransformer(MODEL_PATH)

    print("\nBuilding OLD-chunker index (blind 800-char windows)...")
    old_chunks = scan_and_chunk(PROJECT_DIR, chunk_old, "OLD")
    old_vecs = embed_chunks(model, old_chunks, "OLD")

    print("\nBuilding NEW-chunker index (function-boundary anchored)...")
    new_chunks = scan_and_chunk(PROJECT_DIR, chunk_new, "NEW")
    new_vecs = embed_chunks(model, new_chunks, "NEW")

    print("\n" + "=" * 90)
    print("  RECALL COMPARISON — OLD (blind windows) vs NEW (function-anchored)")
    print("=" * 90)

    old_f, old_c, total, old_detail = run_recall_test(model, old_vecs, old_chunks, "OLD")
    new_f, new_c, total, new_detail = run_recall_test(model, new_vecs, new_chunks, "NEW")

    print("\n" + "=" * 90)
    print("  PER-QUERY DETAIL")
    print("=" * 90)
    print(f"  {'Query':<42}{'OLD file':<10}{'OLD chunk':<11}{'NEW file':<10}{'NEW chunk':<10}")
    print("  " + "-" * 82)
    for i, (query, files, content) in enumerate(ALL_QUERIES):
        _, of, oc, _, _ = old_detail[i]
        _, nf, nc, _, _ = new_detail[i]
        marker = ""
        if nc and not oc:
            marker = "  <-- FIXED by re-chunking"
        elif oc and not nc:
            marker = "  <-- regressed"
        print(f"  {query[:40]:<42}{'PASS' if of else 'MISS':<10}{'PASS' if oc else 'MISS':<11}"
              f"{'PASS' if nf else 'MISS':<10}{'PASS' if nc else 'MISS':<10}{marker}")

    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"  OLD chunker — file: {old_f}/{total} ({old_f/total*100:.0f}%)   chunk: {old_c}/{total} ({old_c/total*100:.0f}%)")
    print(f"  NEW chunker — file: {new_f}/{total} ({new_f/total*100:.0f}%)   chunk: {new_c}/{total} ({new_c/total*100:.0f}%)")
    print(f"\n  Chunk count: OLD={len(old_chunks):,}  NEW={len(new_chunks):,}  "
          f"({(len(new_chunks)/len(old_chunks)-1)*100:+.1f}% chunks)")
    delta = new_c - old_c
    if delta > 0:
        print(f"\n  Function-boundary chunking IMPROVED chunk-level recall by {delta}/{total} "
              f"({delta/total*100:.0f}pp) on this test set.")
    elif delta == 0:
        print(f"\n  No change in chunk-level recall on this test set.")
    else:
        print(f"\n  Function-boundary chunking REGRESSED chunk-level recall by {-delta}/{total}.")
    print("=" * 90)


if __name__ == "__main__":
    main()
