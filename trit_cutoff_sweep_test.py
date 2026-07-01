#!/usr/bin/env python3
"""
Cutoff threshold sweep — does lowering query_codebase's 70%-of-top-score
relevance cutoff recover the recall regressions found in
quality_benchmark_findings.md, and at what token cost?

Tests thresholds: 0.70 (current), 0.60, 0.50, 0.40, 0.30
Reuses the exact same ground-truth + secondary-match queries from
trit_quality_benchmark.py so results are directly comparable.

This does NOT modify trit_mcp_server.py — it reimplements the dedup+cutoff
logic locally with a parameterized threshold, calling engine.search()
directly (same underlying search both real tools use).

Run it:
    python trit_cutoff_sweep_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trit_mcp_server as srv
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
PROJECT_DIR = r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1"

THRESHOLDS = [0.70, 0.60, 0.50, 0.40, 0.30]

# Same test cases as trit_quality_benchmark.py
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


def query_with_threshold(query: str, k: int, threshold_frac: float):
    """Reimplements query_codebase's dedup+cutoff with a parameterized threshold."""
    raw = srv.engine.search(query, k=max(k * 4, 20), base_dir_filter=PROJECT_DIR)
    if not raw:
        return [], ""

    best_per_path = {}
    for r in raw:
        if r["path"] not in best_per_path or r["score"] > best_per_path[r["path"]]["score"]:
            best_per_path[r["path"]] = r
    deduped = sorted(best_per_path.values(), key=lambda r: -r["score"])

    top_score = deduped[0]["score"]
    threshold = top_score * threshold_frac
    filtered = [r for r in deduped if r["score"] >= threshold] or deduped[:1]
    final = filtered[:k]

    lines = []
    hits = []
    for r in final:
        preview = " ".join(r["preview"].split())[:90]
        lines.append(f"{r['path']}  {r['score']:.2f}  {preview}")
        hits.append((r["path"], preview))
    return hits, "\n".join(lines)


def file_and_chunk_match(hits, file_substrings, content_substrings):
    file_found = chunk_found = False
    for path, preview in hits:
        if any(s in path for s in file_substrings):
            file_found = True
            if content_substrings and any(c.lower() in preview.lower() for c in content_substrings):
                chunk_found = True
    return file_found, chunk_found


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def main():
    srv._ensure_loaded()
    if srv._loaded["error"]:
        print(f"Index not ready: {srv._loaded['error']}")
        return

    print("=" * 90)
    print("  CUTOFF THRESHOLD SWEEP — recall vs tokens at each threshold")
    print("=" * 90)
    print(f"  {'Threshold':<12}{'File recall':<14}{'Chunk recall':<15}{'Total tokens':<14}")
    print("  " + "-" * 55)

    results = {}
    for thresh in THRESHOLDS:
        file_pass = chunk_pass = total = 0
        total_tokens = 0
        details = []
        for query, files, content in ALL_QUERIES:
            hits, output = query_with_threshold(query, k=8, threshold_frac=thresh)
            f_ok, c_ok = file_and_chunk_match(hits, files, content)
            total += 1
            file_pass += f_ok
            chunk_pass += c_ok
            total_tokens += count_tokens(output)
            details.append((query, f_ok, c_ok, len(hits)))

        results[thresh] = (file_pass, chunk_pass, total, total_tokens, details)
        print(f"  {thresh:<12}{f'{file_pass}/{total} ({file_pass/total*100:.0f}%)':<14}"
              f"{f'{chunk_pass}/{total} ({chunk_pass/total*100:.0f}%)':<15}{total_tokens:<14}")

    print("\n" + "=" * 90)
    print("  PER-QUERY DETAIL AT EACH THRESHOLD (file-level pass/miss)")
    print("=" * 90)
    for i, (query, _, _) in enumerate(ALL_QUERIES):
        row = f"  {query[:40]:<42}"
        for thresh in THRESHOLDS:
            _, _, _, _, details = results[thresh]
            _, f_ok, c_ok, n = details[i]
            mark = "F+C" if (f_ok and c_ok) else ("F  " if f_ok else "   ")
            row += f"{mark:<8}"
        print(row)
    header = "  " + " " * 42 + "".join(f"{t:<8}" for t in THRESHOLDS)
    print("\n" + header + "  (F=file found, F+C=file+chunk both found)")

    print("\n" + "=" * 90)
    print("  VERDICT")
    print("=" * 90)
    base_file, base_chunk, base_total, base_tokens, _ = results[0.70]
    for thresh in THRESHOLDS[1:]:
        f, c, t, tok, _ = results[thresh]
        df = f - base_file
        dc = c - base_chunk
        dtok = tok - base_tokens
        pct_tok = (dtok / base_tokens * 100) if base_tokens else 0
        print(f"  threshold={thresh}: file recall {'+' if df>=0 else ''}{df}/{t}, "
              f"chunk recall {'+' if dc>=0 else ''}{dc}/{t}, "
              f"tokens {'+' if dtok>=0 else ''}{dtok} ({pct_tok:+.1f}%)")
    print("=" * 90)


if __name__ == "__main__":
    main()
