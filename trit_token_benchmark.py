#!/usr/bin/env python3
"""
Honest token-count measurement for query_codebase vs search_code vs the
realistic no-search-tool baseline (Grep-find-files then Read full files).

This exists to replace an unverified "70-97% token reduction" claim with
a real, measured number on this actual codebase and index. No tokens are
estimated by word count — uses tiktoken's cl100k_base encoding (a real
LLM tokenizer) on the literal tool output strings.

Run it:
    python trit_token_benchmark.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trit_mcp_server as srv
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

QUERIES = [
    "player health and damage handling",
    "weapon firing and projectile logic",
    "shop UI and upgrade purchasing",
    "enemy AI state machine",
    "save and load game state",
]

PROJECT_DIR = r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1"


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def naive_baseline_tokens(query, search_code_output):
    """What a tool-less assistant would realistically do: take the top 3
    files search_code already found relevant, and read them in full —
    this is the honest comparison point, not an arbitrary stand-in."""
    paths = []
    for line in search_code_output.splitlines():
        line = line.strip()
        if line[:2].rstrip(".").isdigit() if line[:1].isdigit() else False:
            pass
    # parse "N. [score] path" lines from search_code's format
    import re
    for line in search_code_output.splitlines():
        m = re.match(r"^\d+\.\s+\[[\d.]+\]\s+(.+)$", line.strip())
        if m:
            paths.append(m.group(1).strip())
    paths = paths[:3]

    total = 0
    for p in paths:
        full_path = Path(PROJECT_DIR) / p
        try:
            text = full_path.read_text(encoding="utf-8", errors="ignore")
            total += count_tokens(text)
        except Exception:
            pass
    return total, paths


def main():
    srv._ensure_loaded()
    if srv._loaded["error"]:
        print(f"Index not ready: {srv._loaded['error']}")
        return

    print(f"{'Query':<45}{'naive(top3 full)':<18}{'search_code':<14}{'query_codebase':<16}")
    print("-" * 93)

    totals = {"naive": 0, "search_code": 0, "query_codebase": 0}
    for q in QUERIES:
        sc_out = srv.search_code(q, k=10, project_dir=PROJECT_DIR)
        qc_out = srv.query_codebase(q, k=8, project_dir=PROJECT_DIR)
        naive_tok, naive_paths = naive_baseline_tokens(q, sc_out)

        sc_tok = count_tokens(sc_out)
        qc_tok = count_tokens(qc_out)

        totals["naive"] += naive_tok
        totals["search_code"] += sc_tok
        totals["query_codebase"] += qc_tok

        print(f"{q[:43]:<45}{naive_tok:<18}{sc_tok:<14}{qc_tok:<16}")

    print("-" * 93)
    print(f"{'TOTAL':<45}{totals['naive']:<18}{totals['search_code']:<14}{totals['query_codebase']:<16}")

    def pct_reduction(base, new):
        return (1 - new / base) * 100 if base else 0.0

    print(f"\nquery_codebase vs naive (read top-3 full files): "
          f"{pct_reduction(totals['naive'], totals['query_codebase']):.1f}% fewer tokens")
    print(f"query_codebase vs search_code: "
          f"{pct_reduction(totals['search_code'], totals['query_codebase']):.1f}% fewer tokens")
    print(f"search_code vs naive: "
          f"{pct_reduction(totals['naive'], totals['search_code']):.1f}% fewer tokens")


if __name__ == "__main__":
    main()
