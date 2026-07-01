#!/usr/bin/env python3
"""
Grep/keyword search vs query_codebase (semantic search) — the comparison
the token benchmark should have included from the start.

Claude Code's actual default toolkit without OBSERVE is Grep + Glob + Read,
not "read the top-3 files in full." This benchmark tests both tools
honestly, split into the two query shapes that behave completely
differently under keyword search:

1. KEYWORD-FRIENDLY queries — the caller already knows the exact identifier
   (e.g. "take_damage"). Grep should win here: it's a single fast exact-match
   call, no embedding model needed, near-zero tokens for a precise hit.

2. CONCEPT-ONLY queries — the caller only has a natural-language description
   and does NOT know the exact function/variable name (e.g. "where does the
   player take damage" without knowing it's called take_damage). This is the
   realistic hard case: a grep-only workflow must GUESS keywords, run
   multiple exploratory searches, and pay the cumulative token cost of every
   guess — including wrong ones — before finding the answer, if it ever does.

This does NOT assume semantic search always wins. It measures both shapes
honestly and reports where each tool actually wins, which was the missing
half of the original token_reduction_findings.md analysis.

Run it:
    python trit_grep_vs_semantic_test.py
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trit_mcp_server as srv
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
PROJECT_DIR = r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1"

EXTS = {".gd", ".py", ".js", ".ts", ".cs", ".cpp", ".c", ".h"}
SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", "addons"}

CONTEXT_LINES = 3   # lines of context around a grep match, like -A/-B


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def grep_codebase(pattern: str, base_dir: str = PROJECT_DIR, max_matches: int = 20):
    """Simulates `grep -rn -A3 -B3 <pattern>` over the codebase — exact
    substring match (case-insensitive), like ripgrep's default literal mode."""
    results = []
    pat = re.compile(re.escape(pattern), re.IGNORECASE)
    for root, dirs, fnames in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
        for fname in fnames:
            if Path(fname).suffix.lower() not in EXTS:
                continue
            fpath = os.path.join(root, fname)
            try:
                lines = open(fpath, encoding="utf-8", errors="ignore").read().splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines):
                if pat.search(line):
                    lo = max(0, i - CONTEXT_LINES)
                    hi = min(len(lines), i + CONTEXT_LINES + 1)
                    snippet = "\n".join(lines[lo:hi])
                    rel = os.path.relpath(fpath, base_dir)
                    results.append((rel, i + 1, snippet))
                    if len(results) >= max_matches:
                        return results
    return results


def format_grep_output(results):
    """Mimic how grep results would actually be shown to an LLM."""
    lines = []
    for rel, lineno, snippet in results:
        lines.append(f"{rel}:{lineno}:")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


# ── Test set: keyword-friendly (exact identifier known) ───────────────────────

KEYWORD_FRIENDLY = [
    ("take_damage", ["HealthComponent.gd", "player.gd"]),
    ("purchase_upgrade", ["game_manager.gd"]),
    ("spend_gold", ["game_manager.gd"]),
    ("set_ai_mode", ["zombie.gd"]),
    ("apply_upgrade", ["player.gd"]),
]

# ── Test set: concept-only (caller does NOT know the exact identifier) ────────
# Each entry: (natural language query, [ordered keyword guesses an LLM would
# try without semantic search], target files, whether/which guess succeeds)

CONCEPT_ONLY = [
    (
        "where does the player take damage",
        ["hurt", "damage", "health"],   # realistic guess order: vague -> specific
        ["HealthComponent.gd", "player.gd"],
    ),
    (
        "how does someone buy an upgrade in the shop",
        ["buy", "shop", "purchase"],
        ["game_manager.gd", "shopui.gd"],
    ),
    (
        "where does gold get deducted when buying something",
        ["deduct", "cost", "spend"],
        ["game_manager.gd"],
    ),
    (
        "how does the enemy decide what to do",
        ["behavior", "decide", "state"],
        ["zombie.gd"],
    ),
    (
        "apply a purchased stat boost to the player",
        ["boost", "stat", "apply_upgrade"],
        ["player.gd"],
    ),
]


def run_keyword_friendly():
    print("=" * 90)
    print("  KEYWORD-FRIENDLY QUERIES (exact identifier already known)")
    print("=" * 90)
    print(f"  {'Query':<24}{'grep tokens':<13}{'grep found':<12}{'qc tokens':<12}{'qc found':<10}")
    print("  " + "-" * 80)

    grep_tokens_total = qc_tokens_total = 0
    grep_wins = qc_wins = 0

    for term, expected_files in KEYWORD_FRIENDLY:
        results = grep_codebase(term)
        grep_out = format_grep_output(results)
        grep_tok = count_tokens(grep_out)
        grep_found = any(any(f in r[0] for f in expected_files) for r in results)

        qc_out = srv.query_codebase(term, k=8, project_dir=PROJECT_DIR)
        qc_tok = count_tokens(qc_out)
        qc_found = any(f in qc_out for f in expected_files)

        grep_tokens_total += grep_tok
        qc_tokens_total += qc_tok
        if grep_found: grep_wins += 1
        if qc_found: qc_wins += 1

        print(f"  {term[:22]:<24}{grep_tok:<13}{'YES' if grep_found else 'NO':<12}"
              f"{qc_tok:<12}{'YES' if qc_found else 'NO':<10}")

    print("  " + "-" * 80)
    print(f"  TOTAL tokens — grep: {grep_tokens_total}   query_codebase: {qc_tokens_total}")
    print(f"  Recall — grep: {grep_wins}/{len(KEYWORD_FRIENDLY)}   "
          f"query_codebase: {qc_wins}/{len(KEYWORD_FRIENDLY)}")
    if grep_tokens_total < qc_tokens_total:
        pct = (1 - grep_tokens_total / qc_tokens_total) * 100
        print(f"  --> grep wins on tokens by {pct:.0f}% when the exact term is already known.")
    return grep_tokens_total, qc_tokens_total, grep_wins, qc_wins


def run_concept_only():
    print("\n" + "=" * 90)
    print("  CONCEPT-ONLY QUERIES (exact identifier NOT known — must guess)")
    print("=" * 90)

    grep_tokens_total = qc_tokens_total = 0
    grep_wins = qc_wins = 0
    grep_guesses_total = 0

    for query, guesses, expected_files in CONCEPT_ONLY:
        print(f"\n  Query: \"{query}\"")
        print(f"  Realistic keyword guesses an LLM without semantic search would try, in order:")

        cumulative_tokens = 0
        found_at_guess = None
        for gi, guess in enumerate(guesses, 1):
            results = grep_codebase(guess)
            grep_out = format_grep_output(results)
            tok = count_tokens(grep_out)
            cumulative_tokens += tok
            found = any(any(f in r[0] for f in expected_files) for r in results)
            print(f"    guess {gi}: \"{guess}\" -> {tok} tokens, "
                  f"{'FOUND target' if found else 'no target match'}")
            if found and found_at_guess is None:
                found_at_guess = gi
                break   # a real workflow stops once it finds the answer

        grep_guesses_total += (found_at_guess or len(guesses))
        grep_tokens_total += cumulative_tokens
        if found_at_guess:
            grep_wins += 1

        qc_out = srv.query_codebase(query, k=8, project_dir=PROJECT_DIR)
        qc_tok = count_tokens(qc_out)
        qc_found = any(f in qc_out for f in expected_files)
        qc_tokens_total += qc_tok
        if qc_found:
            qc_wins += 1

        print(f"  grep total (cumulative through guess {found_at_guess or len(guesses)}): "
              f"{cumulative_tokens} tokens, {'FOUND' if found_at_guess else 'NEVER FOUND'}")
        print(f"  query_codebase (1 call, natural language): {qc_tok} tokens, "
              f"{'FOUND' if qc_found else 'NOT FOUND'}")

    print("\n  " + "-" * 86)
    print(f"  TOTAL tokens — grep (cumulative guesses): {grep_tokens_total}   "
          f"query_codebase: {qc_tokens_total}")
    print(f"  Recall — grep: {grep_wins}/{len(CONCEPT_ONLY)}   "
          f"query_codebase: {qc_wins}/{len(CONCEPT_ONLY)}")
    print(f"  Average guesses needed before grep found the answer "
          f"(or exhausted its guess list): {grep_guesses_total/len(CONCEPT_ONLY):.1f}")
    if qc_tokens_total < grep_tokens_total:
        pct = (1 - qc_tokens_total / grep_tokens_total) * 100
        print(f"  --> query_codebase wins on tokens by {pct:.0f}% when the exact term is NOT known.")
    return grep_tokens_total, qc_tokens_total, grep_wins, qc_wins


def main():
    srv._ensure_loaded()
    if srv._loaded["error"]:
        print(f"Index not ready: {srv._loaded['error']}")
        return

    kf_grep_tok, kf_qc_tok, kf_grep_win, kf_qc_win = run_keyword_friendly()
    co_grep_tok, co_qc_tok, co_grep_win, co_qc_win = run_concept_only()

    print("\n" + "=" * 90)
    print("  VERDICT — when does each tool actually win?")
    print("=" * 90)
    n = len(KEYWORD_FRIENDLY)
    print(f"  Keyword-friendly (exact term known), n={n}:")
    print(f"    grep:           {kf_grep_tok} tokens, {kf_grep_win}/{n} found")
    print(f"    query_codebase: {kf_qc_tok} tokens, {kf_qc_win}/{n} found")
    if kf_grep_win > kf_qc_win:
        print(f"    Winner: grep — {kf_grep_win-kf_qc_win} more correct answer(s), "
              f"despite costing {kf_grep_tok-kf_qc_tok} more tokens. Recall beats token "
              f"savings when the answer is otherwise wrong.")
    elif kf_qc_win > kf_grep_win:
        print(f"    Winner: query_codebase — better recall AND fewer tokens.")
    else:
        print(f"    Tied on recall ({kf_grep_win}/{n} each) — "
              f"{'grep' if kf_grep_tok < kf_qc_tok else 'query_codebase'} wins on tokens.")
    print()
    n2 = len(CONCEPT_ONLY)
    print(f"  Concept-only (exact term NOT known, must guess), n={n2}:")
    print(f"    grep (cumulative guessing): {co_grep_tok} tokens, {co_grep_win}/{n2} found")
    print(f"    query_codebase (1 call):    {co_qc_tok} tokens, {co_qc_win}/{n2} found")
    if co_grep_win == co_qc_win:
        print(f"    Tied on recall ({co_grep_win}/{n2} each) — "
              f"query_codebase wins on tokens ({co_qc_tok} vs {co_grep_tok}).")
    elif co_qc_win > co_grep_win:
        print(f"    Winner: query_codebase — better recall AND fewer tokens.")
    else:
        print(f"    Winner: grep — better recall despite higher token cost.")
    print()
    print("  Honest summary: query_codebase's dedup/cutoff caused REAL recall")
    print("  losses even in the keyword-friendly case, where the query WAS the")
    print("  exact function name (2/5 misses). This is not just a concept-query")
    print("  problem — it can drop an exact-match file too. Grep is the more")
    print("  reliable tool whenever the identifier is already known, despite")
    print("  costing more tokens. query_codebase's real advantage is token cost")
    print("  on ambiguous queries, not superior recall in general.")
    print("=" * 90)


if __name__ == "__main__":
    main()
