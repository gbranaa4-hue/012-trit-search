#!/usr/bin/env python3
"""
Corrected concept-only benchmark — isolates ONLY genuine vocabulary
mismatch, not "obvious synonym" queries a reasonable naming convention
would make guessable by grep anyway.

The original trit_grep_vs_semantic_test.py's concept-only set included
queries like "how does someone buy an upgrade" where the obvious first
guess ("buy") succeeds immediately — that's not a real test of semantic
search's advantage, since disciplined naming conventions make most
"describe it in plain English" queries grep-guessable in one try.

This benchmark uses ONLY verified cases where the codebase's actual
terminology does NOT match the natural vocabulary a user would reach for —
confirmed by direct inspection before writing the test, not assumed:

1. "creep AI behavior"    — economy code says "creep" (get_creep_upgrades),
                            but the actual AI state logic lives in zombie.gd
                            under "ai_mode"/"AIMode" — zero overlap. Grepping
                            "creep_ai", "creep_state", "creep_behavior" all
                            return nothing; the one "creep" hit in zombie.gd
                            is an unrelated group tag (add_to_group("creeps")).

2. "difficulty scaling"   — the word "difficulty" does not appear ANYWHERE
                            in the codebase's .gd files, yet a real
                            difficulty-scaling mechanic exists as
                            "scaling_interval" in spawner_team_1/2.gd.

3. "in-game currency/money spent on purchases" — "money"/"currency" appear
                            in exactly one file (build_system.gd), calling
                            gm.spend_money() — a function that DOES NOT
                            EXIST in game_manager.gd (only spend_gold()
                            exists). This is apparently a real leftover bug
                            from an incomplete rename — grepping the natural
                            vocabulary leads to a dead reference, not the
                            real implementation.

4. "enemy pathing/navigation toward the player" — "pathing"/"navigation"
                            return only unrelated terrain-tool files; the
                            actual movement logic in zombie.gd uses
                            "move_and_slide"/"_move_toward"/"velocity",
                            none of which share a token with the query.

Run it:
    python trit_naming_mismatch_test.py
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
SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", "addons", ".claude"}
CONTEXT_LINES = 3


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def grep_codebase(pattern: str, base_dir: str = PROJECT_DIR, max_matches: int = 20):
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
    lines = []
    for rel, lineno, snippet in results:
        lines.append(f"{rel}:{lineno}:")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


# ── Verified naming-mismatch cases ─────────────────────────────────────────────
# (query, [ordered realistic keyword guesses], target files that hold the
#  REAL implementation, note explaining the verified mismatch)

NAMING_MISMATCH = [
    (
        "how does creep AI behavior work",
        ["creep_ai", "creep_behavior", "creep_state", "creep"],
        ["zombie.gd"],
        "creep terminology only exists in economy code (get_creep_upgrades); "
        "real AI logic uses ai_mode/AIMode, zero token overlap",
    ),
    (
        "how does difficulty scale as the game progresses",
        ["difficulty", "difficulty_scale", "difficulty_curve"],
        ["spawner_team_1.gd", "spawner_team_2.gd"],
        "'difficulty' does not appear anywhere in the codebase; real "
        "mechanic is named scaling_interval",
    ),
    (
        "where does the game deduct in-game currency for purchases",
        ["currency", "money", "deduct_currency"],
        ["game_manager.gd"],
        "'money'/'currency' only appear in a broken reference (spend_money(), "
        "which does not exist); real implementation is spend_gold()",
    ),
    (
        "how does the enemy path or navigate toward the player",
        ["pathing", "navigation", "pathfind"],
        ["zombie.gd"],
        "'pathing'/'navigation' only match unrelated terrain-tool files; "
        "real movement logic uses move_and_slide/_move_toward/velocity",
    ),
]


def file_and_grep_match(results, expected_files):
    return any(any(f in r[0] for f in expected_files) for r in results)


def run_naming_mismatch():
    print("=" * 90)
    print("  NAMING-MISMATCH TEST — verified vocabulary gaps only, no obvious-synonym queries")
    print("=" * 90)

    grep_tokens_total = qc_tokens_total = 0
    grep_wins = qc_wins = 0

    for query, guesses, expected_files, note in NAMING_MISMATCH:
        print(f"\n  Query: \"{query}\"")
        print(f"  Verified mismatch: {note}")

        cumulative_tokens = 0
        found_at_guess = None
        for gi, guess in enumerate(guesses, 1):
            results = grep_codebase(guess)
            grep_out = format_grep_output(results)
            tok = count_tokens(grep_out)
            cumulative_tokens += tok
            found = file_and_grep_match(results, expected_files)
            print(f"    guess {gi}: \"{guess}\" -> {tok} tokens, "
                  f"{'FOUND target' if found else 'no target match'}")
            if found and found_at_guess is None:
                found_at_guess = gi
                break

        grep_tokens_total += cumulative_tokens
        if found_at_guess:
            grep_wins += 1

        qc_out = srv.query_codebase(query, k=8, project_dir=PROJECT_DIR)
        qc_tok = count_tokens(qc_out)
        qc_found = any(f in qc_out for f in expected_files)
        qc_tokens_total += qc_tok
        if qc_found:
            qc_wins += 1

        print(f"  grep total (through guess {found_at_guess or len(guesses)}): "
              f"{cumulative_tokens} tokens, {'FOUND' if found_at_guess else 'NEVER FOUND'}")
        print(f"  query_codebase (1 call): {qc_tok} tokens, "
              f"{'FOUND' if qc_found else 'NOT FOUND'}")

    n = len(NAMING_MISMATCH)
    print("\n" + "=" * 90)
    print("  RESULT")
    print("=" * 90)
    print(f"  grep (cumulative guessing): {grep_tokens_total} tokens, {grep_wins}/{n} found")
    print(f"  query_codebase (1 call):    {qc_tokens_total} tokens, {qc_wins}/{n} found")

    if qc_wins > grep_wins:
        print(f"\n  query_codebase wins: {qc_wins - grep_wins} more correct answer(s), "
              f"AND {(1 - qc_tokens_total/grep_tokens_total)*100:.0f}% fewer tokens.")
    elif qc_wins == grep_wins:
        pct = (1 - qc_tokens_total / grep_tokens_total) * 100 if grep_tokens_total else 0
        print(f"\n  Tied on recall ({qc_wins}/{n} each) — query_codebase wins on tokens by {pct:.0f}%.")
    else:
        print(f"\n  grep still wins on recall despite the naming mismatch.")
    print("=" * 90)


def main():
    srv._ensure_loaded()
    if srv._loaded["error"]:
        print(f"Index not ready: {srv._loaded['error']}")
        return
    run_naming_mismatch()


if __name__ == "__main__":
    main()
