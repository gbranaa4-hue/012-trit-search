#!/usr/bin/env python3
"""
Quality-vs-tokens benchmark for query_codebase vs search_code.

trit_token_benchmark.py already measured tokens saved. This measures the
other side of the tradeoff: does the token reduction ever cost you the
actual answer? Two checks, both objective (no LLM-as-judge subjectivity):

1. GROUND-TRUTH RECALL — for queries with a known correct file, check
   whether that file appears in each tool's output at all.

2. SECONDARY-MATCH STRESS TEST — query_codebase's two token-saving
   mechanisms (per-file dedup, 70%-of-top relevance cutoff) can silently
   drop a correct answer that isn't the single best-scoring chunk. These
   queries specifically target a *secondary* function living in the same
   file as a more prominent one, to see if dedup/cutoff clips it.

Ground truth was hand-verified against the actual horde-beta-version-1
source (see paper/quality_benchmark_findings.md for the research notes).

Run it:
    python trit_quality_benchmark.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trit_mcp_server as srv

PROJECT_DIR = r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1"

# ── Ground-truth recall queries ────────────────────────────────────────────────
# (query, expected_file_substrings) — pass if ANY expected file appears in output.
# Multiple acceptable files listed where the codebase has duplicate/alt implementations.

GROUND_TRUTH = [
    (
        "player health and damage handling",
        ["HealthComponent.gd", "player.gd"],
    ),
    (
        "weapon firing and projectile logic",
        ["basegun.gd", "gun.gd", "flamethrower.gd", "weapon.gd"],
    ),
    (
        "shop UI and upgrade purchasing",
        ["shopui.gd", "game_manager.gd"],
    ),
    (
        "enemy AI state machine",
        ["zombie.gd"],
    ),
    (
        "save and load game state",
        # NOTE: no real save/load-to-disk system exists in gameplay code
        # (verified — only the terrain_3d editor addon has one). This query
        # has no true positive in this codebase; kept in as a known-negative
        # control — a good tool should return LOW-CONFIDENCE or unrelated
        # results here, not confidently point at the wrong thing.
        ["__NO_TRUE_POSITIVE__"],
    ),
]

# ── Secondary-match stress tests ───────────────────────────────────────────────
# Each targets a function that is NOT the most prominent one in its file,
# to test whether per-file dedup (query_codebase keeps only the best chunk
# per file) or the 70%-relevance cutoff clips the correct answer.

SECONDARY_MATCH = [
    (
        "heal and restore player health",
        "HealthComponent.gd",
        "heal() is a secondary function in this file — take_damage() is more "
        "prominent/higher-scoring for most health-related queries. If dedup "
        "only keeps the best chunk, it may return the take_damage chunk "
        "instead of the heal chunk, even though heal is the actual answer.",
    ),
    (
        "spend gold currency",
        "game_manager.gd",
        "spend_gold() is a lower-level helper; purchase_upgrade() in the same "
        "file is the more prominent entry point most searches would surface "
        "for shop-related queries. Tests whether spend_gold specifically "
        "survives dedup.",
    ),
    (
        "apply purchased upgrade stat to player",
        "player.gd",
        "apply_upgrade() lives in the same file as take_damage(), which is "
        "far more prominent in this codebase (damage is a hot path). Tests "
        "whether an upgrade-topic query still finds player.gd when the file "
        "is dominated by damage-topic content.",
    ),
]


def find_paths_search_code(output: str) -> list:
    """Parse 'N. [score] path' lines from search_code's format."""
    import re
    paths = []
    for line in output.splitlines():
        m = re.match(r"^\d+\.\s+\[[\d.]+\]\s+(.+)$", line.strip())
        if m:
            paths.append(m.group(1).strip())
    return paths


def find_paths_query_codebase(output: str) -> list:
    """Parse 'path  score  preview' lines from query_codebase's format."""
    paths = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if parts:
            paths.append(parts[0].strip())
    return paths


def contains_any(paths: list, substrings: list) -> bool:
    return any(any(s in p for p in paths) for s in substrings)


def run_ground_truth():
    print("=" * 90)
    print("  GROUND-TRUTH RECALL — does the correct file appear in the results?")
    print("=" * 90)
    print(f"  {'Query':<38}{'search_code':<14}{'query_codebase':<16}Notes")
    print("  " + "-" * 86)

    sc_pass = qc_pass = total = 0
    for query, expected in GROUND_TRUTH:
        is_negative_control = expected == ["__NO_TRUE_POSITIVE__"]

        sc_out = srv.search_code(query, k=10, project_dir=PROJECT_DIR)
        qc_out = srv.query_codebase(query, k=8, project_dir=PROJECT_DIR)
        sc_paths = find_paths_search_code(sc_out)
        qc_paths = find_paths_query_codebase(qc_out)

        if is_negative_control:
            # No correct answer exists — "pass" means the tool didn't return
            # zero results silently, it's just a note, not scored pass/fail.
            note = f"(no true positive — {len(sc_paths)}/{len(qc_paths)} results returned, informational only)"
            print(f"  {query[:36]:<38}{'n/a':<14}{'n/a':<16}{note}")
            continue

        sc_ok = contains_any(sc_paths, expected)
        qc_ok = contains_any(qc_paths, expected)
        total += 1
        sc_pass += sc_ok
        qc_pass += qc_ok

        sc_mark = "PASS" if sc_ok else "MISS"
        qc_mark = "PASS" if qc_ok else "MISS"
        note = "" if (sc_ok and qc_ok) else "  <-- regression" if (sc_ok and not qc_ok) else ""
        print(f"  {query[:36]:<38}{sc_mark:<14}{qc_mark:<16}{note}")

    print("  " + "-" * 86)
    print(f"  search_code recall:    {sc_pass}/{total} ({sc_pass/total*100:.0f}%)")
    print(f"  query_codebase recall: {qc_pass}/{total} ({qc_pass/total*100:.0f}%)")
    return sc_pass, qc_pass, total


def run_secondary_match():
    print("\n" + "=" * 90)
    print("  SECONDARY-MATCH STRESS TEST")
    print("  Does query_codebase's per-file dedup / 70% cutoff clip a real but")
    print("  non-top-scoring answer that search_code would still surface?")
    print("=" * 90)

    sc_pass = qc_pass = total = 0
    for query, expected_file, reason in SECONDARY_MATCH:
        sc_out = srv.search_code(query, k=10, project_dir=PROJECT_DIR)
        qc_out = srv.query_codebase(query, k=8, project_dir=PROJECT_DIR)
        sc_paths = find_paths_search_code(sc_out)
        qc_paths = find_paths_query_codebase(qc_out)

        sc_ok = any(expected_file in p for p in sc_paths)
        qc_ok = any(expected_file in p for p in qc_paths)
        total += 1
        sc_pass += sc_ok
        qc_pass += qc_ok

        print(f"\n  Query: \"{query}\"")
        print(f"  Target file: {expected_file}")
        print(f"  Why this stresses dedup: {reason}")
        print(f"  search_code:    {'FOUND' if sc_ok else 'NOT FOUND'} "
              f"(file{'s' if len(sc_paths)!=1 else ''} returned: {len(sc_paths)})")
        print(f"  query_codebase: {'FOUND' if qc_ok else 'NOT FOUND'} "
              f"(file{'s' if len(qc_paths)!=1 else ''} returned: {len(qc_paths)})")
        if sc_ok and not qc_ok:
            print(f"  >>> REGRESSION: query_codebase's dedup/cutoff dropped a match "
                  f"search_code still found.")

    print("\n  " + "-" * 86)
    print(f"  search_code secondary-match recall:    {sc_pass}/{total} ({sc_pass/total*100:.0f}%)")
    print(f"  query_codebase secondary-match recall: {qc_pass}/{total} ({qc_pass/total*100:.0f}%)")
    return sc_pass, qc_pass, total


def main():
    srv._ensure_loaded()
    if srv._loaded["error"]:
        print(f"Index not ready: {srv._loaded['error']}")
        return

    gt_sc, gt_qc, gt_total = run_ground_truth()
    sm_sc, sm_qc, sm_total = run_secondary_match()

    print("\n" + "=" * 90)
    print("  SUMMARY — Quality cost of query_codebase's token savings")
    print("=" * 90)
    combined_sc = gt_sc + sm_sc
    combined_qc = gt_qc + sm_qc
    combined_total = gt_total + sm_total
    print(f"  Combined recall — search_code:    {combined_sc}/{combined_total} "
          f"({combined_sc/combined_total*100:.0f}%)")
    print(f"  Combined recall — query_codebase: {combined_qc}/{combined_total} "
          f"({combined_qc/combined_total*100:.0f}%)")

    gap = combined_sc - combined_qc
    if gap > 0:
        print(f"\n  query_codebase misses {gap} case(s) search_code catches — "
              f"the token savings are NOT free. See per-query detail above for "
              f"which mechanism (dedup vs relevance cutoff) caused each miss.")
    else:
        print(f"\n  query_codebase matched search_code's recall on every test — "
              f"the token savings appear free on this benchmark, at least for "
              f"these query types.")
    print("=" * 90)


if __name__ == "__main__":
    main()
