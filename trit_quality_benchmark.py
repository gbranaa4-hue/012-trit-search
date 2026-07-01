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
        ["take_damage"],
    ),
    (
        "weapon firing and projectile logic",
        ["basegun.gd", "gun.gd", "flamethrower.gd", "weapon.gd"],
        ["shoot"],
    ),
    (
        "shop UI and upgrade purchasing",
        ["shopui.gd", "game_manager.gd"],
        ["upgrade", "purchase"],
    ),
    (
        "enemy AI state machine",
        ["zombie.gd"],
        ["ai_mode", "AIMode"],
    ),
    (
        "save and load game state",
        # NOTE: no real save/load-to-disk system exists in gameplay code
        # (verified — only the terrain_3d editor addon has one). This query
        # has no true positive in this codebase; kept in as a known-negative
        # control — a good tool should return LOW-CONFIDENCE or unrelated
        # results here, not confidently point at the wrong thing.
        ["__NO_TRUE_POSITIVE__"],
        [],
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
        ["heal"],
        "heal() is a secondary function in this file — take_damage() is more "
        "prominent/higher-scoring for most health-related queries. If dedup "
        "only keeps the best chunk, it may return the take_damage chunk "
        "instead of the heal chunk, even though heal is the actual answer.",
    ),
    (
        "spend gold currency",
        "game_manager.gd",
        ["spend_gold"],
        "spend_gold() is a lower-level helper; purchase_upgrade() in the same "
        "file is the more prominent entry point most searches would surface "
        "for shop-related queries. Tests whether spend_gold specifically "
        "survives dedup.",
    ),
    (
        "apply purchased upgrade stat to player",
        "player.gd",
        ["apply_upgrade"],
        "apply_upgrade() lives in the same file as take_damage(), which is "
        "far more prominent in this codebase (damage is a hot path). Tests "
        "whether an upgrade-topic query still finds player.gd when the file "
        "is dominated by damage-topic content.",
    ),
]


def find_hits_search_code(output: str) -> list:
    """Parse search_code's 'N. [score] path\\n   preview' format.
    Returns list of (path, preview_text) tuples — one per result, preview
    is the code snippet line(s) that follow the path line."""
    import re
    hits = []
    lines = output.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^\d+\.\s+\[[\d.]+\]\s+(.+)$", lines[i].strip())
        if m:
            path = m.group(1).strip()
            preview = lines[i+1].strip() if i+1 < len(lines) else ""
            hits.append((path, preview))
        i += 1
    return hits


def find_hits_query_codebase(output: str) -> list:
    """Parse 'path  score  preview' lines from query_codebase's format.
    Returns list of (path, preview_text) tuples."""
    hits = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) >= 3:
            hits.append((parts[0].strip(), parts[2]))
        elif len(parts) >= 1:
            hits.append((parts[0].strip(), ""))
    return hits


def contains_any(paths: list, substrings: list) -> bool:
    return any(any(s in p for p in paths) for s in substrings)


def file_and_chunk_match(hits: list, file_substrings: list, content_substrings: list) -> tuple:
    """
    Returns (file_found, chunk_found).
    file_found: True if any hit's path contains one of file_substrings.
    chunk_found: True if a hit whose path matches ALSO has preview text
        containing one of content_substrings (e.g. the actual function name) —
        i.e. the file was found AND the specific relevant chunk was returned,
        not just some unrelated chunk from the same file.
    """
    file_found = False
    chunk_found = False
    for path, preview in hits:
        if any(s in path for s in file_substrings):
            file_found = True
            if content_substrings and any(c.lower() in preview.lower() for c in content_substrings):
                chunk_found = True
    return file_found, chunk_found


def run_ground_truth():
    print("=" * 90)
    print("  GROUND-TRUTH RECALL — file-level AND chunk-level")
    print("  File-level: does the correct file appear anywhere in output?")
    print("  Chunk-level (stricter): does the returned PREVIEW TEXT actually")
    print("  contain the target function/identifier, not just the filename?")
    print("=" * 90)
    print(f"  {'Query':<36}{'sc:file':<9}{'sc:chunk':<10}{'qc:file':<9}{'qc:chunk':<10}Notes")
    print("  " + "-" * 86)

    sc_file_pass = qc_file_pass = sc_chunk_pass = qc_chunk_pass = total = 0
    for query, expected_files, expected_content in GROUND_TRUTH:
        is_negative_control = expected_files == ["__NO_TRUE_POSITIVE__"]

        sc_out = srv.search_code(query, k=10, project_dir=PROJECT_DIR)
        qc_out = srv.query_codebase(query, k=8, project_dir=PROJECT_DIR)
        sc_hits = find_hits_search_code(sc_out)
        qc_hits = find_hits_query_codebase(qc_out)

        if is_negative_control:
            note = f"(no true positive — {len(sc_hits)}/{len(qc_hits)} results returned, informational only)"
            print(f"  {query[:34]:<36}{'n/a':<9}{'n/a':<10}{'n/a':<9}{'n/a':<10}{note}")
            continue

        sc_file, sc_chunk = file_and_chunk_match(sc_hits, expected_files, expected_content)
        qc_file, qc_chunk = file_and_chunk_match(qc_hits, expected_files, expected_content)
        total += 1
        sc_file_pass += sc_file; qc_file_pass += qc_file
        sc_chunk_pass += sc_chunk; qc_chunk_pass += qc_chunk

        note = ""
        if sc_file and not qc_file:
            note = "  <-- file-level regression"
        elif sc_chunk and not qc_chunk:
            note = "  <-- chunk-level regression (file ok, wrong content)"
        print(f"  {query[:34]:<36}{'PASS' if sc_file else 'MISS':<9}"
              f"{'PASS' if sc_chunk else 'MISS':<10}{'PASS' if qc_file else 'MISS':<9}"
              f"{'PASS' if qc_chunk else 'MISS':<10}{note}")

    print("  " + "-" * 86)
    print(f"  search_code    — file-level: {sc_file_pass}/{total} ({sc_file_pass/total*100:.0f}%)   "
          f"chunk-level: {sc_chunk_pass}/{total} ({sc_chunk_pass/total*100:.0f}%)")
    print(f"  query_codebase — file-level: {qc_file_pass}/{total} ({qc_file_pass/total*100:.0f}%)   "
          f"chunk-level: {qc_chunk_pass}/{total} ({qc_chunk_pass/total*100:.0f}%)")
    return sc_file_pass, qc_file_pass, total


def run_secondary_match():
    print("\n" + "=" * 90)
    print("  SECONDARY-MATCH STRESS TEST (file-level AND chunk-level)")
    print("  Does query_codebase's per-file dedup / 70% cutoff clip a real but")
    print("  non-top-scoring answer that search_code would still surface?")
    print("  Chunk-level checks the preview text actually names the target")
    print("  function, not just that the file appeared.")
    print("=" * 90)

    sc_file_pass = qc_file_pass = sc_chunk_pass = qc_chunk_pass = total = 0
    for query, expected_file, expected_content, reason in SECONDARY_MATCH:
        sc_out = srv.search_code(query, k=10, project_dir=PROJECT_DIR)
        qc_out = srv.query_codebase(query, k=8, project_dir=PROJECT_DIR)
        sc_hits = find_hits_search_code(sc_out)
        qc_hits = find_hits_query_codebase(qc_out)

        sc_file, sc_chunk = file_and_chunk_match(sc_hits, [expected_file], expected_content)
        qc_file, qc_chunk = file_and_chunk_match(qc_hits, [expected_file], expected_content)
        total += 1
        sc_file_pass += sc_file; qc_file_pass += qc_file
        sc_chunk_pass += sc_chunk; qc_chunk_pass += qc_chunk

        print(f"\n  Query: \"{query}\"")
        print(f"  Target file: {expected_file}  Target content: {expected_content}")
        print(f"  Why this stresses dedup: {reason}")
        print(f"  search_code:    file={'FOUND' if sc_file else 'MISS'}  "
              f"chunk={'FOUND' if sc_chunk else 'MISS'}  ({len(sc_hits)} results)")
        print(f"  query_codebase: file={'FOUND' if qc_file else 'MISS'}  "
              f"chunk={'FOUND' if qc_chunk else 'MISS'}  ({len(qc_hits)} results)")
        if sc_file and not qc_file:
            print(f"  >>> FILE-LEVEL REGRESSION: query_codebase's dedup/cutoff dropped "
                  f"the file entirely.")
        elif sc_chunk and not qc_chunk:
            print(f"  >>> CHUNK-LEVEL REGRESSION: file present in both, but "
                  f"query_codebase's kept chunk doesn't contain the target function.")

    print("\n  " + "-" * 86)
    print(f"  search_code    — file-level: {sc_file_pass}/{total} ({sc_file_pass/total*100:.0f}%)   "
          f"chunk-level: {sc_chunk_pass}/{total} ({sc_chunk_pass/total*100:.0f}%)")
    print(f"  query_codebase — file-level: {qc_file_pass}/{total} ({qc_file_pass/total*100:.0f}%)   "
          f"chunk-level: {qc_chunk_pass}/{total} ({qc_chunk_pass/total*100:.0f}%)")
    return sc_file_pass, qc_file_pass, total


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
