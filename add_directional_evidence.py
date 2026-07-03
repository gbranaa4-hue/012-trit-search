"""
Post-processes code_entanglement_db.json to add directional/causal claims
to evidence pairs, using each file's real git history.

Turns a symmetric "A and B are 0.5 entangled" score into an asymmetric,
falsifiable claim like "A's file predates B's by 400 days" when the
timestamps are unambiguous -- git blame/log is ground truth here, not a
model guess. Deliberately conservative: only claims direction when the gap
between first-commit dates is large enough (> MIN_GAP_DAYS) that ordinary
commit-timing noise (both authored the same week, different clone/checkout
times, etc.) can't explain it. Ambiguous or unversioned cases are left
alone rather than forced into a guess.

Does not call any LLM -- pure git log, fast enough to run as a
post-processing pass over an existing database rather than re-running the
full (slow) summarization+entanglement pipeline.

Usage:
    python add_directional_evidence.py
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

DB_PATH = Path(__file__).resolve().parent / "code_entanglement_db.json"
MIN_GAP_DAYS = 14  # below this, treat as "same era", not a directional claim

# Real base_dirs, pulled from OBSERVE's own index rather than guessed --
# a hardcoded guess here previously under-resolved most paths (22/565
# resolved instead of the ~95% spot-checked as actually being in git repos).
BASE_DIR_CANDIDATES = []


def _resolve_path(rel_path: str) -> Path:
    for base in BASE_DIR_CANDIDATES:
        p = Path(base) / rel_path
        if p.exists():
            return p
    return None


def _find_git_root(path: Path):
    for parent in [path.parent, *path.parents]:
        if (parent / ".git").exists():
            return parent
    return None


_git_root_cache = {}
_first_commit_cache = {}


def first_commit_date(path: Path):
    """Oldest commit date that touched this file, following renames.
    Returns a timezone-aware datetime, or None if not in a git repo /
    never committed (e.g. still untracked)."""
    key = str(path)
    if key in _first_commit_cache:
        return _first_commit_cache[key]

    root = _git_root_cache.get(str(path.parent))
    if root is None and str(path.parent) not in _git_root_cache:
        root = _find_git_root(path)
        _git_root_cache[str(path.parent)] = root

    result = None
    if root is not None:
        try:
            rel = path.relative_to(root)
            proc = subprocess.run(
                ["git", "log", "--follow", "--format=%aI", "--reverse", "--", str(rel)],
                cwd=str(root), capture_output=True, text=True, timeout=15,
            )
            lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
            if lines:
                result = datetime.fromisoformat(lines[0])
        except Exception:
            result = None

    _first_commit_cache[key] = result
    return result


def main():
    global BASE_DIR_CANDIDATES
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from observe_pipeline import load_engine
    engine = load_engine()
    BASE_DIR_CANDIDATES = sorted({p["base_dir"] for p in engine.path_table})
    print(f"{len(BASE_DIR_CANDIDATES)} known base_dirs from the real index\n")

    db = json.loads(DB_PATH.read_text(encoding="utf-8"))
    total = 0
    resolved_both = 0
    directional = 0

    for pair in db["entanglement"]:
        for ev in pair["evidence"]:
            total += 1
            pa = _resolve_path(ev["a_path"])
            pb = _resolve_path(ev["b_path"])
            if pa is None or pb is None:
                ev["direction"] = None
                continue

            da = first_commit_date(pa)
            db_ = first_commit_date(pb)
            if da is None or db_ is None:
                ev["direction"] = None
                continue

            resolved_both += 1
            gap_days = abs((da - db_).days)
            ev["a_first_commit"] = da.isoformat()
            ev["b_first_commit"] = db_.isoformat()
            ev["commit_gap_days"] = gap_days

            if gap_days >= MIN_GAP_DAYS:
                directional += 1
                if da < db_:
                    ev["direction"] = f"{ev['a_path']} predates {ev['b_path']} by {gap_days} days"
                else:
                    ev["direction"] = f"{ev['b_path']} predates {ev['a_path']} by {gap_days} days"
            else:
                ev["direction"] = None  # same era, not a meaningful claim

    print(f"{total} total evidence entries")
    print(f"{resolved_both} had git history on both sides")
    print(f"{directional} got an unambiguous directional claim (gap >= {MIN_GAP_DAYS} days)")

    DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")
    print(f"\nUpdated in place: {DB_PATH}")

    # Show a few real examples
    examples = [ev for pair in db["entanglement"] for ev in pair["evidence"] if ev.get("direction")]
    print(f"\nSample directional claims:")
    for ev in examples[:10]:
        print(f"  {ev['direction']}")


if __name__ == "__main__":
    main()
