"""
Shared "intake" stage for every OBSERVE analysis tool.

Before this existed, trit_entanglement.py, code_references.py,
corpus_idf.py, calibrate_consensus.py, calibrate_genuine.py, and
add_directional_evidence.py each independently duplicated the same
~15-line block: create a SearchEngine, load the index, spin-wait for
ready, group chunks into projects, resolve base_dirs. Every single script
re-paid the full index-load cost (~8-10s) from scratch, even run
back-to-back, and a bug fixed in one copy (e.g. the hash() vs stable_hash
seeding bug, or the CONTAINER_PREFIXES base_dir fix) had to be manually
re-applied to five other copies rather than being fixed once.

This is the first piece of restructuring the pipeline the way a jet
engine's stages feed each other -- one shared intake (load + resolve),
rather than six independent scripts each re-deriving the same raw input.
Every constant and helper that describes THE INDEX ITSELF (not a specific
analysis technique) lives here now; trit_entanglement.py and friends
import from this module instead of defining their own copies.
"""
import re
import time
import zlib
from collections import defaultdict
from pathlib import Path

from trit_app import SearchEngine

import json

INDEX_DIR = str(Path.home() / ".trit-search" / "index")
MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "code-minilm")
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "all-MiniLM-L6-v2"

# ── PER-DEPLOYMENT CONFIG ────────────────────────────────────────────────
# Everything that describes a SPECIFIC machine / user / software install --
# the folder layout to strip and the non-code denylist -- was previously
# hardcoded to one user (Users/gbran/OneDrive/...) and one person's
# installed apps (Image-Line, Call of Duty, ...). That made the framework
# unusable on any other machine or for any other software implementation
# without editing source. This block makes it portable:
#   1. Container prefixes are DERIVED from the current user's home dir,
#      so they're correct on any machine without hardcoding a username.
#   2. An optional observe_config.json next to this module can override or
#      extend both lists for a new deployment, without touching code.
_CONFIG_PATH = Path(__file__).resolve().parent / "observe_config.json"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


_CONFIG = _load_config()


def _strip_drive(p: str) -> str:
    """Remove a leading Windows drive letter (C:/, G:\\, d:, ...) OR a
    leading POSIX slash. Real portability bug this fixes: the old code did
    p.lstrip('C:/'), which (a) only handled the C: drive -- files indexed
    from G:/ or E:/ were never stripped and so misgrouped -- and (b) is a
    character-set strip, so a real folder literally named starting with
    'C' could be partially eaten. A drive-anchored regex handles every
    drive letter and can't over-strip a folder name."""
    p = re.sub(r"^[A-Za-z]:[/\\]?", "", p)
    return p.lstrip("/\\")


def _default_container_prefixes() -> list:
    """Container prefixes derived from Path.home() instead of a hardcoded
    username, in the same drive-stripped forward-slash form
    _infer_project_name compares against. Ordered most-specific-first so
    the catch-all home prefix only applies when nothing deeper matched."""
    home_norm = _strip_drive(str(Path.home()).replace("\\", "/"))
    subfolders = [
        "OneDrive/Documents", "OneDrive/Desktop", "OneDrive",
        "Documents", "Desktop", "Downloads",
    ]
    prefixes = [f"{home_norm}/{s}/" for s in subfolders]
    prefixes.append(f"{home_norm}/")   # catch-all -- must come last
    return prefixes


# config override wins; otherwise derive from this machine's home dir
CONTAINER_PREFIXES = _CONFIG.get("container_prefixes") or _default_container_prefixes()

# Code file extensions -- if the "first path component" after stripping
# containers ends in one of these, the file was indexed directly under a
# folder with no further subfolder (a loose script, not a real project).
# Measured real case: "NTeleportation.cs" (579 chunks) was a single large
# file misgrouped as if it were a project folder.
_CODE_EXTENSIONS = {
    ".py", ".gd", ".js", ".ts", ".cs", ".c", ".cpp", ".h", ".hpp",
    ".java", ".lua", ".rb", ".php", ".md", ".sh", ".ps1",
}

# Folders that are clearly not "a codebase" -- reported separately, not
# silently dropped. The default list is this machine's installed non-code
# software; a new deployment can replace it entirely via
# observe_config.json's "non_project_hints" (empty list disables it).
_DEFAULT_NON_PROJECT_HINTS = {
    "image-line", "ableton", "native instruments", "universal audio",
    "fabfilter", "blue cat audio", "xfer", "vital", "tone2", "oeksound",
    "naughty seal audio", "zoom", "max 8", "my cheat tables",
    "call of duty", "call of duty modern warfare", "overwatch",
    "starcraft ii", "stronghold kingdoms", "addictive keys logs",
}
NON_PROJECT_HINTS = set(_CONFIG["non_project_hints"]) if "non_project_hints" in _CONFIG \
    else _DEFAULT_NON_PROJECT_HINTS

MIN_CHUNKS_PER_PROJECT = _CONFIG.get("min_chunks_per_project", 8)   # ignore noise -- a handful of stray chunks isn't "a project"


def stable_hash(s: str) -> int:
    """Deterministic string -> int, safe to use as an RNG seed across
    separate process runs. Real bug found and fixed: code throughout this
    pipeline used Python's builtin hash(project) as a random seed, assuming
    it was reproducible run-to-run because it "looks seeded." It isn't --
    Python randomizes string hashing per-process by default (PYTHONHASHSEED,
    a security feature since 3.3), confirmed directly: hash("Spikeling-
    Project") returned three DIFFERENT values across three separate
    `python -c` invocations. zlib.crc32 is not randomized and gives the
    same value every time."""
    return zlib.crc32(s.encode("utf-8"))


def _infer_project_name(base_dir: str, rel_path: str) -> str:
    full = (base_dir.rstrip("/\\") + "/" + rel_path.replace("\\", "/")).replace("//", "/")
    full = _strip_drive(full)   # handles any drive letter, not just C: (see _strip_drive)
    for prefix in CONTAINER_PREFIXES:
        if full.startswith(prefix):
            full = full[len(prefix):]
            break
    parts = full.split("/")
    first = parts[0] if parts else "unknown"
    # A "project name" that's actually a bare filename (e.g. "NTeleportation.cs")
    # means this file was indexed directly under a container with no real
    # project subfolder -- group these under one explicit bucket instead of
    # each pretending to be its own separate "project."
    if Path(first).suffix.lower() in _CODE_EXTENSIONS:
        return "(loose scripts, no project folder)"
    return first


_DUPLICATE_SUFFIX = re.compile(r"\s*\(\d+\)$")   # "Foo(1)", "Foo (2)" -> "Foo"

def group_chunks_by_project(engine: SearchEngine, merge_duplicate_suffixes: bool = True):
    """
    Returns {project_name: [chunk_indices]} using the loaded index's
    metadata directly -- no new search calls needed.

    merge_duplicate_suffixes: if True, folders differing only by a
    trailing "(1)"/"(2)" (the pattern Windows/browsers add when the same
    folder gets downloaded/extracted twice) are merged into one project.
    """
    groups = defaultdict(list)
    for i, m in enumerate(engine.metadata):
        if not (isinstance(m, list) and engine.path_table):
            continue
        p_idx, offset = m
        p = engine.path_table[p_idx]
        project = _infer_project_name(p["base_dir"], p["rel_path"])
        if merge_duplicate_suffixes:
            project = _DUPLICATE_SUFFIX.sub("", project)
        groups[project].append(i)
    return groups


def get_chunk_preview(engine: SearchEngine, chunk_idx: int, chars: int = 300) -> str:
    m = engine.metadata[chunk_idx]
    p_idx, offset = m
    p = engine.path_table[p_idx]
    try:
        full_path = Path(p["base_dir"]) / p["rel_path"]
        text = full_path.read_text(encoding="utf-8", errors="ignore")
        return " ".join(text[offset:offset + chars].split())
    except Exception:
        return ""


def get_chunk_path(engine: SearchEngine, chunk_idx: int) -> str:
    m = engine.metadata[chunk_idx]
    p_idx, offset = m
    p = engine.path_table[p_idx]
    return p["rel_path"]


def load_engine(index_dir: str = INDEX_DIR, model_path: str = MODEL_PATH, status_cb=None) -> SearchEngine:
    """The actual "intake" -- load the index and block until ready. One
    place, so a fix here (or a future change to how loading works) doesn't
    need to be copy-pasted into every analysis script."""
    engine = SearchEngine()
    engine.load(index_dir, model_path, status_cb or (lambda m: None))
    while not engine.ready:
        time.sleep(0.2)
    return engine


def load_pipeline_inputs(min_chunks: int = MIN_CHUNKS_PER_PROJECT, status_cb=None,
                          index_dir: str = INDEX_DIR, model_path: str = MODEL_PATH):
    """Full intake + compression stage in one call: load the engine,
    group chunks into projects, filter noise-sized groups, split out
    real projects vs known non-project software, and resolve base_dirs.
    Returns (engine, groups, real_projects, base_dirs) -- every downstream
    analysis stage consumes this same tuple instead of re-deriving it."""
    engine = load_engine(index_dir, model_path, status_cb)
    groups = group_chunks_by_project(engine)
    groups = {k: v for k, v in groups.items() if len(v) >= min_chunks}
    real_projects = [n for n in groups if n.lower() not in NON_PROJECT_HINTS]
    base_dirs = sorted({p["base_dir"] for p in engine.path_table})
    return engine, groups, real_projects, base_dirs
