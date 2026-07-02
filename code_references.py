"""
Real cross-file/cross-project code REFERENCES -- not similarity scores.

Everything else in this toolchain (compute_entanglement, the compound-
identifier signal, AST shape matching) infers a relationship from how code
LOOKS -- embedding similarity, shared vocabulary, structural shape. None
of that is the code actually speaking to another file. This is different
and stronger: parse real import/load/include statements, resolve them to
an actual file that exists in the index, and report only genuine,
resolvable reference edges. "A imports B" is not an inference, it's a
fact you can point at a specific line for.

The interesting case is a CROSS-PROJECT reference: one project's file
literally importing/loading/including a file that lives inside a
DIFFERENT indexed project. That's rare and, when it resolves, a much
stronger and more specific claim than "these two files are 0.5 similar."

Extraction, by extension (static, regex/ast-based, no LLM):
  .py           -- ast Import / ImportFrom module names
  .gd           -- preload("..."), load("..."), extends "..." res:// paths
  .cs           -- using X.Y; namespace references (weaker signal --
                    namespaces don't map 1:1 to files, reported separately)
  .js/.ts       -- require('...'), import ... from '...'
  .c/.h/.cpp    -- #include "..." / #include <...>

Usage:
    python code_references.py
"""
import ast
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trit_app import SearchEngine
from trit_entanglement import (
    INDEX_DIR, MODEL_PATH, MIN_CHUNKS_PER_PROJECT,
    group_chunks_by_project, get_chunk_path, NON_PROJECT_HINTS,
)

_RE_GD_LOAD = re.compile(r'(?:preload|load)\(\s*"([^"]+)"\s*\)')
_RE_GD_EXTENDS = re.compile(r'^\s*extends\s+"([^"]+)"', re.MULTILINE)
_RE_JS_REQUIRE = re.compile(r'require\(\s*[\'"]([^\'"]+)[\'"]\s*\)')
_RE_JS_IMPORT = re.compile(r'''import\s+.*?\s+from\s+['"]([^'"]+)['"]''')
_RE_C_INCLUDE = re.compile(r'#include\s*[<"]([^>"]+)[>"]')
_RE_CS_USING = re.compile(r'^\s*using\s+([\w.]+)\s*;', re.MULTILINE)


def extract_references(path: str, text: str) -> dict:
    """Returns {'resolvable': [...raw ref strings...], 'namespaces': [...]}
    -- resolvable ones are attempted against the real file index;
    namespaces (C# using) are reported separately since they don't map
    1:1 onto a single file."""
    ext = Path(path).suffix.lower()
    resolvable = []
    namespaces = []

    if ext == ".py":
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        resolvable.append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    resolvable.append(node.module)
        except Exception:
            pass
    elif ext == ".gd":
        resolvable += _RE_GD_LOAD.findall(text)
        resolvable += _RE_GD_EXTENDS.findall(text)
    elif ext in (".js", ".ts"):
        resolvable += _RE_JS_REQUIRE.findall(text)
        resolvable += _RE_JS_IMPORT.findall(text)
    elif ext in (".c", ".h", ".cpp", ".hpp"):
        resolvable += _RE_C_INCLUDE.findall(text)
    elif ext == ".cs":
        namespaces += _RE_CS_USING.findall(text)

    return {"resolvable": resolvable, "namespaces": namespaces}


def _basename_no_ext(p: str) -> str:
    return Path(p.replace("\\", "/")).stem.lower()


def main():
    print("Loading OBSERVE index...")
    engine = SearchEngine()
    engine.load(INDEX_DIR, MODEL_PATH, lambda m: print(f"  {m}"))
    while not engine.ready:
        time.sleep(0.2)

    groups = group_chunks_by_project(engine)
    groups = {k: v for k, v in groups.items() if len(v) >= MIN_CHUNKS_PER_PROJECT}
    real_projects = [n for n in groups if n.lower() not in NON_PROJECT_HINTS]
    base_dirs = sorted({p["base_dir"] for p in engine.path_table})

    # file basename (no ext) -> [(project, full_rel_path), ...] -- used to
    # resolve a bare import/load target ("Spikeling", "Sky3D") to a real
    # indexed file. Ambiguous basenames (many projects have a file called
    # "player" or "utils") are reported honestly as ambiguous, not guessed.
    basename_index = defaultdict(list)
    # project -> set of normalized (lowercase, forward-slash) rel_paths --
    # used to check "does this res://... path match a file that ACTUALLY
    # exists within the source's own project" before ever considering a
    # cross-project match. Real bug found and fixed: the first version of
    # this resolver matched purely by basename across the WHOLE corpus,
    # so a within-project reference to
    # res://assets/weapons/resources/Player/zombie.gd (a real file that
    # genuinely exists inside horde-beta-version-1 itself) got misattributed
    # as a "cross-project reference" to all_scripts/zombie.gd, a completely
    # different file that just happens to share the same basename. Verified
    # directly: horde-beta-version-1/assets/weapons/resources/Player/
    # zombie.gd exists on disk. Same-project resolution must be checked
    # FIRST, using the fuller path (not just the basename), before a
    # cross-project claim is allowed.
    project_normalized_paths = defaultdict(set)
    project_of_file = {}
    seen = set()
    for proj in real_projects:
        for idx in groups[proj]:
            rel = get_chunk_path(engine, idx)
            key = (proj, rel)
            if key in seen:
                continue
            seen.add(key)
            basename_index[_basename_no_ext(rel)].append((proj, rel))
            project_normalized_paths[proj].add(rel.replace("\\", "/").lower())
            project_of_file[(proj, rel)] = proj

    print(f"{len(seen)} unique files across {len(real_projects)} real projects\n")

    def resolves_within_project(raw_ref: str, proj: str) -> bool:
        """True if raw_ref (a res://... or relative path) matches a real
        file's path SUFFIX within the source's own project -- checked
        before any cross-project claim is allowed."""
        norm_ref = raw_ref.replace("res://", "").replace("\\", "/").lower().lstrip("/")
        if not norm_ref or "/" not in norm_ref:
            return False  # bare basename refs can't be suffix-checked reliably
        for p in project_normalized_paths[proj]:
            if p.endswith(norm_ref) or norm_ref.endswith(p):
                return True
        return False

    def resolves_same_directory(raw_ref: str, source_file: Path) -> bool:
        """Python-specific: `import X` / `from X import Y` with a bare
        module name resolves, for a standalone script run directly (not a
        real installed package), against a sibling file in the SAME
        DIRECTORY on disk -- this is the normal case for the loose research
        scripts in this corpus. Checked directly against the filesystem,
        not the OBSERVE index, because the real false-positive case found
        (HealthComponent.gd) wasn't even indexed despite genuinely
        existing on disk -- index-based checking alone isn't enough."""
        candidate = source_file.parent / f"{raw_ref}.py"
        return candidate.exists()

    cross_project_refs = []
    within_project_count = 0
    unresolved_count = 0
    namespace_refs = defaultdict(set)  # project -> set of using-namespaces (not file-resolved)

    for proj in real_projects:
        for idx in groups[proj]:
            rel = get_chunk_path(engine, idx)
            full = None
            for b in base_dirs:
                cand = Path(b) / rel
                if cand.exists():
                    full = cand
                    break
            if full is None:
                continue
            try:
                text = full.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            refs = extract_references(rel, text)
            for ns in refs["namespaces"]:
                namespace_refs[proj].add(ns)

            for raw_ref in refs["resolvable"]:
                # Same-project resolution checked FIRST, using the fuller
                # path -- see resolves_within_project's docstring for the
                # real false-positive this prevents.
                if resolves_within_project(raw_ref, proj):
                    within_project_count += 1
                    continue
                if rel.lower().endswith(".py") and resolves_same_directory(raw_ref, full):
                    within_project_count += 1
                    continue

                target_basename = _basename_no_ext(raw_ref)
                candidates = basename_index.get(target_basename, [])
                # Exclude self-references and same-file matches
                candidates = [(p, r) for p, r in candidates if not (p == proj and r == rel)]
                if not candidates:
                    unresolved_count += 1
                    continue

                # Real bug found and fixed via manual GUI verification: a
                # module can exist ANYWHERE within the source's own
                # project, not just its exact same directory (the
                # resolves_same_directory check above only catches the
                # single-folder case). Confirmed real case: acoustic-
                # vortex-sim scripts under reservoir_computing/ and
                # filter_bank_concept/ import "fem_plate_bending_
                # homogenized", and acoustic-vortex-sim genuinely has TWO
                # of its own copies of that file elsewhere in its own
                # tree (plate_bending_review/ and QuasicrystalMEMS_Paper_
                # Branaa/plate_bending_review/) -- neither in the same
                # directory as the importing script, so both were missed
                # and this got wrongly reported as ambiguous cross-project.
                # If ANY candidate is in the source's own project, prefer
                # that over any cross-project guess, regardless of which
                # subdirectory it's in.
                same_project_elsewhere = [(p, r) for p, r in candidates if p == proj]
                if same_project_elsewhere:
                    within_project_count += 1
                    continue

                cross = [(p, r) for p, r in candidates if p != proj]
                if cross:
                    for target_proj, target_path in cross:
                        cross_project_refs.append({
                            "source_project": proj, "source_path": rel,
                            "target_project": target_proj, "target_path": target_path,
                            "raw_reference": raw_ref,
                            "ambiguous": len(cross) > 1,
                        })
                else:
                    within_project_count += 1

    # Dedup by BASENAME, not exact path -- the same physical file can be
    # indexed under multiple base_dir conventions with different
    # source_path strings for the identical file (confirmed real case:
    # health_component.gd appeared as "health_component.gd",
    # "horde-beta-version-1\health_component.gd", and a third full-path
    # variant, all for the same underlying reference). Deduping on the
    # exact path string alone missed this.
    seen_edges = set()
    deduped = []
    for r in cross_project_refs:
        key = (r["source_project"], _basename_no_ext(r["source_path"]),
               r["target_project"], _basename_no_ext(r["target_path"]))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        deduped.append(r)
    cross_project_refs = deduped

    print(f"{within_project_count} within-project references resolved")
    print(f"{unresolved_count} references didn't resolve to any indexed file (external libs, stdlib, etc.)")
    print(f"{len(cross_project_refs)} CROSS-PROJECT references found (deduplicated)\n")

    print("=" * 70)
    print("  CROSS-PROJECT REFERENCES (code literally importing/loading")
    print("  a file from a DIFFERENT indexed project)")
    print("=" * 70)
    for r in cross_project_refs:
        amb = " [AMBIGUOUS -- multiple files share this name]" if r["ambiguous"] else ""
        print(f"\n{r['source_project']}:{r['source_path']}")
        print(f"  -> imports/loads \"{r['raw_reference']}\"")
        print(f"  -> resolves to {r['target_project']}:{r['target_path']}{amb}")

    if namespace_refs:
        print("\n" + "=" * 70)
        print("  C# NAMESPACE REFERENCES (weaker signal -- not file-resolved)")
        print("=" * 70)
        for proj, namespaces in namespace_refs.items():
            non_system = sorted(n for n in namespaces if not n.startswith(("System", "Unity", "UnityEngine")))
            if non_system:
                print(f"\n{proj}: {non_system[:15]}")

    out_path = Path(__file__).resolve().parent / "code_references_results.json"
    out_path.write_text(json.dumps({
        "cross_project_references": cross_project_refs,
        "within_project_count": within_project_count,
        "unresolved_count": unresolved_count,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
