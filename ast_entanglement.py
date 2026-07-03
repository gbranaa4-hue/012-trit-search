"""
AST-level structural entanglement -- Python files only.

The main entanglement pipeline (trit_entanglement.py) finds relationships
via embedding similarity on raw text chunks. That's shallow: it can't tell
you WHY two files are related (shared algorithm? copy-pasted function?
convergent implementation?), and it's language-agnostic in a way that
actually means "compares surface text," which two files in different
languages implementing the same idea won't share (see the cross-language
ground-truth gap found while calibrating likely_genuine).

This is a genuinely different, structural signal: for every function in
every Python file across the indexed projects, build a normalized
"shape" fingerprint -- the sequence of control-flow node types in its
body, with all identifiers/literals stripped out. Two functions with the
same shape have the same control-flow skeleton regardless of variable
names, meaning "this looks like a renamed/refactored copy of that,"
which raw embedding similarity cannot claim.

Scope, stated plainly: Python only, via the stdlib `ast` module (zero new
dependencies). Most of the indexed projects here are GDScript, not Python
-- covering those would need tree-sitter with a GDScript grammar, which is
real, separate work, not attempted in this pass. This is a first real
slice of the idea, not the whole idea.

Usage:
    python ast_entanglement.py [min_shape_len]
"""
import ast
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trit_app import SearchEngine
from observe_pipeline import get_chunk_path, load_pipeline_inputs


def _shape_of(node: ast.AST, depth: int = 0, max_depth: int = 6) -> tuple:
    """Structural fingerprint of a function body: the sequence of
    control-flow-relevant node types, recursively, with identifiers,
    literals, and call targets all stripped -- only shape survives."""
    if depth > max_depth:
        return ("...",)
    shape = []
    for child in ast.iter_child_nodes(node):
        kind = type(child).__name__
        if kind in ("Name", "Constant", "Attribute", "Load", "Store", "arg", "arguments"):
            continue  # identifiers/literals -- not structural
        if kind in ("If", "For", "While", "Try", "With", "Return", "Call",
                    "Assign", "AugAssign", "FunctionDef", "AsyncFunctionDef",
                    "ClassDef", "Break", "Continue", "Raise", "Yield", "Await",
                    "BoolOp", "Compare", "BinOp", "ListComp", "DictComp"):
            shape.append(kind)
            shape.extend(_shape_of(child, depth + 1, max_depth))
    return tuple(shape)


def extract_function_shapes(source: str, path: str) -> list:
    """Returns [(function_name, shape_tuple, lineno), ...] for every
    top-level and nested function in this file. Skips files that don't
    parse (not an error -- just not includable in this signal)."""
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            shape = _shape_of(node)
            out.append((node.name, shape, node.lineno))
    return out


def find_ast_matches(engine, groups: dict, real_projects: list, base_dirs: list,
                      min_shape_len: int = 6, on_status=print) -> dict:
    """Core "combustion stage" logic, extracted from main() so an
    orchestrator (run_full_pipeline.py) can call this directly against an
    already-loaded engine/groups/real_projects/base_dirs from a single
    shared intake. Returns the same dict main() saves to
    ast_entanglement_results.json."""
    # project -> [(func_name, shape, path, lineno), ...]
    project_shapes = defaultdict(list)
    seen_files = set()

    for proj in real_projects:
        for idx in groups[proj]:
            rel_path = get_chunk_path(engine, idx)
            if not rel_path.lower().endswith(".py"):
                continue
            key = (proj, rel_path)
            if key in seen_files:
                continue
            seen_files.add(key)

            full = None
            for b in base_dirs:
                cand = Path(b) / rel_path
                if cand.exists():
                    full = cand
                    break
            if full is None:
                continue
            try:
                source = full.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for fname, shape, lineno in extract_function_shapes(source, rel_path):
                if len(shape) >= min_shape_len:
                    project_shapes[proj].append((fname, shape, rel_path, lineno))

    total_funcs = sum(len(v) for v in project_shapes.values())
    on_status(f"{total_funcs} functions with shape length >= {min_shape_len}, across {len(project_shapes)} projects with .py files\n")

    # Cross-project matches: same project pair, exact shape match (or very
    # close -- exact match first, since that's the strong, defensible case)
    matches = []
    projects_with_py = sorted(project_shapes.keys())
    for i, pa in enumerate(projects_with_py):
        for pb in projects_with_py[i + 1:]:
            shapes_a = project_shapes[pa]
            shapes_b = project_shapes[pb]
            for fname_a, shape_a, path_a, line_a in shapes_a:
                for fname_b, shape_b, path_b, line_b in shapes_b:
                    if shape_a == shape_b:
                        matches.append({
                            "project_a": pa, "project_b": pb,
                            "func_a": fname_a, "path_a": path_a, "line_a": line_a,
                            "func_b": fname_b, "path_b": path_b, "line_b": line_b,
                            "shape_len": len(shape_a),
                        })

    matches.sort(key=lambda m: -m["shape_len"])
    on_status(f"{len(matches)} exact structural-shape matches across project pairs\n")
    on_status("=" * 70)
    on_status("  TOP STRUCTURAL MATCHES (identical control-flow skeleton)")
    on_status("=" * 70)
    for m in matches[:20]:
        on_status(f"\n[shape_len={m['shape_len']}] {m['project_a']}:{m['path_a']}:{m['func_a']}() (line {m['line_a']})")
        on_status(f"                <-> {m['project_b']}:{m['path_b']}:{m['func_b']}() (line {m['line_b']})")

    return {"matches": matches, "total_functions": total_funcs}


def main():
    min_shape_len = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    print("Loading OBSERVE index...")
    engine, groups, real_projects, base_dirs = load_pipeline_inputs(
        status_cb=lambda m: print(f"  {m}")
    )
    print(f"{len(real_projects)} real projects\n")

    result = find_ast_matches(engine, groups, real_projects, base_dirs,
                               min_shape_len=min_shape_len, on_status=print)

    out_path = Path(__file__).resolve().parent / "ast_entanglement_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
