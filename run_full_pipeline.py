"""
Full OBSERVE analysis pipeline, run as one continuous flow instead of
four independent scripts each re-deriving the same intake.

This is the "combustion + exhaust" half of the jet-engine-style
restructuring (observe_pipeline.py was the "intake" half). Before this,
running the full analysis meant invoking trit_entanglement.py,
code_references.py, and ast_entanglement.py as four separate processes,
each independently loading the 58,224-chunk index and grouping projects
from scratch (~8-10s each), then add_directional_evidence.py as a fifth
process that re-read the resulting JSON from disk just to mutate it and
write it back. None of the four stages could see each other's output
in-memory.

This runs the shared intake exactly once, feeds the SAME engine/groups/
real_projects/base_dirs into all three independent analysis techniques
(embedding-similarity entanglement, real code references, AST structural
matching), applies the directional-evidence augmentation directly to the
in-memory entanglement db (no disk round-trip), and saves ONE combined
database with all four results in clearly separated sections -- instead
of four disconnected JSON files with no shared schema.

Usage:
    python run_full_pipeline.py
"""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from observe_pipeline import load_pipeline_inputs, MIN_CHUNKS_PER_PROJECT
from trit_entanglement import build_entanglement_db
from code_references import find_all_references
from ast_entanglement import find_ast_matches
from add_directional_evidence import add_directional_evidence

OUTPUT_PATH = Path(__file__).resolve().parent / "observe_full_database.json"


def main():
    t0 = time.time()

    print("=" * 70)
    print("  INTAKE -- load index, group projects (once, shared by every stage)")
    print("=" * 70)
    engine, groups, real_projects, base_dirs = load_pipeline_inputs(
        status_cb=lambda msg: print(f"  {msg}")
    )
    print(f"Index ready — {len(engine.metadata):,} chunks")
    print(f"{len(groups)} projects with >= {MIN_CHUNKS_PER_PROJECT} chunks, "
          f"{len(real_projects)} real, {len(base_dirs)} base_dirs\n")
    print(f"[{time.time()-t0:.1f}s elapsed]\n")

    print("=" * 70)
    print("  STAGE 1/4 -- embedding-similarity entanglement (summaries + scores)")
    print("=" * 70)
    entanglement_db = build_entanglement_db(engine, groups, real_projects, base_dirs, on_status=print)
    print(f"\n[{time.time()-t0:.1f}s elapsed]\n")

    print("=" * 70)
    print("  STAGE 2/4 -- real code references (imports/loads/includes)")
    print("=" * 70)
    references_result = find_all_references(engine, groups, real_projects, base_dirs, on_status=print)
    print(f"\n[{time.time()-t0:.1f}s elapsed]\n")

    print("=" * 70)
    print("  STAGE 3/4 -- AST structural matches (Python only)")
    print("=" * 70)
    ast_result = find_ast_matches(engine, groups, real_projects, base_dirs, on_status=print)
    print(f"\n[{time.time()-t0:.1f}s elapsed]\n")

    print("=" * 70)
    print("  STAGE 4/4 -- directional evidence (git history, applied in-memory)")
    print("=" * 70)
    add_directional_evidence(entanglement_db, base_dirs, on_status=print)
    print(f"\n[{time.time()-t0:.1f}s elapsed]\n")

    combined = {
        "projects": entanglement_db["projects"],
        "entanglement": entanglement_db["entanglement"],
        "cross_project_references": references_result["cross_project_references"],
        "reference_stats": {
            "within_project_count": references_result["within_project_count"],
            "unresolved_count": references_result["unresolved_count"],
        },
        "ast_structural_matches": ast_result["matches"],
        "ast_total_functions_scanned": ast_result["total_functions"],
    }

    OUTPUT_PATH.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("=" * 70)
    print(f"Saved combined database: {OUTPUT_PATH}")
    print(f"Total time: {time.time()-t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
