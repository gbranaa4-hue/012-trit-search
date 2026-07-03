"""
Calibration harness for the `likely_genuine` entanglement tag.

Current heuristic (trit_entanglement.py, compute_entanglement): a match is
tagged likely_genuine when filename_similarity > 0.3. This has never been
measured against real ground truth -- it was only spot-checked on a
handful of cases (duplicate spikeling.py files vs. the RUSTERVER/miniaudio.h
false-positive pair) when it was first built. This harness measures it
properly, the same way calibrate_consensus.py measured the claim-verifier.

Ground truth, independent of the heuristic being tested: full-FILE content
similarity via difflib.SequenceMatcher on the actual file text (not the
embedding, not the filename). A genuine duplicate/near-duplicate file will
have very high whole-file content similarity; a coincidental match (two
unrelated files whose sampled CHUNK happened to embed close together) will
have near-zero whole-file similarity despite matching at the chunk level.
This is a fair, non-circular check because it's a different signal than
filename similarity -- it doesn't just re-test the heuristic against itself.

Usage:
    python calibrate_genuine.py [n_pairs]
"""
import difflib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

DB_PATH = Path(__file__).resolve().parent / "code_entanglement_db.json"

# Ground-truth thresholds on whole-file content similarity. Middle band is
# genuinely ambiguous (could be a partial/refactored duplicate, could be
# unrelated files that happen to share some boilerplate) -- excluded from
# calibration rather than force-labeled, since a made-up label there would
# corrupt the measurement rather than inform it.
GENUINE_THRESHOLD = 0.5
COINCIDENTAL_THRESHOLD = 0.1


def _resolve_full_path(rel_path: str, candidates: list) -> Path:
    """rel_path as stored in evidence doesn't carry its base_dir. Try each
    known base_dir (gathered from the OBSERVE index's path table) until one
    actually exists on disk."""
    for base_dir in candidates:
        p = Path(base_dir) / rel_path
        if p.exists():
            return p
    return None


def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    db = json.loads(DB_PATH.read_text(encoding="utf-8"))
    all_evidence = [(pair["a"], pair["b"], ev) for pair in db["entanglement"] for ev in pair["evidence"]]
    print(f"{len(all_evidence)} total evidence pairs in db, sampling {min(n_pairs, len(all_evidence))}")

    # Gather all known base_dirs by loading OBSERVE's own path table --
    # reuse the same resolution the pipeline itself uses, no guessing.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from observe_pipeline import load_engine
    engine = load_engine()
    base_dirs = sorted({p["base_dir"] for p in engine.path_table})
    print(f"{len(base_dirs)} known base_dirs\n")

    import random
    random.seed(0)
    sample = random.sample(all_evidence, min(n_pairs, len(all_evidence)))

    results = []
    skipped = 0
    for a_proj, b_proj, ev in sample:
        pa = _resolve_full_path(ev["a_path"], base_dirs)
        pb = _resolve_full_path(ev["b_path"], base_dirs)
        if pa is None or pb is None:
            skipped += 1
            continue
        try:
            text_a = pa.read_text(encoding="utf-8", errors="ignore")
            text_b = pb.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            skipped += 1
            continue
        if not text_a or not text_b:
            skipped += 1
            continue

        content_sim = difflib.SequenceMatcher(None, text_a, text_b).quick_ratio()

        if content_sim >= GENUINE_THRESHOLD:
            ground_truth = True
        elif content_sim <= COINCIDENTAL_THRESHOLD:
            ground_truth = False
        else:
            continue  # ambiguous middle band, excluded from calibration

        results.append({
            "a_path": ev["a_path"], "b_path": ev["b_path"],
            "content_file_similarity": content_sim,
            "ground_truth_genuine": ground_truth,
            "heuristic_likely_genuine": ev["likely_genuine"],
            "filename_similarity": ev["filename_similarity"],
        })

    print(f"skipped {skipped} pairs (file unresolvable/unreadable)")
    print(f"{len(results)} pairs with unambiguous ground truth (excluded middle band {COINCIDENTAL_THRESHOLD}-{GENUINE_THRESHOLD})\n")

    tp = sum(1 for r in results if r["ground_truth_genuine"] and r["heuristic_likely_genuine"])
    fn = sum(1 for r in results if r["ground_truth_genuine"] and not r["heuristic_likely_genuine"])
    tn = sum(1 for r in results if not r["ground_truth_genuine"] and not r["heuristic_likely_genuine"])
    fp = sum(1 for r in results if not r["ground_truth_genuine"] and r["heuristic_likely_genuine"])

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    accuracy = (tp + tn) / len(results) if results else float("nan")

    print("=" * 60)
    print("  likely_genuine CALIBRATION RESULT")
    print("=" * 60)
    print(f"TP={tp}  FN={fn}  TN={tn}  FP={fp}")
    print(f"precision={precision:.2f}  recall={recall:.2f}  accuracy={accuracy:.2f}")

    if fn:
        print(f"\nMISSED genuine duplicates (heuristic said coincidental, content says genuine):")
        for r in results:
            if r["ground_truth_genuine"] and not r["heuristic_likely_genuine"]:
                print(f"  content_sim={r['content_file_similarity']:.2f} filename_sim={r['filename_similarity']:.2f}  {r['a_path']} <-> {r['b_path']}")

    if fp:
        print(f"\nFALSE POSITIVES (heuristic said genuine, content says coincidental):")
        for r in results:
            if not r["ground_truth_genuine"] and r["heuristic_likely_genuine"]:
                print(f"  content_sim={r['content_file_similarity']:.2f} filename_sim={r['filename_similarity']:.2f}  {r['a_path']} <-> {r['b_path']}")

    out_path = Path(__file__).resolve().parent / "genuine_calibration_results.json"
    out_path.write_text(json.dumps({
        "results": results, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "precision": precision, "recall": recall, "accuracy": accuracy,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
