"""
Retunes the likely_genuine formula after a real false positive was caught
by actually using the GUI: GatherManager.cs (Rust plugin, excavator
resource logic) <-> trit_mega_train.py / top/gamer_manager.gd both tagged
"genuine" purely because filename_similarity was 0.43 and 0.85
respectively -- shared_compound_identifiers was empty in both cases, and
the actual content (excavator resource dicts vs. a file-scanner script /
an island-generator check) has nothing to do with each other. "Manager"
is just a common enough word that filename similarity alone shouldn't be
trusted at the current 0.3 threshold.

Uses the same ground-truth method as calibrate_genuine.py (whole-file
difflib similarity, independent of the heuristic being tested) against
the CURRENT database (which has content_score AND shared_compound_
identifiers per evidence entry, unlike the earlier calibration run that
predates the compound-identifier integration), and tests several
candidate formulas head-to-head on identical cases before picking one.

Usage:
    python retune_likely_genuine.py [n_pairs]
"""
import difflib
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

DB_PATH = Path(__file__).resolve().parent / "code_entanglement_db.json"
GENUINE_THRESHOLD = 0.5
COINCIDENTAL_THRESHOLD = 0.1

BASE_DIR_CANDIDATES = [
    "C:/Users/gbran/OneDrive/Documents",
    "C:/Users/gbran/OneDrive/Desktop",
    "C:/",
]


def _resolve(rel_path: str):
    for b in BASE_DIR_CANDIDATES:
        p = Path(b) / rel_path
        if p.exists():
            return p
    return None


CANDIDATES = {
    "current (fn_sim>0.3 OR compound)":
        lambda e: e["filename_similarity"] > 0.3 or len(e.get("shared_compound_identifiers", [])) > 0,

    "tightened (fn_sim>0.6 OR compound OR (fn_sim>0.3 AND content>0.5))":
        lambda e: (e["filename_similarity"] > 0.6
                   or len(e.get("shared_compound_identifiers", [])) > 0
                   or (e["filename_similarity"] > 0.3 and e["content_score"] > 0.5)),

    "compound-only (drop filename entirely)":
        lambda e: len(e.get("shared_compound_identifiers", [])) > 0,

    "fn_sim>0.5 OR compound":
        lambda e: e["filename_similarity"] > 0.5 or len(e.get("shared_compound_identifiers", [])) > 0,
}


def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 250

    db = json.loads(DB_PATH.read_text(encoding="utf-8"))
    all_evidence = [ev for pair in db["entanglement"] for ev in pair["evidence"]]
    random.seed(0)
    sample = random.sample(all_evidence, min(n_pairs, len(all_evidence)))
    print(f"Testing against {len(sample)} sampled evidence pairs\n")

    cases = []
    skipped = 0
    for ev in sample:
        pa = _resolve(ev["a_path"])
        pb = _resolve(ev["b_path"])
        if pa is None or pb is None:
            skipped += 1
            continue
        try:
            ta = pa.read_text(encoding="utf-8", errors="ignore")
            tb = pb.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            skipped += 1
            continue
        if not ta or not tb:
            skipped += 1
            continue

        content_sim = difflib.SequenceMatcher(None, ta, tb).quick_ratio()
        if content_sim >= GENUINE_THRESHOLD:
            ground_truth = True
        elif content_sim <= COINCIDENTAL_THRESHOLD:
            ground_truth = False
        else:
            continue

        cases.append((ev, ground_truth))

    print(f"skipped {skipped}, {len(cases)} cases with unambiguous ground truth\n")

    print("=" * 70)
    for name, formula in CANDIDATES.items():
        tp = fn = tn = fp = 0
        for ev, gt in cases:
            pred = formula(ev)
            if gt and pred: tp += 1
            elif gt and not pred: fn += 1
            elif not gt and not pred: tn += 1
            else: fp += 1
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        accuracy = (tp + tn) / len(cases) if cases else float("nan")
        print(f"{name}")
        print(f"  TP={tp} FN={fn} TN={tn} FP={fp}  precision={precision:.2f} recall={recall:.2f} accuracy={accuracy:.2f}")
        print()


if __name__ == "__main__":
    main()
