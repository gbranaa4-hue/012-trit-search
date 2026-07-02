"""
Applies the compound-identifier-overlap signal (found in
calibrate_cross_language.py) across all 565 real evidence pairs already in
code_entanglement_db.json, instead of the 5 hand-picked examples it was
found on. This is the honest follow-through on that finding's stated
caveat: a signal validated on 5 examples isn't trustworthy until it's been
run against real, larger-scale data and inspected for how often it fires
and whether the pairs it flags actually look genuine.

Does not change likely_genuine or the database itself -- this is a
measurement pass, reporting what the signal WOULD say, so the results can
be inspected before deciding whether to wire it into the pipeline for real.

Usage:
    python apply_compound_signal.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_idf import load_or_build, idf, identifier_only_tokens

DB_PATH = Path(__file__).resolve().parent / "code_entanglement_db.json"
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


def main():
    df, n_files = load_or_build()
    print(f"Using corpus IDF: {n_files} files, {len(df)} tokens\n")

    db = json.loads(DB_PATH.read_text(encoding="utf-8"))
    all_evidence = [(pair["a"], pair["b"], ev) for pair in db["entanglement"] for ev in pair["evidence"]]
    print(f"{len(all_evidence)} total evidence pairs\n")

    results = []
    skipped = 0
    for a_proj, b_proj, ev in all_evidence:
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

        toks_a = identifier_only_tokens(ta)
        toks_b = identifier_only_tokens(tb)
        compound_shared = {t for t in (toks_a & toks_b) if "_" in t}
        compound_idf_sum = sum(idf(t, df, n_files) for t in compound_shared) if compound_shared else 0.0

        results.append({
            "a_path": ev["a_path"], "b_path": ev["b_path"],
            "likely_genuine_existing": ev.get("likely_genuine", False),
            "filename_similarity": ev.get("filename_similarity", 0.0),
            "compound_shared_count": len(compound_shared),
            "compound_shared_tokens": sorted(compound_shared),
            "compound_idf_sum": compound_idf_sum,
        })

    print(f"skipped {skipped} (unresolvable/unreadable)")
    print(f"{len(results)} pairs checked\n")

    fired = [r for r in results if r["compound_shared_count"] > 0]
    print(f"{len(fired)}/{len(results)} pairs have >=1 shared compound identifier ({100*len(fired)/len(results):.1f}%)\n")

    agree_genuine = sum(1 for r in fired if r["likely_genuine_existing"])
    agree_not = sum(1 for r in fired if not r["likely_genuine_existing"])
    print(f"Of those {len(fired)} pairs the compound signal fired on:")
    print(f"  {agree_genuine} already tagged likely_genuine by the filename heuristic (agreement)")
    print(f"  {agree_not} NOT tagged likely_genuine by filename heuristic (signal disagrees / adds new evidence)")

    print(f"\nAll pairs where compound signal fired but filename heuristic said NOT genuine")
    print("(these are candidates the current pipeline is MISSING):")
    for r in sorted(fired, key=lambda r: -r["compound_idf_sum"]):
        if not r["likely_genuine_existing"]:
            print(f"  compound_idf={r['compound_idf_sum']:.1f} shared={r['compound_shared_tokens']}")
            print(f"    {r['a_path']} <-> {r['b_path']}")

    out_path = Path(__file__).resolve().parent / "compound_signal_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
