"""
Investigates the cross-language ground-truth gap found while calibrating
`likely_genuine` (calibrate_genuine.py). That harness used whole-file
difflib text similarity as ground truth, which is structurally blind to
genuine cross-language conceptual duplicates: a C header and a GDScript
file implementing the same design (e.g. Spikeling-Project's
parallel-audio/include/spikeling_hw.h and tribe's spikeling.gd, confirmed
manually to share the exact "threshold=110" neuron convention) will never
share raw text, so difflib scored them near 0.0 and miscounted a real
relationship as a heuristic false positive.

This tests a different, language-agnostic signal instead: distinctive
LITERAL token overlap. Extract numbers with >=2 digits, identifiers of
length >= 5, and quoted strings from each file's full text -- filtering
out common short/generic tokens -- and measure Jaccard overlap. A shared
domain-specific constant or naming convention (110, N5325, "threshold",
"spikeling") surviving across a language boundary is real, checkable
evidence a human would find convincing, independent of syntax.

This is NOT a full solution -- it's a probe to see whether the signal is
viable at all, checked against the one confirmed-real cross-language case
we have (spikeling_hw.h <-> Spikeling.gd/spikeling.gd) plus a few
confirmed-coincidental pairs from the same false-positive list, to see
if it actually separates them.

Usage:
    python calibrate_cross_language.py
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

# Confirmed-genuine cross-language pairs (manually verified earlier: both
# sides implement the Spikeling neuron model, sharing threshold=110 and
# neuron-ID naming conventions, despite being a .h file and a .gd file).
CONFIRMED_GENUINE = [
    (r"C:\Users\gbran\OneDrive\Documents\Spikeling-Project\parallel-audio\include\spikeling_hw.h",
     r"C:\Users\gbran\Downloads\tribe godot drop in\Spikeling.gd"),
    (r"C:\Users\gbran\OneDrive\Documents\Spikeling-Project\sdk-verilog\spikeling_hw.h",
     r"C:\Users\gbran\OneDrive\Documents\tribe\spikeling.gd"),
]

# Confirmed-coincidental pairs from calibrate_genuine.py's false-positive
# list where content_sim was near-zero AND there's no known real
# relationship (miniaudio.h is a huge generic audio library -- matching
# unrelated files by chance is expected, not a hidden real connection).
CONFIRMED_COINCIDENTAL = [
    (r"C:\Users\gbran\OneDrive\Documents\Spikeling-Project\parallel-audio\spikeling_test\miniaudio.h",
     r"C:\Users\gbran\acoustic-vortex-sim\reservoir_computing\reservoir_rung5_saturation.py"),
    (r"C:\Users\gbran\OneDrive\Documents\Spikeling-Project\parallel-audio\include\miniaudio.h",
     r"C:\Users\gbran\OneDrive\Documents\3d-simulation\main.gd"),
    (r"C:\Users\gbran\OneDrive\Desktop\RUSTERVER\RUSTSERVERROOT\oxide\plugins\NTeleportation.cs",
     r"C:\Users\gbran\symmetry-sensor\symmetry_isolation.py"),
]

# Generic tokens that would create false overlap regardless of any real
# relationship -- common keywords, tiny numbers, boilerplate.
_STOPWORDS = {
    "self", "return", "false", "true", "none", "null", "print", "import",
    "class", "public", "private", "static", "const", "float", "int",
    "string", "bool", "void", "function", "def", "var", "let", "extends",
    "export", "value", "index", "count", "error", "result", "data",
}


def distinctive_tokens(text: str) -> set:
    # Numbers below this many digits (loop bounds, small constants like 10,
    # 12, 13) are near-universal noise -- they showed up as "shared" between
    # totally unrelated files in testing and provided zero real signal.
    # Only genuinely distinctive numeric literals (>=3 digits, e.g. a
    # specific threshold=110 value... actually 110 is 3 digits, keep >= 3)
    # survive.
    numbers = {m for m in re.findall(r"\b\d{3,}\b", text)}
    idents = {m.lower() for m in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{4,}\b", text)}
    idents -= _STOPWORDS
    strings = {m.lower() for m in re.findall(r'"([^"]{4,40})"', text)}
    return numbers | idents | strings


def identifier_only_tokens(text: str) -> set:
    """Stricter variant: identifiers/strings only, no numbers at all --
    numbers turned out to be almost pure noise (loop bounds, indices,
    generic constants) even at >=3 digits when tested."""
    idents = {m.lower() for m in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{4,}\b", text)}
    idents -= _STOPWORDS
    strings = {m.lower() for m in re.findall(r'"([^"]{4,40})"', text)}
    return idents | strings


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def idf_weighted_overlap(text_a: str, text_b: str, df: dict, n_files: int):
    """A shared token counts in proportion to how RARE it is across the
    whole corpus, not just its presence. This is the fix for the earlier
    negative result: raw overlap was dominated by common programming
    vocabulary (append, array, between) that a fixed stopword list
    couldn't fully exclude, because it doesn't know what's actually rare
    in THIS corpus specifically -- only IDF (computed from the real
    document-frequency counts in corpus_idf.py) can."""
    from corpus_idf import idf, identifier_only_tokens
    toks_a = identifier_only_tokens(text_a)
    toks_b = identifier_only_tokens(text_b)
    shared = toks_a & toks_b
    if not shared:
        return 0.0, []
    weighted_score = sum(idf(t, df, n_files) for t in shared)
    ranked = sorted(shared, key=lambda t: -idf(t, df, n_files))
    return weighted_score, ranked


def main():
    print("Testing distinctive-literal-overlap signal on known cases:\n")

    from corpus_idf import load_or_build
    df, n_files = load_or_build()
    print(f"\nUsing corpus IDF: {n_files} files, {len(df)} distinct tokens\n")

    for label, pairs in (("CONFIRMED GENUINE (cross-language)", CONFIRMED_GENUINE),
                          ("CONFIRMED COINCIDENTAL", CONFIRMED_COINCIDENTAL)):
        print(f"=== {label} ===")
        for pa, pb in pairs:
            pa, pb = Path(pa), Path(pb)
            if not pa.exists() or not pb.exists():
                print(f"  (missing file, skipping) {pa} <-> {pb}")
                continue
            ta = pa.read_text(encoding="utf-8", errors="ignore")
            tb = pb.read_text(encoding="utf-8", errors="ignore")
            toks_a = distinctive_tokens(ta)
            toks_b = distinctive_tokens(tb)
            shared = toks_a & toks_b
            score = jaccard(toks_a, toks_b)

            ida = identifier_only_tokens(ta)
            idb = identifier_only_tokens(tb)
            shared_id = ida & idb
            score_id = jaccard(ida, idb)

            idf_score, ranked = idf_weighted_overlap(ta, tb, df, n_files)
            mean_idf = idf_score / len(shared_id) if shared_id else 0.0

            # Compound identifiers (snake_case, camelCase-ish underscore
            # constructs) are actual code vocabulary, not incidental
            # English prose that happens to appear in a comment. Restrict
            # to those and see if THAT separates genuine from coincidental,
            # since raw IDF (both sum and mean) did not.
            from corpus_idf import idf as _idf
            compound_shared = {t for t in shared_id if "_" in t}
            compound_idf_sum = sum(_idf(t, df, n_files) for t in compound_shared) if compound_shared else 0.0

            print(f"  jaccard(mixed)={score:.4f} shared={len(shared)}   "
                  f"IDF-weighted={idf_score:.2f}  mean-IDF={mean_idf:.2f}   "
                  f"compound-shared={len(compound_shared)} compound-IDF={compound_idf_sum:.2f}   "
                  f"{pa.name} <-> {pb.name}")
            if compound_shared:
                print(f"    compound identifiers shared: {sorted(compound_shared)}")
        print()


if __name__ == "__main__":
    main()
