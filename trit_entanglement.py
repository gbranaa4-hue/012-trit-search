"""
Code Entanglement Database — an OBSERVE extension.

Scans everything already in OBSERVE's index, groups chunks into distinct
projects (the existing index mixes real codebases with software configs
and stray folders — this reports what it finds honestly, doesn't silently
assume everything is "a project"), summarizes each with a local model
(what it is / what it does / what it's for), and computes pairwise
cross-project semantic overlap using the SAME embeddings already built —
backed by actual matching chunk evidence, not just a bare similarity
number.

Usage:
    python trit_entanglement.py                 Build the full database
    python trit_entanglement.py --list           Just list detected projects
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trit_app import SearchEngine

INDEX_DIR = str(Path.home() / ".trit-search" / "index")
MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "code-minilm")
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "all-MiniLM-L6-v2"

OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

OUTPUT_DB = str(Path(__file__).resolve().parent / "code_entanglement_db.json")

# Known "container" path prefixes to strip before taking the next folder
# as the project name — otherwise everything under Documents groups as
# one giant "project."
CONTAINER_PREFIXES = [
    "Users/gbran/OneDrive/Documents/",
    "Users/gbran/OneDrive/Desktop/",
    "Users/gbran/OneDrive/",   # catch-all for other OneDrive subfolders — must come AFTER the more specific ones above
    "Users/gbran/Downloads/",
    "Users/gbran/",
]

# Code file extensions — if the "first path component" after stripping
# containers ends in one of these, the file was indexed directly under a
# folder with no further subfolder (a loose script, not a real project).
# Measured real case: "NTeleportation.cs" (579 chunks) was a single large
# file misgrouped as if it were a project folder.
_CODE_EXTENSIONS = {
    ".py", ".gd", ".js", ".ts", ".cs", ".c", ".cpp", ".h", ".hpp",
    ".java", ".lua", ".rb", ".php", ".md", ".sh", ".ps1",
}

# Folders that are clearly not "a codebase" in any meaningful sense —
# reported separately, not silently dropped, so this tool is honest about
# what's actually in the index rather than pretending it's all curated.
NON_PROJECT_HINTS = {
    "image-line", "ableton", "native instruments", "universal audio",
    "fabfilter", "blue cat audio", "xfer", "vital", "tone2", "oeksound",
    "naughty seal audio", "zoom", "max 8", "my cheat tables",
    "call of duty", "call of duty modern warfare", "overwatch",
    "starcraft ii", "stronghold kingdoms", "addictive keys logs",
}

MIN_CHUNKS_PER_PROJECT = 8   # ignore noise — a handful of stray chunks isn't "a project"


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 90) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip()


def _infer_project_name(base_dir: str, rel_path: str) -> str:
    full = (base_dir.rstrip("/\\") + "/" + rel_path.replace("\\", "/")).replace("//", "/")
    full = full.lstrip("C:/").lstrip("/")
    for prefix in CONTAINER_PREFIXES:
        if full.startswith(prefix):
            full = full[len(prefix):]
            break
    parts = full.split("/")
    first = parts[0] if parts else "unknown"
    # A "project name" that's actually a bare filename (e.g. "NTeleportation.cs")
    # means this file was indexed directly under a container with no real
    # project subfolder — group these under one explicit bucket instead of
    # each pretending to be its own separate "project."
    if Path(first).suffix.lower() in _CODE_EXTENSIONS:
        return "(loose scripts, no project folder)"
    return first


_DUPLICATE_SUFFIX = re.compile(r"\s*\(\d+\)$")   # "Foo(1)", "Foo (2)" -> "Foo"

def group_chunks_by_project(engine: SearchEngine, merge_duplicate_suffixes: bool = True):
    """
    Returns {project_name: [chunk_indices]} using the loaded index's
    metadata directly — no new search calls needed.

    merge_duplicate_suffixes: if True, folders differing only by a
    trailing "(1)"/"(2)" (the pattern Windows/browsers add when the same
    folder gets downloaded/extracted twice) are merged into one project.
    Measured real case: "QuasicrystalMEMS_Paper_Branaa" and
    "QuasicrystalMEMS_Paper_Branaa(1)" both appeared as separate
    "projects" at 206 chunks each — almost certainly the same folder
    duplicated, not two genuinely distinct codebases. Merging is a
    reasonable default but not guaranteed correct — a "(1)" folder COULD
    legitimately be a different, later revision with real differences;
    this just reports it as one project rather than silently treating
    duplicates as separate entanglement-worthy relationships.
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


SUMMARY_VERIFY_MODELS = ("qwen2.5-coder:7b", "deepseek-r1:7b", "llama3.2:latest")

def _verify_summary_claims(samples_text: str, summary: str, models=SUMMARY_VERIFY_MODELS):
    """
    Consensus check: does the generated summary make any specific claims
    (genre, technology, purpose) NOT actually supported by the sampled
    snippets? Measured real case this was built for: a summary of `tribe`
    (an NPC trust/loyalty simulation — dogs, camps, territory, spiking
    neurons) confidently claimed it was "likely a first-person shooter (FPS)
    style game" — a genuine hallucination with zero support in the actual
    sampled content. Same consensus-vote principle validated for
    propose_change: poll multiple independent models, since any single
    model (including the one that wrote the summary) can be wrong in ways
    a second, differently-trained model catches.

    Returns a list of unsupported-claim strings (empty if none found by
    majority vote), or None if no model could be reached.
    """
    prompt = (
        f"Source snippets (real, sampled from the project):\n{samples_text}\n\n"
        f"Generated summary of this project:\n{summary}\n\n"
        f"Does the summary state any SPECIFIC claim (a genre, a technology, "
        f"a stated purpose) that is NOT actually supported by the source "
        f"snippets above? List each unsupported claim on its own line, or "
        f"respond with exactly the word NONE if the summary is fully "
        f"grounded in the snippets shown."
    )
    votes = []
    for m in models:
        try:
            raw = _call_ollama(prompt, model=m, timeout=90)
            votes.append(raw)
        except Exception:
            pass
    if not votes:
        return None
    # Majority: if 2+ of the models both said something other than "NONE"
    # (allowing for minor formatting), treat this as a confirmed issue —
    # same majority-vote principle as propose_change's semantic-role check.
    #
    # Real bug found and fixed: originally only checked the first 20
    # characters for "NONE", which missed cases where a model explained
    # its reasoning BEFORE stating the final verdict (measured real case:
    # RUSTERVER's summary got wrongly flagged because one model's response
    # was "The summary does not contain any specific claims... NONE" -
    # "NONE" appeared at the end, past the 20-char cutoff, so a genuinely
    # clean verdict was miscounted as a flagged one). Search the whole
    # response for "NONE" as a distinct word instead of a text-prefix.
    flagged = [v for v in votes if not re.search(r"\bNONE\b", v.upper())]
    if len(flagged) > len(votes) / 2:
        return flagged
    return []


def summarize_project(engine: SearchEngine, project: str, chunk_indices: list, sample_n: int = 8) -> dict:
    """Sample representative chunks, ask a local model to summarize what
    this project is/does/is for, then verify the summary's specific claims
    against the actual sampled content via consensus check (not trusting
    the summarizing model's own self-report — see the real "tribe = FPS"
    hallucination this verification step was built to catch)."""
    rng = np.random.default_rng(hash(project) % (2**31))
    sample_idx = rng.choice(chunk_indices, size=min(sample_n, len(chunk_indices)), replace=False)

    samples = []
    for idx in sample_idx:
        path = get_chunk_path(engine, idx)
        preview = get_chunk_preview(engine, idx, chars=250)
        if preview:
            samples.append(f"{path}: {preview}")

    if not samples:
        return {"summary": "(no readable content sampled)", "unsupported_claims": []}

    samples_text = "\n\n".join(samples)
    prompt = (
        f"Here are {len(samples)} code snippets sampled from a project called \"{project}\":\n\n"
        + samples_text
        + "\n\nBased ONLY on these real snippets, answer in 3 short sentences: "
        "(1) what kind of project is this, (2) what does it appear to do, "
        "(3) what is it likely for. Do not invent details not suggested by the snippets."
    )
    try:
        summary = _call_ollama(prompt)
    except Exception as e:
        return {"summary": f"(summarization failed: {e})", "unsupported_claims": []}

    unsupported = _verify_summary_claims(samples_text, summary)
    return {"summary": summary, "unsupported_claims": unsupported or []}


def _filename_similarity(path_a: str, path_b: str) -> float:
    """
    Independent second signal, uncorrelated with content embeddings:
    do these two files share naming patterns? Real motivating case:
    spikeling_verilog.py <-> spikeling.py (genuine duplicate, both share
    "spikeling") scored only slightly higher on content alone (0.708)
    than miniaudio.h <-> libraries~b28b7af69.js (coincidental dense-text
    similarity, 0.645) — too close to trust content similarity alone.
    Filename similarity should cleanly separate these: genuine relations
    overwhelmingly share naming; coincidental technical-text matches
    almost never do.
    """
    import difflib
    name_a = Path(path_a).stem.lower()
    name_b = Path(path_b).stem.lower()
    # Strip common noise (version numbers, hash suffixes) so "file_v2"
    # and "file" still match reasonably
    name_a = re.sub(r"[\d_\-~]+$", "", name_a)
    name_b = re.sub(r"[\d_\-~]+$", "", name_b)
    return difflib.SequenceMatcher(None, name_a, name_b).ratio()


def compute_entanglement(engine: SearchEngine, groups: dict, project_a: str, project_b: str,
                          top_k: int = 3, sample_cap: int = 300, candidate_pool: int = 60):
    """
    Cross-project semantic overlap, with a SECOND independent signal
    (filename similarity) used to separate genuine relationships from
    coincidental dense-technical-text matches that content embeddings
    alone cannot reliably distinguish (measured real case: a C audio
    header and minified JavaScript scored only ~0.06 below a genuine
    duplicate file, using content similarity alone — too close to trust).

    Method: take the top `candidate_pool` matches by content similarity,
    then re-rank that pool by content_score + filename_similarity_bonus,
    surfacing the top_k results from the RE-RANKED list. Both scores are
    reported separately in the evidence — content-only high-scorers that
    don't also share naming patterns will still appear in output, just
    ranked lower and clearly labeled, not hidden.
    """
    idx_a = groups[project_a]
    idx_b = groups[project_b]
    if len(idx_a) > sample_cap:
        idx_a = list(np.random.default_rng(0).choice(idx_a, sample_cap, replace=False))
    if len(idx_b) > sample_cap:
        idx_b = list(np.random.default_rng(1).choice(idx_b, sample_cap, replace=False))

    vecs_a = engine.index[idx_a]     # (Na, dim) — RAW ternary vectors, entries in {-1,0,+1}, NOT unit-normalized
    vecs_b = engine.index[idx_b]     # (Nb, dim)
    # Real bug found and fixed: engine.index stores raw ternary vectors,
    # not unit-normalized ones (unlike the query vector search() compares
    # against, which IS normalized). Comparing two raw ternary vectors
    # directly gives dot products up to ~sqrt(dim), not bounded [-1, 1]
    # cosine similarity — measured real scores in the 90-160 range, wildly
    # outside the 0-10 range every other search in this project uses.
    # Normalize both sides here to get a genuine, comparable score.
    norm_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
    vecs_a = vecs_a / np.clip(norm_a, 1e-8, None)
    vecs_b = vecs_b / np.clip(norm_b, 1e-8, None)
    sims = vecs_a @ vecs_b.T          # (Na, Nb) — now genuine cosine similarity, range [-1, 1]

    best_per_a = sims.max(axis=1)
    avg_score = float(best_per_a.mean())

    # Widen the candidate pool beyond top_k, then re-rank with the
    # filename-similarity signal before taking the final top_k.
    flat_pool = np.argsort(-sims, axis=None)[:candidate_pool]
    candidates = []
    for flat_idx in flat_pool:
        ia, ib = np.unravel_index(flat_idx, sims.shape)
        real_ia, real_ib = idx_a[ia], idx_b[ib]
        content_score = float(sims[ia, ib])
        a_path = get_chunk_path(engine, real_ia)
        b_path = get_chunk_path(engine, real_ib)
        name_sim = _filename_similarity(a_path, b_path)
        candidates.append({
            "content_score": content_score,
            "filename_similarity": name_sim,
            "combined_score": content_score + name_sim,   # equal weight — a deliberate, simple first pass
            "a_path": a_path,
            "a_preview": get_chunk_preview(engine, real_ia, 150),
            "b_path": b_path,
            "b_preview": get_chunk_preview(engine, real_ib, 150),
        })

    candidates.sort(key=lambda c: -c["combined_score"])

    # Dedup by (a_path, b_path) — multiple chunks from the same two files
    # matching just re-confirms the same file-level relationship, not a
    # separate piece of evidence. Keep only the single best-scoring chunk
    # per unique file pair, so top_k evidence entries are diverse.
    seen_pairs = set()
    evidence = []
    for c in candidates:
        pair_key = (c["a_path"], c["b_path"])
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        evidence.append(c)
        if len(evidence) >= top_k:
            break

    for e in evidence:
        e["likely_genuine"] = e["filename_similarity"] > 0.3   # rough, stated threshold — not a hard guarantee
        e["score"] = e["content_score"]   # keep back-compat key for existing callers/db format

    return avg_score, evidence


def main():
    print("Loading OBSERVE index...")
    engine = SearchEngine()
    done = {"flag": False}
    engine.load(INDEX_DIR, MODEL_PATH, lambda msg: print(f"  {msg}"))
    while not engine.ready:
        time.sleep(0.5)
    print(f"Index ready — {len(engine.metadata):,} chunks\n")

    print("Grouping chunks into projects...")
    groups = group_chunks_by_project(engine)
    groups = {k: v for k, v in groups.items() if len(v) >= MIN_CHUNKS_PER_PROJECT}
    print(f"Found {len(groups)} projects with >= {MIN_CHUNKS_PER_PROJECT} chunks\n")

    real_projects = []
    noise_projects = []
    for name, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        is_noise = name.lower() in NON_PROJECT_HINTS
        (noise_projects if is_noise else real_projects).append(name)
        tag = "[non-project config/software]" if is_noise else ""
        print(f"  {name:<35} {len(idxs):>6} chunks  {tag}")

    if "--list" in sys.argv:
        return

    print(f"\n{len(real_projects)} real projects, {len(noise_projects)} flagged as non-project software/config\n")

    db = {"projects": {}, "entanglement": []}

    print("=" * 70)
    print("  PER-PROJECT SUMMARIES")
    print("=" * 70)
    for name in real_projects:
        print(f"\n{name}:")
        result = summarize_project(engine, name, groups[name])
        print(f"  {result['summary']}")
        if result["unsupported_claims"]:
            print(f"  >>> FLAGGED — unsupported claim(s) found by consensus verification:")
            for claim_text in result["unsupported_claims"]:
                print(f"      {claim_text[:200]}")
        db["projects"][name] = {
            "chunk_count": len(groups[name]),
            "summary": result["summary"],
            "unsupported_claims": result["unsupported_claims"],
        }

    print("\n" + "=" * 70)
    print("  CROSS-PROJECT ENTANGLEMENT")
    print("=" * 70)
    for i, a in enumerate(real_projects):
        for b in real_projects[i + 1:]:
            score, evidence = compute_entanglement(engine, groups, a, b)
            print(f"\n{a} <-> {b}: entanglement score = {score:.3f}")
            for e in evidence:
                genuine_tag = "GENUINE" if e.get("likely_genuine") else "coincidental?"
                print(
                    f"  [content={e['score']:.2f} filename_sim={e.get('filename_similarity', 0):.2f} "
                    f"combined={e.get('combined_score', e['score']):.2f}] ({genuine_tag}) "
                    f"{a}:{e['a_path']} <-> {b}:{e['b_path']}"
                )
                print(f"        A: {e['a_preview'][:100]}")
                print(f"        B: {e['b_preview'][:100]}")
            db["entanglement"].append({"a": a, "b": b, "score": score, "evidence": evidence})

    with open(OUTPUT_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print(f"\nSaved: {OUTPUT_DB}")


if __name__ == "__main__":
    main()
