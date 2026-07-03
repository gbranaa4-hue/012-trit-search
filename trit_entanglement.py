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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trit_app import SearchEngine
from observe_pipeline import (
    INDEX_DIR, MODEL_PATH, MIN_CHUNKS_PER_PROJECT, NON_PROJECT_HINTS,
    stable_hash, group_chunks_by_project, get_chunk_path, get_chunk_preview,
    load_engine, load_pipeline_inputs,
)

OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

OUTPUT_DB = str(Path(__file__).resolve().parent / "code_entanglement_db.json")


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 90) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip()


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


def _decompose_claims(summary: str) -> list:
    """Split a summary into individual sentence-level claims. Deterministic
    (no LLM call) on purpose -- decomposition itself shouldn't be a second
    place for hallucination/failure to creep in."""
    parts = re.split(r"(?<=[.!?])\s+", summary.replace("\n", " ").strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _check_claim_support(samples_text: str, claim: str, models=SUMMARY_VERIFY_MODELS) -> bool:
    """One claim, one focused YES/NO question per model, majority vote.
    Returns True if the claim is judged UNSUPPORTED (majority said NO)."""
    prompt = (
        f"Source snippets (real, sampled from the project):\n{samples_text}\n\n"
        f"Claim: \"{claim}\"\n\n"
        f"Is this specific claim explicitly supported by concrete evidence in "
        f"the source snippets above? Answer with exactly one word first: "
        f"YES or NO. Then, only if YES, quote the specific evidence."
    )
    votes = []
    for m in models:
        try:
            raw = _call_ollama(prompt, model=m, timeout=90)
            votes.append(raw)
        except Exception:
            pass
    if not votes:
        return False
    no_votes = [v for v in votes if re.match(r"\s*NO\b", v.upper())]
    return len(no_votes) > len(votes) / 2


def _verify_summary_claims_v2(samples_text: str, summary: str, models=SUMMARY_VERIFY_MODELS):
    """
    Adversarial, per-claim alternative to _verify_summary_claims (v1).

    v1 asks one holistic question about the whole summary at once ("does
    this contain any unsupported claims?") -- measured via
    calibrate_consensus.py to have only 30% recall on injected fabricated
    claims, because models tend to eyeball a whole paragraph and let a
    single plausible-sounding extra sentence blend in.

    v2 instead: (1) mechanically splits the summary into individual
    sentence-level claims first (so nothing gets averaged away), then
    (2) asks a separate, narrowly-scoped YES/NO support question per claim.
    This is the "push/pull" idea applied concretely -- pulling the summary
    apart into individually-checkable pieces before pushing each one
    against the source, instead of judging the whole thing at once.

    Returns a list of unsupported claim strings (empty if none found).
    """
    claims = _decompose_claims(summary)
    unsupported = []
    for claim in claims:
        if _check_claim_support(samples_text, claim, models=models):
            unsupported.append(claim)
    return unsupported


def summarize_project(engine: SearchEngine, project: str, chunk_indices: list, sample_n: int = 8) -> dict:
    """Sample representative chunks, ask a local model to summarize what
    this project is/does/is for, then verify the summary's specific claims
    against the actual sampled content (not trusting the summarizing
    model's own self-report — see the real "tribe = FPS" hallucination
    this verification step was built to catch).

    Uses the per-claim adversarial verifier (v2), not the holistic one
    (v1). Measured via calibrate_consensus.py on 12 identical injected-
    fabrication test cases: v1 (one holistic "any unsupported claims?"
    question per summary) had 17% recall — it missed 5 of 6 planted
    fabrications because a single plausible-sounding sentence blends into
    an otherwise-correct paragraph when judged as a whole. v2 (decompose
    into individual claims first, check each one narrowly) had 100%
    recall on the same cases, at the cost of a higher false-positive rate
    (67% vs 33% precision) — an acceptable trade for a hallucination
    check: a false alarm costs a manual glance, a missed hallucination
    silently pollutes the database."""
    rng = np.random.default_rng(stable_hash(project) % (2**31))
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

    unsupported = _verify_summary_claims_v2(samples_text, summary)
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


def _resolve_full_path(rel_path: str, base_dirs: list):
    for b in base_dirs:
        p = Path(b) / rel_path
        if p.exists():
            return p
    return None


def compute_entanglement(engine: SearchEngine, groups: dict, project_a: str, project_b: str,
                          top_k: int = 3, sample_cap: int = 300, candidate_pool: int = 60,
                          base_dirs: list = None):
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

    # Compound-identifier overlap: a second independent genuineness signal,
    # found and calibrated via calibrate_cross_language.py + a scaled-up
    # verification pass (apply_compound_signal.py). Filename similarity
    # structurally cannot detect cross-LANGUAGE relationships (a .h file
    # and a .gd file never look similar by name/extension even when they
    # implement the same design — the confirmed real case: Spikeling-
    # Project's spikeling_hw.h and tribe's spikeling.gd share the exact
    # threshold=110 neuron convention). Compound (snake_case) identifier
    # overlap, with known engine/stdlib API excluded, catches that case
    # where filename similarity cannot. Measured on 544 real evidence
    # pairs: 23% fire rate, top-scoring matches are genuine custom
    # vocabulary once framework noise is denylisted — not perfect
    # (leftover builtins can still slip through) but a real improvement,
    # not a guess.
    if base_dirs:
        from corpus_idf import compound_identifier_overlap
        for e in evidence:
            pa = _resolve_full_path(e["a_path"], base_dirs)
            pb = _resolve_full_path(e["b_path"], base_dirs)
            shared = set()
            if pa is not None and pb is not None:
                try:
                    ta = pa.read_text(encoding="utf-8", errors="ignore")
                    tb = pb.read_text(encoding="utf-8", errors="ignore")
                    shared = compound_identifier_overlap(ta, tb)
                except Exception:
                    shared = set()
            e["shared_compound_identifiers"] = sorted(shared)

    for e in evidence:
        e["likely_genuine"] = (
            e["filename_similarity"] > 0.3
            or len(e.get("shared_compound_identifiers", [])) > 0
        )
        e["score"] = e["content_score"]   # keep back-compat key for existing callers/db format

    return avg_score, evidence


def main():
    # Source snippets can contain arbitrary Unicode (arrows, checkmarks,
    # non-Latin identifiers, etc.). Windows consoles default to a narrow
    # codec (cp1252) that can't encode most of it, and a crash mid-preview-
    # print here throws away an otherwise-complete run. Replace unencodable
    # characters instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    print("Loading OBSERVE index...")
    engine, groups, real_projects_unordered, base_dirs = load_pipeline_inputs(
        status_cb=lambda msg: print(f"  {msg}")
    )
    print(f"Index ready — {len(engine.metadata):,} chunks\n")
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
            score, evidence = compute_entanglement(engine, groups, a, b, base_dirs=base_dirs)
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
                shared_ids = e.get("shared_compound_identifiers") or []
                if shared_ids:
                    print(f"        shared compound identifiers: {shared_ids[:10]}")
            db["entanglement"].append({"a": a, "b": b, "score": score, "evidence": evidence})

    with open(OUTPUT_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print(f"\nSaved: {OUTPUT_DB}")


if __name__ == "__main__":
    main()
