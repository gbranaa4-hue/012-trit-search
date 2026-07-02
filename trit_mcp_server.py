"""
012 OBSERVE MCP Server

Exposes OBSERVE's compressed semantic code search as an MCP (Model
Context Protocol) server, so AI coding assistants (Claude Code, Claude
Desktop, or any MCP-compatible client) can call it as a tool — search
your local, fine-tuned, ternary-compressed code index directly from a
conversation, without leaving your editor or copy-pasting results.

This wraps the exact same SearchEngine used by trit_app.py (OBSERVE's
GUI) — same model, same compressed index, same search logic. No GUI is
launched; this runs headless over stdio, the standard local-MCP transport.

Setup:
  pip install mcp
  Build an index first with trit_app.py (click INDEX CODEBASE) or
  trit_search.py --index — this server reads the existing index, it
  does not build one itself.

Usage (standalone test):
  python trit_mcp_server.py

Usage (as an MCP server, e.g. in Claude Code's mcp config):
  {
    "mcpServers": {
      "observe": {
        "command": "python",
        "args": ["C:/path/to/012-ternary/trit_mcp_server.py"]
      }
    }
  }
"""
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trit_app import SearchEngine

from mcp.server.fastmcp import FastMCP

INDEX_DIR  = str(Path.home() / ".trit-search" / "index")
MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "code-minilm")
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "all-MiniLM-L6-v2"  # fall back to baseline if not fine-tuned locally

mcp = FastMCP("observe")
engine = SearchEngine()

import time

_loaded = {"done": False, "error": None, "loading_started": False}
_status = {"msg": "Not started", "ts": 0.0}

def _ensure_loaded():
    if _loaded["done"]:
        return
    if not _loaded["loading_started"]:
        def on_status(msg):
            _status["msg"] = msg
            _status["ts"] = time.monotonic()
            print(f"[OBSERVE] {msg}", file=sys.stderr)
        _status["msg"] = "Starting..."
        _status["ts"] = time.monotonic()
        engine.load(INDEX_DIR, MODEL_PATH, on_status)
        _loaded["loading_started"] = True

    # engine.load() spawns a background daemon thread; block until ready
    # for the synchronous tool call below. If the status hasn't moved in
    # STALL_LIMIT seconds and we're still not ready, treat the load as
    # stuck/dead rather than silently re-waiting forever: clear
    # loading_started so the *next* call kicks off a fresh load thread.
    STALL_LIMIT = 60
    TOTAL_LIMIT = 180
    waited = 0.0
    last_seen_ts = _status["ts"]
    while not engine.ready and waited < TOTAL_LIMIT:
        time.sleep(0.5)
        waited += 0.5
        if _status["ts"] != last_seen_ts:
            last_seen_ts = _status["ts"]
        elif time.monotonic() - last_seen_ts > STALL_LIMIT:
            _loaded["loading_started"] = False
            _loaded["error"] = (
                f"Load appears stalled (no progress for {STALL_LIMIT}s, "
                f"last status: \"{_status['msg']}\"). Will retry on next call."
            )
            return

    _loaded["done"] = engine.ready
    if engine.ready:
        _loaded["error"] = None
    else:
        _loaded["loading_started"] = False
        _loaded["error"] = f"Still loading after {TOTAL_LIMIT}s (last status: \"{_status['msg']}\") — try again shortly"

@mcp.tool()
def search_code(query: str, k: int = 10, project_dir: str = "") -> str:
    """
    Search the local OBSERVE code index by meaning, not exact keywords.

    Use this to find relevant code in the user's indexed codebase(s) when
    they ask about functionality, e.g. "where is health damage handled"
    or "find the function that retries network requests" — it searches
    by semantic similarity using a fine-tuned embedding model, so it
    finds conceptually relevant code even if it doesn't share words with
    the query.

    IMPORTANT — prefer Grep when you already know the exact identifier
    (function/variable/class name, error string, etc). Measured directly
    (see paper/grep_vs_semantic_findings.md): on 5 exact-keyword queries,
    Grep found the correct file 5/5 times; this tool's dedup+relevance-cutoff
    missed 2/5 even when the query WAS the literal function name. This tool's
    real advantage is on natural-language/conceptual queries where the exact
    identifier is unknown — there it matched Grep's recall using a single
    call instead of several sequential exploratory Grep guesses, at ~90%
    fewer tokens. Use Grep first if you can name the thing you're looking
    for; use semantic search when you can only describe it.

    Args:
        query: A natural-language description of what to find.
        k: Number of results to return (default 10).
        project_dir: Optional absolute path to scope results to a single
            indexed project's base directory (e.g. the user's current
            working directory). When set, results from other indexed
            codebases are excluded instead of crowding out the relevant
            project. Leave empty to search across all indexed codebases.

    Returns:
        A formatted list of matching files with relevance scores and
        code previews, ranked by relevance (highest first).
    """
    _ensure_loaded()
    if _loaded["error"]:
        return f"Error: {_loaded['error']}. Build an index first with trit_app.py or trit_search.py --index."

    results = engine.search(query, k=k, base_dir_filter=project_dir or None)
    if not results:
        if project_dir:
            return f"No results under {project_dir} — it may not be indexed, or try without project_dir to search all indexed codebases."
        return "No results — index may be empty. Build one with trit_app.py (INDEX CODEBASE) or trit_search.py --index."

    lines = [f"Found {len(results)} results for: \"{query}\"\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['score']:.3f}] {r['path']}")
        lines.append(f"   {r['preview'][:150]}")
    return "\n".join(lines)

@mcp.tool()
def query_codebase(query: str, k: int = 8, project_dir: str = "") -> str:
    """
    Token-tight variant of search_code: same semantic search, but
    deduplicated (one result per file, best-scoring chunk only),
    relevance-filtered (drops low-scoring results below a fraction of the
    top score instead of always returning a fixed k), and formatted as
    one compact line per result instead of search_code's two-line,
    blank-line-separated format.

    Use this instead of search_code when you want the smallest reasonable
    token footprint and don't need every low-relevance result — e.g. when
    scanning many queries in one session, or working in a tight context
    budget. Use search_code instead when you want the full, uncollapsed
    result set (e.g. multiple chunks from the same file matter, or you
    want to see low-scoring results too).

    IMPORTANT — prefer Grep when you already know the exact identifier.
    This tool's dedup+relevance-cutoff is more aggressive than search_code's
    and measured worse on exact-keyword queries as a result (3/5 recall vs
    Grep's 5/5, see paper/grep_vs_semantic_findings.md) — it can drop a file
    even when the query is a dead-exact keyword match. Its real advantage is
    natural-language/conceptual queries where no exact identifier is known.

    Args:
        query: A natural-language description of what to find.
        k: Maximum number of results to return after dedup/filtering (default 8).
        project_dir: Optional absolute path to scope results to one indexed project.

    Returns:
        One compact line per result: "path  score  preview".
    """
    _ensure_loaded()
    if _loaded["error"]:
        return f"Error: {_loaded['error']}. Build an index first with trit_app.py or trit_search.py --index."

    # over-fetch so dedup/filtering still has enough of a pool to choose from
    raw = engine.search(query, k=max(k * 4, 20), base_dir_filter=project_dir or None)
    if not raw:
        if project_dir:
            return f"No results under {project_dir}."
        return "No results."

    # dedup: keep only the best-scoring chunk per file
    best_per_path = {}
    for r in raw:
        if r["path"] not in best_per_path or r["score"] > best_per_path[r["path"]]["score"]:
            best_per_path[r["path"]] = r
    deduped = sorted(best_per_path.values(), key=lambda r: -r["score"])

    # relevance cutoff: drop results far below the top score instead of
    # padding out to a fixed k with noise (always keep at least 1)
    top_score = deduped[0]["score"]
    threshold = top_score * 0.7
    filtered = [r for r in deduped if r["score"] >= threshold] or deduped[:1]
    final = filtered[:k]

    lines = []
    for r in final:
        preview = " ".join(r["preview"].split())[:90]   # collapse whitespace, tight preview
        lines.append(f"{r['path']}  {r['score']:.2f}  {preview}")
    return "\n".join(lines)

OLLAMA_MODEL = "qwen2.5-coder:7b"
# Separate model for the semantic-role verification call — measured
# directly: qwen2.5-coder:7b (code-focused) missed a real semantic
# mismatch that deepseek-r1:7b (reasoning-focused) caught correctly, on
# the identical prompt (see paper/observe_ollama_findings.md). Generation
# and verification are different tasks; the best model for one is not
# necessarily the best model for the other, even both being 7B-scale.
OLLAMA_VERIFY_MODEL = "deepseek-r1:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    import urllib.request, json as _json
    body = _json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return _json.loads(resp.read())["response"].strip()

LOW_CONFIDENCE_THRESHOLD = 4.0   # below this top-score, treat search context as weak (see trit_cutoff_sweep_test.py for real score ranges)

@mcp.tool()
def propose_change(request: str, project_dir: str = "", model: str = OLLAMA_MODEL) -> str:
    """
    For a vague, ambiguous code-change request (e.g. "make turrets shoot
    faster", "make the shop cheaper") with no specific number or exact
    behavior stated, proposes ONE concrete, grounded default and phrases
    a short confirmation question — instead of silently guessing or
    blocking on the ambiguity.

    Grounds the proposal in REAL existing code: runs a semantic search
    over the indexed project first, and includes the actual matched
    code (with any real existing numeric values) in the prompt. The
    model is REQUIRED to quote the literal source line it grounds a
    proposal in, or explicitly say no relevant existing value was found
    — measured testing (see paper/) found it will otherwise sometimes
    fabricate a plausible-sounding but fake specific detail when the
    search context is weak or truncated, which is worse than an honest
    "no grounding available."

    Uses a local model via Ollama (standard GGUF quantization) running
    entirely on this machine — no cloud call. NOTE: this project's own
    experimental ternary weight quantization was tested on a 7B model
    and produced incoherent garbage (see weight_quantization_findings.md)
    — this tool deliberately uses Ollama's proven quantization instead.

    Use this BEFORE implementing an ambiguous request, so the specific
    number/behavior chosen is explicit, grounded, and confirmable, not
    a silent guess buried inside a code change.

    Args:
        request: The user's plain-language request, as given.
        project_dir: Absolute path to scope the grounding search to one
            indexed project. Strongly recommended — without it, the
            proposal has no real code to anchor to.
        model: Which Ollama model to use (default qwen2.5-coder:7b).
            Pass a different installed model name (e.g. "deepseek-r1:7b")
            to compare hallucination/grounding behavior across models.

    Returns:
        A short proposal: what's ambiguous, a specific suggested value
        (with a literal quote of its real source, or an explicit
        "no grounding found" statement), and a confirmation question.
    """
    _ensure_loaded()
    context = ""
    confidence_note = ""
    if not _loaded["error"]:
        results = engine.search(request, k=5, base_dir_filter=project_dir or None)
        if results:
            top_score = results[0]["score"]
            lines = []
            for r in results:
                preview = " ".join(r["preview"].split())[:350]   # widened from 150 — reduces truncation cutting off the real value
                lines.append(f"{r['path']} (score={r['score']:.2f}): {preview}")
            context = "\n\nReal existing code found in this project, relevant to the request:\n" + "\n".join(lines)
            if top_score < LOW_CONFIDENCE_THRESHOLD:
                confidence_note = (
                    f"\n\nNOTE: the best match above scored only {top_score:.2f}, a WEAK match "
                    "(low confidence). Do not treat this as reliable grounding — if nothing above "
                    "is clearly relevant, say so explicitly rather than using it."
                )
        else:
            confidence_note = "\n\nNOTE: no search results were found at all. There is no existing code to ground a proposal in."

    prompt = (
        "A user gave this code-change request: \"" + request + "\"\n"
        + context + confidence_note + "\n\n"
        "If a specific real value relevant to this request appears in the code above, "
        "quote it directly and propose a moderate, easily-reversible adjustment to it. "
        "If nothing above is clearly relevant, propose a reasonable generic default "
        "instead — do not invent a specific-sounding detail (a number, mechanism, or "
        "field name) that does not literally appear in the search results above. "
        "If the request is already fully specific, just say so in one sentence. "
        "Respond in 2-4 sentences maximum, no preamble."
    )
    try:
        raw = _call_ollama(prompt, model=model)
        return _tag_with_verified_grounding(raw, context, request, model=model)
    except Exception as e:
        return f"Error calling local Ollama model ({model}): {e}. Is Ollama running?"

import re

_COMMON_WORDS = {
    "the", "a", "an", "is", "are", "to", "of", "in", "for", "and", "or",
    "this", "that", "with", "value", "code", "make", "adjust", "adjustment",
    "increase", "decrease", "consider", "could", "would", "should", "may",
    "specific", "real", "default", "generic", "moderate", "current", "change",
    "existing", "relevant", "request", "project", "provided", "snippet",
    "snippets", "shown", "above", "however", "reasonable", "without",
    "being", "than", "based", "using", "used", "already", "somewhat",
}
# Programming keywords across the languages this project indexes — these
# trivially co-occur in almost any two code samples and are not evidence
# of grounding in any SPECIFIC piece of code.
_KEYWORDS = {
    "if", "else", "elif", "for", "while", "def", "func", "var", "return",
    "continue", "break", "class", "import", "from", "true", "false", "none",
    "null", "and", "or", "not", "pass", "self", "print", "int", "float",
    "str", "bool", "void", "extends", "export", "const", "let", "static",
    "public", "private", "protected", "try", "except", "finally", "with",
    "as", "lambda", "yield", "async", "await",
}

def _distinctive_tokens(text: str, exclude: set = frozenset()) -> set:
    """
    Tokens worth treating as evidence of real grounding: numeric literals
    (2+ chars, so bare "1"/"0" don't count), and identifier-shaped words
    (snake_case/camelCase/dotted, length > 5) — filtering out common
    English filler, programming keywords, and (via `exclude`) words
    already present in the user's own request, since a query word
    trivially "matching" the context it was used to search for is
    circular, not evidence of real grounding.
    """
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*|\d+\.?\d*", text)
    out = set()
    for t in tokens:
        low = t.lower()
        if low in _COMMON_WORDS or low in _KEYWORDS or low in exclude:
            continue
        if t.replace(".", "").isdigit():
            cleaned = _clean_number(t)   # strip trailing sentence-period, e.g. "110." -> "110"
            if len(cleaned.replace(".", "")) >= 2:
                out.add(cleaned)   # numeric literal — e.g. "110", "0.92" — excludes bare "1", "0"
        elif "_" in t or any(c.isupper() for c in t[1:]) or len(t) > 5:
            out.add(t)   # identifier-shaped — e.g. "fire_rate", "N_DEFECTS", "threshold"
    return out

def _tag_with_verified_grounding(response: str, context: str, request: str = "", model: str = OLLAMA_MODEL) -> str:
    """
    Objectively checks whether the model's response shares distinctive
    tokens (numbers, identifiers) with the real search context, instead of
    trusting the model's own self-report (measured unreliable — see
    paper/observe_ollama_findings.md). Excludes the user's own query words
    (circular matching) and programming keywords (trivial co-occurrence).

    Additionally cross-checks numeric literals specifically: if the
    response states a number as if it were the CURRENT/existing value,
    but that exact number never appears anywhere in the real context,
    it's flagged separately — this catches the harder "right variable,
    fabricated number" case (e.g. correctly citing `threshold` but
    inventing `100` when the real value is `110`), which plain token
    overlap alone does not catch, since `threshold` alone would still
    match even with a wrong number attached to it.
    """
    if not context:
        return "[no search context available — unverified default] " + response

    query_words = {w.lower() for w in re.findall(r"[A-Za-z_]+", request)}
    response_tokens = _distinctive_tokens(response, exclude=query_words)
    context_tokens = _distinctive_tokens(context, exclude=query_words)
    shared = response_tokens & context_tokens

    # Numeric cross-check: numbers claimed in the response that never
    # appear anywhere in the real context at all (possible fabrication)
    response_numbers = {t for t in response_tokens if t.replace(".", "").isdigit()}
    context_numbers = {t for t in context_tokens if t.replace(".", "").isdigit()}
    unverified_numbers = response_numbers - context_numbers

    # Deterministic variable=value pair cross-check — catches the subtler
    # case token-overlap alone misses: a real variable name correctly
    # referenced, but paired with a number that never actually belonged
    # to it (a coincidental match elsewhere in context created false
    # confidence — e.g. citing spawn_timer.wait_time=1.0 when the real
    # pair is spawn_timer.wait_time=1.5, where "1.0" happens to appear
    # elsewhere in context attached to something unrelated).
    #
    # Only checked against pairs the response presents as CURRENT/EXISTING
    # state — a proposed NEW value is SUPPOSED to differ from context by
    # design (that's the point of a change), so proposed values are
    # excluded via _extract_current_state_pairs' proximity heuristic
    # (measured false-positive case: "changing k=10 to k=5" was originally
    # flagged as a "mismatch" against the real k=10, when it was actually
    # a correct, intentional proposal — see paper/observe_ollama_findings.md).
    context_pairs = _extract_assignment_pairs(context)
    response_pairs = _extract_current_state_pairs(response)
    mismatches = []
    for var, claimed_vals in response_pairs.items():
        for ctx_var, real_val in context_pairs.items():
            if var == ctx_var or var in ctx_var or ctx_var in var:
                # Only flag if NONE of the surviving claimed values for this
                # variable match reality — a single dict overwrite silently
                # dropping the correct claim (measured real bug: "threshold=100.
                # changing threshold=80 instead" kept only 80, discarding the
                # correct 100, because "instead" alone wasn't a recognized
                # proposal-phrase trigger) previously caused false mismatches.
                # Checking against ALL surviving values, not just the last
                # one seen, is robust to that class of bug even when the
                # phrase-trigger list is incomplete.
                if real_val not in claimed_vals:
                    shown = "/".join(claimed_vals)
                    mismatches.append(f'"{var}" claimed CURRENT={shown} but real value is {real_val}')
                break

    parts = []
    if shared:
        shown = ", ".join(sorted(shared)[:5])
        parts.append(f"shares real tokens: {shown}")
    else:
        parts.append("no shared distinctive tokens with search context")
    if unverified_numbers:
        shown_n = ", ".join(sorted(unverified_numbers)[:5])
        parts.append(f"numbers not found anywhere in context: {shown_n}")
    if mismatches:
        parts.append("VARIABLE=VALUE MISMATCH (verified wrong): " + "; ".join(mismatches))

    if mismatches:
        status = "FABRICATION DETECTED"
    elif shared and not unverified_numbers:
        status = "verified grounded"
    elif shared:
        status = "PARTIAL — grounded reference but unverified number(s)"
    else:
        status = "unverified"

    # Semantic-role check — a NARROWER, separate class of error that token
    # matching alone cannot catch: a number genuinely present in context,
    # correctly identified as "shared," but attached to a completely wrong
    # MEANING in the response (e.g. citing a real "1958" from an unrelated
    # historical reference elsewhere in the codebase as if it were a
    # tunable "benchmark year" parameter).
    #
    # Runs on any SHARED numeric token, regardless of overall status — NOT
    # gated on status == "verified grounded". Measured bug: that gate
    # required zero unverified numbers, but a proposed NEW value (e.g. the
    # response's own suggested replacement) is *always* unverified by
    # design, so nearly every real response landed on "PARTIAL" and the
    # check never ran in practice. Shared tokens can still have a wrong
    # semantic role even when other, unrelated numbers in the same
    # response are legitimately-unverified proposals — these are
    # independent concerns and must be checked independently. Skipped
    # entirely if a MISMATCH was already found (that's the stronger,
    # already-confirmed signal; no need for a second, costlier call).
    if not mismatches:
        shared_numbers = [t for t in shared if t.replace(".", "").isdigit()]
        for token in shared_numbers[:2]:   # cap at 2 checks — cost/latency limit
            verdict = _check_semantic_role(token, response, context)
            if verdict is False:
                status = "SEMANTIC MISMATCH DETECTED"
                parts.append(f'token "{token}" is real but its claimed meaning does not match its actual role in the source')
                break

    return f"[{status} — {'; '.join(parts)}] {response}"

def _extract_window(text: str, token: str, chars: int = 100) -> str:
    """Return the text surrounding one occurrence of `token`, as a stand-in
    for 'the sentence this token appears in' without needing real sentence
    parsing — good enough for a narrow yes/no role-matching prompt."""
    idx = text.find(token)
    if idx == -1:
        return ""
    return text[max(0, idx - chars): idx + len(token) + chars].strip()

# Consensus verification pool — deliberately diverse model families
# (code-specialized, reasoning-specialized, general-purpose), not just
# multiple sizes of the same model, since correlated failure modes would
# defeat the point of voting. Applies this project's own scoping rule
# from paper/npc_consensus_findings.md, confirmed in 5 independent domains
# tonight: voting beats trusting a single source specifically when
# individual signals are uncalibrated and fail unpredictably — which is
# exactly what was measured here (qwen2.5-coder:7b wrong, deepseek-r1:7b
# right, on the identical verification prompt).
CONSENSUS_VERIFY_MODELS = ("qwen2.5-coder:7b", "deepseek-r1:7b", "llama3.2:latest")

def _check_semantic_role(token: str, response: str, context: str, models=CONSENSUS_VERIFY_MODELS) -> bool:
    """
    Consensus verification: does this token's claimed role/meaning in the
    response actually match its real role in the source text? Polls
    multiple independent models and requires a MAJORITY to agree on
    "DIFFERENT" before flagging a mismatch — not a single model's opinion.

    Returns True (majority says matches, or no clear majority — err
    toward not flagging), False (majority says mismatch — real bug
    caught, cross-confirmed by multiple independent models), or None
    (no model could be reached at all).
    """
    response_snippet = _extract_window(response, token)
    context_snippet = _extract_window(context, token)
    if not response_snippet or not context_snippet:
        return None

    prompt = (
        f'Source text (real, from the codebase): "{context_snippet}"\n'
        f'Claim (from a proposed code change): "{response_snippet}"\n\n'
        f'Both mention the number {token}. Does the CLAIM use {token} with the '
        f'SAME meaning/role it actually has in the SOURCE text (e.g. the same '
        f'variable, setting, or concept) — or does the claim attach {token} to '
        f'a different, unrelated meaning than what it represents in the source? '
        f'Answer with exactly one word: SAME or DIFFERENT.'
    )

    votes = []
    for m in models:
        try:
            raw = _call_ollama(prompt, model=m).strip().upper()
            if "DIFFERENT" in raw:
                votes.append(False)
            elif "SAME" in raw:
                votes.append(True)
            # unparseable response from this model — no vote cast, not an error
        except Exception:
            pass   # this model unreachable/errored — skip it, don't block on one failure

    if not votes:
        return None   # no model could be reached at all
    different_votes = votes.count(False)
    same_votes = votes.count(True)
    # Majority rule — ties (e.g. 1-1 with one model unreachable) default
    # to NOT flagging, since this is a supplementary check and a false
    # "mismatch" alarm has its own cost (see paper/observe_ollama_findings.md
    # false-positive history throughout this whole build).
    return different_votes <= same_votes

_ASSIGNMENT_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_.]*)\s*(?::\s*\w+\s*)?=\s*([-+]?\d+\.?\d*)"
)
# Prose equivalents — real responses describe a value in many different
# ways ("threshold is set to 100", "fire_rate is currently 0.92", "might
# be around 100", "has a delay of 1 second"), not just code syntax or one
# fixed phrasing. This is an open-ended natural-language coverage problem
# (infinite ways to phrase "the current value is X") — these patterns
# cover the phrasings measured in real responses so far, not an exhaustive
# set. See paper/observe_ollama_findings.md for the specific cases each
# pattern was added to catch.
_PROSE_VALUE_PATTERNS = [
    re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s+is\s+(?:currently\s+|set\s+to\s+|)([-+]?\d+\.?\d*)"),
    re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s+(?:might\s+be|appears?\s+to\s+be|seems?\s+to\s+be)\s+(?:around\s+|about\s+|)([-+]?\d+\.?\d*)"),
    re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s+(?:currently\s+)?has\s+a\s+\w+\s+of\s+([-+]?\d+\.?\d*)"),
]

def _clean_number(val: str) -> str:
    """Strip a trailing bare '.' left over from sentence-ending punctuation
    the regex incidentally captured (e.g. "100." at the end of a sentence)."""
    return val[:-1] if val.endswith(".") and val.count(".") == 1 else val

def _extract_assignment_pairs(text: str) -> dict:
    """
    Deterministic extraction of (identifier, value) pairs from an
    `identifier = number` or `identifier: type = number` shape — matches
    the real patterns seen across this project's languages: GDScript
    (`@export var fire_rate: float = 0.25`, `spawn_timer.wait_time = 1.5`)
    and Python (`threshold=110`, `N_DEFECTS = 40`). No model involved —
    pure regex/structural matching, so it either finds a real match or it
    doesn't, with no hallucination risk in the check itself. Last match
    per identifier wins (in case of duplicates in one text blob).
    """
    pairs = {}
    for var, val in _ASSIGNMENT_PATTERN.findall(text):
        pairs[var] = _clean_number(val)
    for prose_pattern in _PROSE_VALUE_PATTERNS:
        for var, val in prose_pattern.findall(text):
            pairs.setdefault(var, _clean_number(val))
    return pairs

# Phrases that signal the number right after them is a PROPOSED new value,
# not a claim about the current/existing state — measured from real
# response phrasing patterns ("changing X to Y", "consider Y", "such as Y").
_PROPOSAL_LEAD_PHRASES = (
    " to ", " to`", "to →", "->", "consider ", "propose ", "suggest ",
    "such as ", "instead of ", "could be ", "moderate adjustment",
    "default of ", "a default ", "you might change it to",
)

def _extract_current_state_pairs(text: str, window: int = 30) -> dict:
    """
    Same extraction as _extract_assignment_pairs, but excludes any
    identifier=number match whose number is immediately preceded (within
    `window` characters) by a proposal-indicating phrase — those are the
    model's PROPOSED new value, which is supposed to differ from the real
    context by design, not a false claim about current state.

    Returns {var: [values]} — a LIST per variable, not a single value.
    Measured real bug with a single-value dict: "threshold=100. changing
    threshold=80 instead" kept only 80 (last-seen wins on dict overwrite),
    silently discarding the correct current-state claim of 100, because
    "instead" alone wasn't a recognized proposal-phrase trigger. Returning
    all surviving values lets the caller check "does ANY surviving claim
    match reality" instead of trusting whichever value happened to be
    extracted last — robust to gaps in the phrase-trigger list.
    """
    pairs = {}
    for pattern in (_ASSIGNMENT_PATTERN, *_PROSE_VALUE_PATTERNS):
        for m in pattern.finditer(text):
            var, val = m.group(1), _clean_number(m.group(2))
            # Also check inside the match itself for a leading "= " immediately
            # after words like "to"/"->" (covers "changing X to Y" where Y
            # isn't its own identifier=value pair but follows "to").
            lookback = text[max(0, m.start() - window): m.start()].lower()
            if any(p in lookback for p in _PROPOSAL_LEAD_PHRASES):
                continue   # this is a proposed value, not a current-state claim — skip
            pairs.setdefault(var, []).append(val)
    return pairs

@mcp.tool()
def index_status() -> str:
    """
    Report the current OBSERVE index status — how many chunks/files are
    indexed, and which model is being used. Use this to check whether a
    search is likely to find anything before running search_code.
    """
    _ensure_loaded()
    if _loaded["error"]:
        return f"Not ready: {_loaded['error']}"
    n_chunks = len(engine.metadata)
    return f"OBSERVE index ready — {n_chunks:,} chunks indexed, model: {MODEL_PATH}"

GODOT_EXE = str(Path.home() / "Downloads" / "Godot_v4.6.2-stable_win64.exe" / "Godot_v4.6.2-stable_win64_console.exe")

@mcp.tool()
def apply_and_verify(file_path: str, old_text: str, new_text: str) -> str:
    """
    Closes the loop that search_code/query_codebase/propose_change stop
    short of: applies a real edit to a real file, then verifies it by
    ACTUALLY RUNNING the code — not just checking the text changed.

    Honest scope: this verifies (1) the edit applied cleanly to exactly
    one location, (2) the file still parses/compiles after the edit
    (catches a syntax-breaking edit), and (3) for .gd files, that Godot
    can load the script without error. It does NOT automatically write or
    run a custom behavioral test asserting the new value does what you
    intended — that's a genuinely separate, per-feature step (see the
    turret fire_rate / bulk-discount TDD examples built earlier this
    session, done by hand with a purpose-built test file). This tool is
    the safety-net "did I break anything" check, not a substitute for a
    real test asserting the specific intended behavior.

    Use this AFTER propose_change has given you a specific, grounded
    old_text/new_text pair to apply — not as a way to skip writing a real
    test for anything that matters.

    Args:
        file_path: Absolute path to the real file to edit.
        old_text: The exact existing text to replace — must appear
            exactly once in the file, or the edit is rejected (ambiguous
            edits are refused rather than guessed at).
        new_text: The replacement text.

    Returns:
        A report: whether the edit applied, whether the file still
        parses/loads after the change, and the file reverted if it broke.
    """
    path = Path(file_path)
    if not path.exists():
        return f"FAILED — file does not exist: {file_path}"

    original = path.read_text(encoding="utf-8")
    count = original.count(old_text)
    if count == 0:
        return f"FAILED — old_text not found in {file_path}. No edit applied."
    if count > 1:
        return (f"FAILED — old_text appears {count} times in {file_path}, ambiguous which "
                f"one to replace. No edit applied. Include more surrounding context in old_text "
                f"to make it unique.")

    edited = original.replace(old_text, new_text)
    path.write_text(edited, encoding="utf-8")

    # Verify the file still parses/loads — catches a syntax-breaking edit
    ext = path.suffix.lower()
    if ext == ".py":
        result = subprocess.run([sys.executable, "-m", "py_compile", str(path)],
                                capture_output=True, text=True, timeout=30)
        ok = result.returncode == 0
        detail = result.stderr.strip() if not ok else "compiles cleanly"
    elif ext == ".gd":
        if not Path(GODOT_EXE).exists():
            path.write_text(original, encoding="utf-8")   # can't verify — revert to be safe
            return (f"FAILED — edit applied but could not verify: Godot not found at "
                    f"{GODOT_EXE}. Reverted the edit to be safe.")
        result = subprocess.run(
            [GODOT_EXE, "--headless", "--check-only", "--script", str(path)],
            capture_output=True, text=True, timeout=60, cwd=str(path.parent)
        )
        # Godot's --check-only returns 0 and no "Parse Error" on a valid script
        ok = result.returncode == 0 and "error" not in result.stderr.lower()
        detail = "loads cleanly in Godot" if ok else (result.stderr.strip() or result.stdout.strip())
    else:
        ok = True
        detail = f"no syntax checker for {ext} files — edit applied, unverified"

    if not ok:
        path.write_text(original, encoding="utf-8")   # revert — don't leave a broken file
        return (f"REVERTED — edit broke the file. {detail}\n"
                f"The file has been restored to its original state. No changes were kept.")

    return (f"APPLIED AND VERIFIED — {file_path} edited successfully, replacing:\n"
            f'  "{old_text}"\nwith:\n  "{new_text}"\n'
            f"Syntax check: {detail}\n"
            f"NOTE: this confirms the file still parses/loads — it does NOT confirm the new "
            f"behavior is what you intended. Write a real test for that if this change matters.")

if __name__ == "__main__":
    mcp.run(transport="stdio")
