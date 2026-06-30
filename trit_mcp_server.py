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

if __name__ == "__main__":
    mcp.run(transport="stdio")
