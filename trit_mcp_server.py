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
