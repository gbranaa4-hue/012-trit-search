#!/usr/bin/env python3
"""
OBSERVE one-command demo.

    observe-demo [directory]

Indexes the given directory (default: the current one) into a dedicated
demo index at ~/.trit-search/demo-index — it never touches your main
OBSERVE index — then runs a few natural-language searches against it and
prints what to set up next. First run downloads the embedding model if
the fine-tuned one isn't present.
"""
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trit_app import SearchEngine

DEMO_INDEX_DIR = str(Path.home() / ".trit-search" / "demo-index")

DEMO_QUERIES = [
    "where is the main entry point",
    "error handling and retrying after failure",
    "where configuration or settings get loaded",
]


def main():
    target = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
    if not os.path.isdir(target):
        print(f"Not a directory: {target}")
        return 1

    model_path = str(Path(__file__).resolve().parent / "models" / "code-minilm")
    if not Path(model_path).exists():
        model_path = "all-MiniLM-L6-v2"   # auto-downloads on first use

    print(f"OBSERVE demo\n  indexing: {target}\n  demo index: {DEMO_INDEX_DIR} "
          f"(separate from any main index)\n  model: {model_path}\n")

    from sentence_transformers import SentenceTransformer
    print("Loading embedding model (first run may download it)...")
    engine = SearchEngine()
    engine.model = SentenceTransformer(model_path)

    done = threading.Event()
    status = {"msg": ""}

    def on_status(msg):
        status["msg"] = msg
        print(f"  {msg}")

    engine.build_index([target], DEMO_INDEX_DIR, on_status, done.set)
    t0 = time.time()
    while not done.is_set():
        if status["msg"].startswith("Index error"):
            print("\nIndexing failed — see message above.")
            return 1
        if time.time() - t0 > 900:
            print("\nIndexing timed out after 15 minutes.")
            return 1
        time.sleep(0.5)

    print("\n--- Example searches (semantic — no exact keywords needed) ---")
    for q in DEMO_QUERIES:
        print(f'\n>> "{q}"')
        results = engine.search(q, k=3)
        if not results:
            print("   (no results)")
            continue
        for r in results:
            print(f"   [{r['score']:.2f}] {r['path']}")
            if r["preview"]:
                print(f"          {r['preview'][:110]}")

    print("""
--- Next steps ---
  GUI (index more folders, browse results):    observe
  CLI:                                          observe-search --search "your query"
  Plug into Claude Code as an MCP tool:
      claude mcp add observe -- observe-mcp
  (then build your real index via the GUI; the demo index above is throwaway)

  What's measured, including where grep beats this: see OBSERVE.md
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
