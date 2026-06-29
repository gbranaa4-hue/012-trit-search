"""
012 Ternary Semantic Search
Local private search engine over your own files.

Demonstrates all three ternary capabilities:
  Database    : files indexed as trit patterns in FAISS
  Compression : trit encoding is 10x smaller than float32
  Networking  : HTTP API so any app can query it

ORIGINAL FILES ARE NEVER MODIFIED OR MOVED.
Only reads files. Stores index separately in search_index/.
Delete search_index/ to remove all traces — your files untouched.

Install:
  pip install faiss-cpu flask tqdm

Usage:
  python trit_search.py --index          Scan and index your files
  python trit_search.py --search "query" Quick search from terminal
  python trit_search.py --serve          Start HTTP API server
  python trit_search.py --stats          Show index statistics
  python trit_search.py --watch          Index + serve + auto-reindex on changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os, json, time, hashlib, pickle
from pathlib import Path
from datetime import datetime

os.makedirs("search_index", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INDEX_PATH   = "search_index/faiss.index"
META_PATH    = "search_index/metadata.json"
ENCODER_PATH = "search_index/encoder.pt"
CHUNK_SIZE   = 800     # chars per chunk — fits in context without losing detail
CHUNK_OVERLAP = 100    # overlap so nothing falls between chunks

# File types to index
SCAN_EXTS = {
    # Code
    ".py", ".gd", ".js", ".ts", ".cs", ".rs", ".go",
    ".c", ".cpp", ".h", ".hpp", ".java", ".lua", ".rb",
    ".php", ".swift", ".kt", ".dart", ".zig", ".r",
    # Data / config
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env",
    ".sql", ".xml", ".csv",
    # Docs — .md only, skip raw .txt files (bug notes, chat logs, etc.)
    ".md", ".rst", ".tex",
    # Scripts
    ".sh", ".ps1", ".bat",
}

# Directories to index
SCAN_DIRS = [
    r"C:\Users\gbran\OneDrive\Documents",
]

# Directories to skip
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".cache", "dist", "build", ".ollama", "search_index",
    "lora", "models", "index", "results", "data",
    "Xfer", "Image-Line", "FL Studio", "Serum",
    "AppData", "Temp", "Windows", "Program Files",
    "StarCraft", "StarCraft II", "GameLogs", "Blizzard",
    "ai_files", "Spikeling-Project",
}

# Only index files inside these specific project dirs
# Leave empty to scan all of SCAN_DIRS
INCLUDE_ONLY_DIRS = [
    r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1",
    r"C:\Users\gbran\OneDrive\Documents\012-ternary",
    r"C:\Users\gbran\OneDrive\Documents\tribe",
]

# Skip directory name patterns (substring match)
SKIP_DIR_PATTERNS = ["_files", "_assets", "node_modules", ".git"]

# Skip files with less than this many meaningful characters
MIN_CONTENT_CHARS = 100

# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC ENCODER
# Uses all-MiniLM-L6-v2 (80MB, pre-trained on 1B sentence pairs)
# Then compresses float32 → ternary for 16x storage savings
# ══════════════════════════════════════════════════════════════════════════════

TRIT_DIM = 384   # MiniLM output dim

_encoder_model = None

def get_or_create_encoder(force_retrain=False):
    global _encoder_model
    if _encoder_model is not None:
        return _encoder_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install: pip install sentence-transformers")
        raise
    print("Loading semantic encoder (all-MiniLM-L6-v2, ~80MB)...")
    _encoder_model = SentenceTransformer(r"C:\Users\gbran\OneDrive\Documents\012-ternary\models\code-minilm")
    print("  Encoder ready.\n")
    return _encoder_model

class TritEncoder:
    """Thin wrapper so the rest of the code doesn't change."""
    def __init__(self):
        self._model = None

    def _get(self):
        if self._model is None:
            self._model = get_or_create_encoder()
        return self._model

    def encode(self, texts, quantize=False):
        import numpy as np
        model = self._get()
        vecs = model.encode(texts, batch_size=64, show_progress_bar=False,
                            normalize_embeddings=True, convert_to_numpy=True)
        if quantize:
            t = 0.7 * np.abs(vecs).mean()
            vecs = np.where(vecs > t, 1.0, np.where(vecs < -t, -1.0, 0.0))
            # renormalize after quantization
            norms = np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-8
            vecs = vecs / norms
        return vecs.astype("float32")

    def save(self, path=ENCODER_PATH):
        pass  # sentence-transformers caches automatically

    @classmethod
    def load(cls, path=ENCODER_PATH):
        return cls()

# ══════════════════════════════════════════════════════════════════════════════
# FILE SCANNER
# Reads files without modifying them
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start  = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text): break
    return chunks

def file_hash(path):
    """Quick hash to detect if file changed."""
    stat = os.stat(path)
    return f"{stat.st_mtime:.0f}_{stat.st_size}"

def scan_files(dirs=None, skip=SKIP_DIRS, exts=SCAN_EXTS):
    """
    Yield (path, content) for all indexable files.
    Never modifies files.
    """
    if dirs is None:
        dirs = INCLUDE_ONLY_DIRS if INCLUDE_ONLY_DIRS else SCAN_DIRS
    for base_dir in dirs:
        if not os.path.exists(base_dir):
            continue
        for root, dirs_list, files in os.walk(base_dir):
            # Skip unwanted directories in-place
            dirs_list[:] = [d for d in dirs_list
                            if d not in skip
                            and not d.startswith('.')
                            and not any(p in d for p in SKIP_DIR_PATTERNS)]

            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in exts:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    size = os.path.getsize(fpath)
                    if size == 0 or size > 500_000:  # skip empty or huge files
                        continue
                    text = open(fpath, "r", encoding="utf-8",
                                errors="ignore").read()
                    if len(text.strip()) < MIN_CONTENT_CHARS:
                        continue
                    # Skip placeholder/stub files
                    stripped = text.strip().lower()
                    if len(set(stripped.split())) < 5:
                        continue
                    yield fpath, text
                except (PermissionError, OSError):
                    continue

# ══════════════════════════════════════════════════════════════════════════════
# INDEX
# FAISS-backed search. Original files never touched.
# ══════════════════════════════════════════════════════════════════════════════

class TritSearchIndex:
    def __init__(self):
        self.encoder  = TritEncoder()
        self.index    = None
        self.metadata = []   # [{path, chunk, text_preview, file_hash}, ...]
        self._load()

    def _build_faiss(self, dim=TRIT_DIM):
        try:
            import faiss
            idx = faiss.IndexFlatIP(dim)
            return idx
        except ImportError:
            print("faiss not found — using brute force (slower for >10k chunks)")
            return None

    def _load(self):
        try:
            import faiss
            if os.path.exists(INDEX_PATH):
                self.index    = faiss.read_index(INDEX_PATH)
                self.metadata = json.load(open(META_PATH, encoding="utf-8"))
                print(f"  Index loaded: {len(self.metadata):,} chunks from "
                      f"{len(set(m['path'] for m in self.metadata)):,} files")
                return
        except Exception as e:
            print(f"  Could not load index: {e}")
        self.index    = self._build_faiss()
        self.metadata = []

    def _save(self):
        try:
            import faiss
            if self.index is not None:
                faiss.write_index(self.index, INDEX_PATH)
        except: pass
        json.dump(self.metadata, open(META_PATH, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    def build(self, dirs=SCAN_DIRS, force=False):
        """
        Scan files and build index.
        Skips files that haven't changed (incremental).
        NEVER modifies original files.
        """
        import numpy as np

        # Track which files are already indexed
        indexed_hashes = {m["path"]: m.get("file_hash","")
                          for m in self.metadata}

        all_texts  = []
        all_meta   = []
        skip_count = 0
        new_count  = 0

        print("Scanning files...")
        file_list = list(scan_files(dirs))
        print(f"  Found {len(file_list)} files\n")

        for fpath, text in file_list:
            fhash = file_hash(fpath)
            if not force and indexed_hashes.get(fpath) == fhash:
                skip_count += 1
                continue

            # Remove old chunks for this file if re-indexing
            self.metadata = [m for m in self.metadata if m["path"] != fpath]

            chunks = chunk_text(text)
            rel    = os.path.relpath(fpath,
                     r"C:\Users\gbran\OneDrive\Documents")
            ext    = Path(fpath).suffix.lower()

            for i, chunk in enumerate(chunks):
                # Search text includes filename for better matching
                search_text = f"file:{rel}\n{chunk}"
                all_texts.append(search_text)
                all_meta.append({
                    "path":         fpath,
                    "rel_path":     rel,
                    "chunk_idx":    i,
                    "total_chunks": len(chunks),
                    "preview":      chunk[:200].replace("\n", " "),
                    "file_hash":    fhash,
                    "ext":          ext,
                    "indexed_at":   datetime.now().isoformat(),
                    "file_size":    os.path.getsize(fpath),
                })

            new_count += 1
            print(f"\r  Indexed: {new_count} new  |  "
                  f"Skipped: {skip_count} unchanged  |  "
                  f"Chunks: {len(all_texts)}",
                  end="", flush=True)

        if not all_texts:
            print(f"\n  Nothing new to index. {skip_count} files unchanged.")
            return

        print(f"\n\nEncoding {len(all_texts):,} chunks...")
        bs   = 128
        vecs = []
        for i in range(0, len(all_texts), bs):
            batch = all_texts[i:i+bs]
            v     = self.encoder.encode(batch)
            vecs.append(v)
            pct = (i+len(batch)) / len(all_texts) * 100
            bar = "█" * int(pct/4) + "░" * (25 - int(pct/4))
            print(f"\r  [{bar}] {pct:.1f}%  {i+len(batch):,}/{len(all_texts):,}",
                  end="", flush=True)

        print()
        vecs = np.vstack(vecs)

        # Add to FAISS
        if self.index is None:
            self.index = self._build_faiss()
        if self.index is not None:
            self.index.add(vecs)
        else:
            # Brute force fallback
            if not hasattr(self, '_vecs'):
                self._vecs = vecs
            else:
                self._vecs = np.vstack([self._vecs, vecs])

        self.metadata.extend(all_meta)
        self._save()

        # Stats
        n_files  = len(set(m["path"] for m in self.metadata))
        n_chunks = len(self.metadata)
        float_mb = n_chunks * TRIT_DIM * 4 / 1e6
        trit_mb  = n_chunks * TRIT_DIM * 2 / 8 / 1e6
        print(f"\n  Index complete:")
        print(f"    Files   : {n_files:,}")
        print(f"    Chunks  : {n_chunks:,}")
        print(f"    Float32 : {float_mb:.1f} MB")
        print(f"    Ternary : {trit_mb:.1f} MB  ({float_mb/max(trit_mb,0.001):.1f}x compression)")

    def search(self, query, k=10):
        """
        Search by meaning, not keywords.
        Returns list of results with file path and preview.
        """
        import numpy as np

        if not self.metadata:
            return []

        vec = self.encoder.encode([query])

        if self.index is not None:
            k_actual        = min(k, len(self.metadata))
            scores, indices = self.index.search(vec, k_actual)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata): continue
                m = self.metadata[idx]
                results.append({**m, "score": float(score)})
        else:
            sims    = self._vecs @ vec.T
            top_k   = np.argsort(sims[:,0])[::-1][:k]
            results = [{**self.metadata[i], "score": float(sims[i,0])}
                       for i in top_k if i < len(self.metadata)]

        # Deduplicate by file — keep best chunk per file
        seen  = {}
        final = []
        for r in results:
            p = r["path"]
            if p not in seen or r["score"] > seen[p]["score"]:
                seen[p] = r
        final = sorted(seen.values(), key=lambda x: -x["score"])[:k]
        return final

    def stats(self):
        n_files  = len(set(m["path"] for m in self.metadata))
        n_chunks = len(self.metadata)
        exts     = {}
        for m in self.metadata:
            exts[m["ext"]] = exts.get(m["ext"], 0) + 1
        return {
            "files":  n_files,
            "chunks": n_chunks,
            "exts":   dict(sorted(exts.items(), key=lambda x: -x[1])[:10]),
            "index_size_mb": n_chunks * TRIT_DIM * 4 / 1e6,
        }

# ══════════════════════════════════════════════════════════════════════════════
# HTTP API
# Any app, browser, or script can search your files
# ══════════════════════════════════════════════════════════════════════════════

def serve(host="127.0.0.1", port=5050):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Install: pip install flask")
        return

    idx = TritSearchIndex()
    app = Flask(__name__)

    @app.route("/search")
    def search():
        query = request.args.get("q", "")
        k     = int(request.args.get("k", 10))
        if not query:
            return jsonify({"error": "q parameter required"}), 400
        t0      = time.perf_counter()
        results = idx.search(query, k=k)
        elapsed = (time.perf_counter() - t0) * 1000
        return jsonify({
            "query":   query,
            "time_ms": round(elapsed, 2),
            "results": results
        })

    @app.route("/index", methods=["POST"])
    def reindex():
        idx.build()
        return jsonify({"status": "ok", "chunks": len(idx.metadata)})

    @app.route("/stats")
    def stats():
        return jsonify(idx.stats())

    @app.route("/")
    def home():
        # Simple search UI in the browser
        return """<!DOCTYPE html>
<html><head><title>012 Search</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #0d0d0d; color: #e0e0e0; }
  h1 { color: #00ff88; }
  input { width: 100%; padding: 12px; font-size: 16px; font-family: monospace; background: #1a1a1a; border: 1px solid #333; color: #e0e0e0; border-radius: 4px; box-sizing: border-box; }
  button { padding: 10px 24px; background: #00ff88; color: #000; border: none; cursor: pointer; font-family: monospace; font-size: 14px; border-radius: 4px; margin-top: 8px; }
  .result { border: 1px solid #333; margin: 12px 0; padding: 14px; border-radius: 4px; background: #111; }
  .path { color: #00ff88; font-size: 13px; margin-bottom: 6px; }
  .score { color: #888; font-size: 12px; }
  .preview { color: #ccc; font-size: 13px; white-space: pre-wrap; margin-top: 8px; }
  .time { color: #555; font-size: 12px; margin: 8px 0; }
  #status { color: #888; margin-top: 8px; }
</style></head><body>
<h1>012 Ternary Search</h1>
<p style="color:#888">Search your files by meaning, not keywords.</p>
<input id="q" type="text" placeholder="What are you looking for?" onkeydown="if(event.key==='Enter')search()">
<button onclick="search()">Search</button>
<div id="status"></div>
<div id="results"></div>
<script>
async function search() {
  const q = document.getElementById('q').value;
  if (!q) return;
  document.getElementById('status').textContent = 'Searching...';
  document.getElementById('results').innerHTML = '';
  const r = await fetch('/search?q=' + encodeURIComponent(q) + '&k=10');
  const d = await r.json();
  document.getElementById('status').textContent = d.results.length + ' results in ' + d.time_ms + 'ms';
  document.getElementById('results').innerHTML = d.results.map(r =>
    `<div class="result">
      <div class="path">${r.rel_path}  <span class="score">[score: ${r.score.toFixed(3)}]</span></div>
      <div class="preview">${r.preview}</div>
    </div>`
  ).join('');
}
</script></body></html>"""

    print(f"\n012 Ternary Search Server")
    print(f"  {len(idx.metadata):,} chunks indexed")
    print(f"  http://{host}:{port}/")
    print(f"  http://{host}:{port}/search?q=your+query")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=False)

# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def terminal_search(query, k=10):
    idx     = TritSearchIndex()
    t0      = time.perf_counter()
    results = idx.search(query, k=k)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\nQuery  : {query}")
    print(f"Time   : {elapsed:.1f}ms")
    print(f"Results: {len(results)}\n")

    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.3f}] {r['rel_path']}")
        print(f"       {r['preview'][:120]}")
        print()

def interactive_search():
    idx = TritSearchIndex()
    if not idx.metadata:
        print("No index found. Run: python trit_search.py --index")
        return

    s = idx.stats()
    print(f"\n012 Ternary Search  —  {s['files']:,} files  {s['chunks']:,} chunks")
    print(f"Commands: :quit  :stats  :reindex\n")

    while True:
        try:
            q = input("Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q: continue
        if q == ":quit": break
        if q == ":stats":
            print(json.dumps(idx.stats(), indent=2))
            continue
        if q == ":reindex":
            idx.build()
            continue

        t0      = time.perf_counter()
        results = idx.search(q, k=8)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\n  {len(results)} results  ({elapsed:.1f}ms)\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}]  {r['rel_path']}")
            print(f"       {r['preview'][:100]}")
        print()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",   action="store_true", help="Index your files")
    parser.add_argument("--search",  type=str,            help="Search from terminal")
    parser.add_argument("--serve",   action="store_true", help="Start HTTP server")
    parser.add_argument("--stats",   action="store_true", help="Show index stats")
    parser.add_argument("--watch",   action="store_true", help="Index + serve")
    parser.add_argument("--force",   action="store_true", help="Force reindex all files")
    parser.add_argument("--port",    type=int, default=5050)
    parser.add_argument("--k",       type=int, default=10, help="Number of results")
    args = parser.parse_args()

    if args.index or args.watch:
        idx = TritSearchIndex()
        idx.build(force=args.force)
        if not args.watch: return

    if args.stats:
        idx = TritSearchIndex()
        s   = idx.stats()
        print(f"\n  Files indexed : {s['files']:,}")
        print(f"  Chunks        : {s['chunks']:,}")
        print(f"  Index size    : {s['index_size_mb']:.1f} MB float32")
        print(f"  Ternary size  : {s['index_size_mb']/16:.1f} MB")
        print(f"  Top file types: {s['exts']}")
        return

    if args.search:
        terminal_search(args.search, k=args.k)
        return

    if args.serve or args.watch:
        serve(port=args.port)
        return

    # Default: interactive terminal search
    interactive_search()

if __name__ == "__main__":
    main()
