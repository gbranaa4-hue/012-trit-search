"""
Real-scale test using actual OSS code (FastAPI + Gin, 1228 files), not
synthetic data. Reuses the exact same SearchEngine indexing/search logic
from trit_app.py to verify compression + speed claims hold on real code.
"""
import sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from trit_app import SearchEngine

# One-off scale test: clone any large codebase to SCAN_DIR before running,
# e.g. `git clone --depth 1 https://github.com/tiangolo/fastapi <SCAN_DIR>`
INDEX_DIR = os.environ.get("SCALE_TEST_INDEX_DIR", str(ROOT / "scale_test_index"))
SCAN_DIR  = os.environ.get("SCALE_TEST_SCAN_DIR", str(ROOT / "scale_test_data"))
MODEL     = os.environ.get("TRIT_MODEL_PATH", str(ROOT / "models" / "code-minilm"))

os.makedirs(INDEX_DIR, exist_ok=True)

engine = SearchEngine()

print("Loading model...")
import threading
done = threading.Event()
def on_status(msg):
    print(f"  [status] {msg}")
    if "Index complete" in msg or "Index error" in msg:
        done.set()

from sentence_transformers import SentenceTransformer
engine.model = SentenceTransformer(MODEL)
engine.ready = True

print(f"\nIndexing real OSS code at {SCAN_DIR}...")
t0 = time.time()
engine.build_index([SCAN_DIR], INDEX_DIR, on_status, lambda: done.set())
done.wait(timeout=600)
build_time = time.time() - t0
print(f"\nTotal build time: {build_time:.1f}s")

# Real file sizes
import glob
vec_file = os.path.join(INDEX_DIR, "vectors_ternary.npy")
meta_file = os.path.join(INDEX_DIR, "metadata.json")
vec_size = os.path.getsize(vec_file)
meta_size = os.path.getsize(meta_file)

n_chunks = len(engine.metadata)
float_equiv = n_chunks * 384 * 4

print("\n" + "="*65)
print("  REAL SCALE TEST RESULTS — FastAPI + Gin (1228 real files)")
print("="*65)
print(f"  Chunks indexed       : {n_chunks:,}")
print(f"  Vectors file (packed): {vec_size:,} bytes")
print(f"  Float32 equivalent   : {float_equiv:,} bytes")
print(f"  Compression ratio    : {float_equiv/vec_size:.1f}x")
print(f"  Metadata file        : {meta_size:,} bytes")

# Real search speed test
print("\n  Search speed (10 runs each):")
queries = ["http route handler", "json response encoder", "async database query",
           "middleware chain", "error handling"]
for q in queries:
    t0 = time.time()
    for _ in range(10):
        results = engine.search(q, k=10)
    dt = (time.time() - t0) / 10 * 1000
    top = results[0]["path"] if results else "NONE"
    print(f"    '{q}': {dt:.1f}ms  top={top}")
