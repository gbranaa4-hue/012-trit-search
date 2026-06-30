"""
Tests ternary compression + search speed at realistic production scale
(50k and 500k chunks) using the same pack/unpack code as trit_app.py,
without needing to download large OSS datasets.
"""
import numpy as np
import time

def pack_ternary(trits):
    digits = (trits + 1).astype(np.int32)
    N, D = digits.shape
    pad = (-D) % 5
    if pad:
        digits = np.pad(digits, ((0, 0), (0, pad)), constant_values=1)
    G = digits.shape[1] // 5
    digits = digits.reshape(N, G, 5)
    weights = np.array([1, 3, 9, 27, 81], dtype=np.int32)
    return (digits * weights).sum(axis=2).astype(np.uint8)

def unpack_ternary(packed, orig_dim):
    N, G = packed.shape
    vals = packed.astype(np.int32)
    out = np.zeros((N, G, 5), dtype=np.int8)
    for i in range(5):
        out[:, :, i] = (vals % 3) - 1
        vals //= 3
    return out.reshape(N, G * 5)[:, :orig_dim]

DIM = 384

for n_chunks in [326, 5_000, 50_000, 500_000]:
    print(f"\n{'='*65}\n  N = {n_chunks:,} chunks (realistic embedding distribution)\n{'='*65}")

    rng = np.random.default_rng(42)
    float_vecs = rng.standard_normal((n_chunks, DIM)).astype("float32")
    float_vecs /= np.linalg.norm(float_vecs, axis=1, keepdims=True)

    t = 0.7 * np.abs(float_vecs).mean()
    trit_vecs = np.where(float_vecs > t, 1, np.where(float_vecs < -t, -1, 0)).astype("int8")

    t0 = time.time()
    packed = pack_ternary(trit_vecs)
    pack_time = time.time() - t0

    float_mb  = float_vecs.nbytes / 1e6
    int8_mb   = trit_vecs.nbytes / 1e6
    packed_mb = packed.nbytes / 1e6

    print(f"  Float32 size   : {float_mb:>10.2f} MB")
    print(f"  Int8 ternary    : {int8_mb:>10.2f} MB  ({float_mb/int8_mb:.1f}x)")
    print(f"  Packed ternary  : {packed_mb:>10.2f} MB  ({float_mb/packed_mb:.1f}x)")

    query = rng.standard_normal(DIM).astype("float32")
    query /= np.linalg.norm(query)

    # Float32 search (baseline FAISS-equivalent)
    t0 = time.time()
    for _ in range(10):
        sims = float_vecs @ query
        top  = np.argsort(-sims)[:10]
    float_search_ms = (time.time() - t0) / 10 * 1000

    # Packed-ternary search (unpack then dot)
    t0 = time.time()
    for _ in range(10):
        unpacked = unpack_ternary(packed, DIM)
        sims2    = unpacked.astype("float32") @ query
        top2     = np.argsort(-sims2)[:10]
    packed_search_ms = (time.time() - t0) / 10 * 1000

    print(f"\n  Pack time (one-time, at index build): {pack_time*1000:.1f}ms")
    print(f"  Float32 search latency  : {float_search_ms:>8.2f}ms")
    print(f"  Packed-ternary search   : {packed_search_ms:>8.2f}ms  "
          f"({'faster' if packed_search_ms < float_search_ms else 'SLOWER'} by "
          f"{abs(packed_search_ms/float_search_ms - 1)*100:.0f}%)")
