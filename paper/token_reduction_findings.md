# query_codebase: real, measured token reduction (not the pitched mechanism)

This replaces an unverified claim ("MCP wrappers reduce token usage by
70-97%, via ternary compression + consensus-gate ranking") with a real
measurement on this actual codebase, using `tiktoken`'s `cl100k_base`
encoding on literal tool output strings — not estimated, not asserted.

## Correcting the original pitch first

Two of the three claimed mechanisms don't actually apply at this layer:

- **"16.9x ternary compression reduces tokens"** — false causal link.
  That compression is disk storage for the embedding *index*
  (`vectors_ternary.npy` vs float32), entirely server-side, before any
  query runs. It has zero effect on how many tokens are sent back to an
  AI assistant in conversation — that's determined by how many results
  you return and how long each preview is, not how the index is packed
  on disk.
- **"Consensus-gate wins on multi-signal arbitration, use it to rank
  results"** — backwards application of this project's own findings.
  `order_acceptance_findings.md` and `ensemble_ml_findings.md` establish
  that weighted/continuous combination beats voting specifically when a
  calibrated continuous score already exists. Search relevance ranking
  *already has* a continuous similarity score — exactly the regime where
  our own research says voting is the wrong tool, not the right one.
- **"70-97% reduction (the mcp-compressor result)"** — unsourced,
  unverified, not something this project measured. Not used as a number
  anywhere below.

## What `query_codebase` actually does (the real mechanism)

Three concrete, unglamorous techniques, none involving compression or
consensus-gating:
1. **Dedup per file** — keep only the best-scoring chunk per file instead
   of letting multiple chunks from the same file pad out the result list.
2. **Relevance cutoff** — drop results scoring below 70% of the top
   result's score, instead of always padding out to a fixed `k` with
   low-relevance noise.
3. **Compact formatting** — one line per result (`path  score  preview`,
   90-char preview) instead of `search_code`'s two-line,
   header-and-blank-line format.

## Measured result

5 real queries against this repo's own index (horde-beta-version-1, 58,224
chunks), comparing three real outputs: the naive no-search-tool baseline
(read the top-3 files `search_code` itself found relevant, in full —
the realistic alternative an assistant without a search tool would take),
`search_code`, and `query_codebase`.

| Query | naive (top-3 full files) | search_code | query_codebase |
|---|---|---|---|
| player health and damage handling | 13,156 | 472 | 177 |
| weapon firing and projectile logic | 1,675 | 445 | 221 |
| shop UI and upgrade purchasing | 36,810 | 490 | 36 |
| enemy AI state machine | 1,342 | 466 | 63 |
| save and load game state | 7,326 | 448 | 285 |
| **TOTAL** | **60,309** | **2,321** | **782** |

- **query_codebase vs naive (read full files): 98.7% fewer tokens**
- **query_codebase vs search_code: 66.3% fewer tokens**
- **search_code vs naive: 96.2% fewer tokens**

## Honest caveats

- **The naive-baseline comparison is real but noisy** — it's dominated by
  how big the top-matching files happen to be (the "shop UI" query hit a
  490-line file, producing a 36,810-token naive baseline; "enemy AI"
  query's top files were small, giving only 1,342). That's expected and
  realistic (file sizes really do vary that much), but it means the
  *naive* comparison's percentage swings a lot query to query — the
  **search_code vs query_codebase comparison (66.3%) is the fairer,
  more stable number**, since both already use the same semantic search
  and only differ in dedup/cutoff/formatting.
- **This is not free — it's a real recall/completeness tradeoff.**
  Dropping results below 70% of the top score and collapsing to one
  chunk per file means `query_codebase` *will* sometimes miss a
  relevant second match in the same file, or a legitimately-relevant
  but lower-scoring result `search_code` would have surfaced. That's
  the actual cost being traded for the token reduction — not
  mentioned in the original pitch, which framed this as a pure win.
- **5 queries, one codebase, one session.** This is a real, reproducible
  measurement (`trit_token_benchmark.py`), not a large-scale study —
  it should be read as "this works on this codebase, by this much," not
  as a universal guarantee.

## Follow-up: does the token reduction cost you the answer?

See [quality_benchmark_findings.md](quality_benchmark_findings.md) for a
direct measurement. Short version: yes, sometimes. On a 3-query stress test
targeting secondary functions living in the same file as a more prominent
one, query_codebase's dedup+cutoff missed 1/3 (67% recall) where search_code
caught all 3 (100%). Combined with a tied ground-truth recall test (75%
each), the overall measured recall was 86% (search_code) vs 71%
(query_codebase) — a real, non-hypothetical cost for the token savings,
not just a caveat.

## Verdict

The actual savings (66-99% depending on what you compare against) are
in the same ballpark as the pitched 70-97% number — but arrived at
honestly, through measurement, via mechanisms (dedup, relevance cutoff,
compact formatting) that have nothing to do with the compression or
consensus-gate machinery the original pitch credited. Worth using and
citing with the real number and the real mechanism, not the inflated
story.
