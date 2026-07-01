# Grep vs Semantic Search: When Does Each Actually Win?

`token_reduction_findings.md` compared `query_codebase` against `search_code`
and a "read full files" baseline — but never against the tool an assistant
actually reaches for by default: **Grep**. This closes that gap, split into
the two query shapes that behave completely differently under keyword
search, using the same real codebase and the same real token measurement
methodology (`tiktoken` on literal output) as the earlier benchmarks.

## Method

Two query sets against `horde-beta-version-1`:

1. **Keyword-friendly** — the caller already knows the exact identifier
   (e.g. querying `"take_damage"` directly). Grep should be strong here:
   one exact-match pass, no embedding model, minimal tokens for a precise hit.
2. **Concept-only** — the caller has only a natural-language description and
   does **not** know the exact function/variable name (e.g. "where does the
   player take damage" without knowing it's called `take_damage`). This
   models the realistic hard case: a grep-only workflow must guess keywords,
   run multiple exploratory searches, and pay the cumulative cost of every
   guess — including wrong ones — before finding the answer, if it ever does.

Grep was simulated faithfully: exact case-insensitive substring match with
±3 lines of context per hit (equivalent to `grep -rn -A3 -B3`), same file
extensions as the live index scans. `query_codebase` was called for real
against the actual OBSERVE index. Both measured with `tiktoken`'s
`cl100k_base` encoding on the literal output text.

## Results

### Keyword-friendly (exact term known)

| Term | grep tokens | grep found | query_codebase tokens | query_codebase found |
|---|---|---|---|---|
| take_damage | 1,049 | YES | 229 | YES |
| purchase_upgrade | 120 | YES | 106 | YES |
| spend_gold | 497 | YES | 106 | YES |
| set_ai_mode | 865 | YES | 188 | **NO** |
| apply_upgrade | 721 | YES | 107 | **NO** |
| **TOTAL** | **3,252** | **5/5** | **736** | **3/5** |

**grep wins outright.** `query_codebase` used 77% fewer tokens but missed
2 of 5 — including two cases where the query *was the literal function
name*. When you already know the identifier, cheaper-but-wrong is not a win.
Grep is the correct tool here, full stop.

### Concept-only (exact term NOT known — must guess)

| Query | grep (cumulative guesses) | grep found | query_codebase (1 call) | query_codebase found |
|---|---|---|---|---|
| where does the player take damage | 3,802 (3 guesses: hurt, damage, health) | NO — never found | 265 | **YES** |
| how does someone buy an upgrade in the shop | 320 (1 guess: buy) | YES | 36 | YES |
| where does gold get deducted when buying something | 1,183 (2 guesses: deduct, cost) | YES | 64 | NO |
| how does the enemy decide what to do | 1,137 (3 guesses: behavior, decide, state) | NO — never found | 110 | NO |
| apply a purchased stat boost to the player | 2,012 (3 guesses: boost, stat, apply_upgrade) | YES | 284 | YES |
| **TOTAL** | **8,454** | **3/5** | **759** | **3/5** |

Both tools found the same 3 of 5 — but grep needed an average of **2.4
sequential exploratory guesses per query** to get there, at 91% more total
tokens than `query_codebase`'s single natural-language call.

## Verdict — the real decision rule

Neither tool wins universally. The result depends entirely on whether the
caller already knows the exact identifier:

- **Know the exact name/term? Use Grep.** It is 100% reliable in this test
  and grep's higher token cost (~3,250 vs ~740 tokens) is a reasonable price
  for guaranteed correctness. `query_codebase`'s dedup/relevance-cutoff
  mechanism can drop a file even when the query is a dead-exact keyword
  match — this is a stronger warning than anything in
  `quality_benchmark_findings.md`, which only found misses on secondary,
  non-exact queries.
- **Only have a natural-language description? Use `query_codebase`.**
  Grep-only search requires guessing keywords and paying for every wrong
  guess sequentially; `query_codebase` gets equivalent recall in one call
  at a fraction of the cost specifically because it doesn't need the exact
  term to work.

## Why the keyword-friendly misses happen

`set_ai_mode` and `apply_upgrade` failing even as literal-keyword queries
traces to the same mechanism documented in `quality_benchmark_findings.md`'s
cutoff analysis: the embedding model doesn't treat an exact substring match
as an automatic top score — semantic similarity can still rank a different
file above the literal keyword's home file, and the 70%-of-top-score cutoff
then excludes the correct one. Grep has no such failure mode for an exact
term by construction; it either contains the substring or it doesn't.

## Practical guidance (now reflected in docstrings — see below)

The right default isn't "always use semantic search" or "always use grep."
It's: **grep first if you can name the thing you're looking for; fall back
to `query_codebase` when you can only describe it.** An assistant already
holding a specific identifier (from an error message, a stack trace, a
previous grep hit) should prefer Grep. An assistant starting from a vague
functional description with no known identifiers should prefer
`query_codebase`.

Script: `trit_grep_vs_semantic_test.py`
