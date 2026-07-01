# Quality Cost of query_codebase's Token Savings

`token_reduction_findings.md` measured tokens saved (66.3% fewer than
search_code) but never measured whether that reduction costs you the actual
answer. This benchmark closes that gap with two objective (non-LLM-judged)
tests against the real horde-beta-version-1 codebase, at two levels of
strictness.

**Revision note:** the first pass of this benchmark only checked whether the
correct *file* appeared in results. A follow-up pass added a stricter
chunk-level check — does the returned preview text actually contain the
target function's name, not just the filename — and it found the real
picture is considerably worse than the file-level numbers suggested, for
*both* tools. See "File-level vs chunk-level" below; this is the more
important finding in this document.

## Method

Ground truth was hand-verified against the actual source (not guessed) via
direct code search. Two test types, each checked at two levels:

1. **Ground-truth recall** — for queries with a known correct file:
   - *File-level:* does the correct file appear anywhere in output?
   - *Chunk-level:* does the specific returned preview/snippet text contain
     the target function name, or did the tool surface the right file but
     an unrelated chunk from it?
2. **Secondary-match stress test** — targets a function that is *not* the
   most prominent one in its file (e.g. `heal()` living alongside the far
   more search-relevant `take_damage()` in the same file). This directly
   stresses query_codebase's two token-saving mechanisms: per-file dedup
   (keeps only the best-scoring chunk per file) and the 70%-of-top-score
   relevance cutoff. Checked at both levels as above.

## Results

### Ground-truth recall — file-level vs chunk-level

| Query | sc:file | sc:chunk | qc:file | qc:chunk |
|---|---|---|---|---|
| player health and damage handling | PASS | **MISS** | PASS | **MISS** |
| weapon firing and projectile logic | PASS | **MISS** | PASS | **MISS** |
| shop UI and upgrade purchasing | PASS | PASS | PASS | PASS |
| enemy AI state machine | MISS | MISS | MISS | MISS |
| save and load game state | n/a (no true positive) | | | |

**File-level:** search_code 3/4 (75%), query_codebase 3/4 (75%) — tied.
**Chunk-level: search_code 1/4 (25%), query_codebase 1/4 (25%) — also tied, but far lower.**

This is the central finding of this revision: **file-level recall drastically
overstates real accuracy for both tools.** Two of the three file-level
"passes" (player health, weapon firing) surfaced the correct file but a
chunk that never actually shows the target function — the model matched on
topical similarity (health-related variables, weapon-related helper code)
without the returned snippet containing `take_damage` or `shoot` at all.
Neither tool is meaningfully better than the other here — this is a shared
limitation of chunk-selection in the underlying search, not something
query_codebase's compression specifically caused.

Both tools missed "enemy AI state machine" entirely at the file level too —
the ground truth (`zombie.gd`'s `AIMode` enum and `set_ai_mode()`) doesn't
score highly enough semantically for either tool to surface it in the
top-10/top-8.

The "save and load game state" query has **no true positive in this codebase**
— verified directly: no `save_game()`/`load_game()` exists for player
progress, gold, or upgrades anywhere in gameplay code. The only persistence
code is in the third-party `addons/terrain_3d` editor plugin. Kept as an
informational negative control, not scored pass/fail.

### Secondary-match stress test — file-level vs chunk-level

| Query | Target function | sc:file | sc:chunk | qc:file | qc:chunk |
|---|---|---|---|---|---|
| heal and restore player health | `HealthComponent.gd` heal() | FOUND | FOUND | FOUND | **MISS** |
| spend gold currency | `game_manager.gd` spend_gold() | FOUND | **MISS** | FOUND | **MISS** |
| apply purchased upgrade stat to player | `player.gd` apply_upgrade() | FOUND | **MISS** | **MISS** | MISS |

**File-level: search_code 3/3 (100%), query_codebase 2/3 (67%).**
**Chunk-level: search_code 1/3 (33%), query_codebase 0/3 (0%).**

At the chunk level the gap widens further: query_codebase found the correct
*content* in zero of the three stress-test queries, versus search_code's one.
The "heal" query is the clearest illustration — both tools' file-level check
passed, but only search_code's actual returned preview text contains the
word "heal"; query_codebase's dedup kept a different, higher-scoring chunk
from the same file (almost certainly the more prominent `take_damage`
content) that doesn't mention healing at all.

### The regression, explained with the actual raw scores

Query: *"apply purchased upgrade stat to player"*
Target: `player.gd`'s `apply_upgrade()` function (line 282)

`search_code`'s raw top 10 (same embeddings, same index as query_codebase —
this isn't two different searches):

```
1. [6.973] shopui.gd
2. [6.041] game_manager.gd
3. [5.648] shopui.gd
4. [5.442] game_phase_script.gd
5. [5.226] upgrade_panel.gd
6. [4.395] shopui.gd
7. [4.278] zombie.gd
8. [4.204] iceturret.gd
9. [4.129] player.gd    <- counted as a PASS in the table above
10.[4.112] basegun.gd
```

`query_codebase`'s output, same query:

```
shopui.gd            6.97
game_manager.gd      6.04
game_phase_script.gd 5.44
upgrade_panel.gd     5.23
```

**The mechanism is arithmetic, not dedup.** query_codebase's cutoff keeps
anything scoring ≥ 70% of the top score. Top score here is 6.973, so the
cutoff is 6.973 × 0.7 = **4.881**. `player.gd` scored 4.129 — 0.75 points
below the line. It's excluded by the relevance cutoff alone; dedup never
enters into it here, because `player.gd` only had one chunk in the top 10
to begin with (there was nothing to collapse).

Why the cutoff bites here specifically: the score sequence
6.97→6.04→5.65→5.44→5.23→4.40→4.28→4.20→**4.13**→4.11 has no natural
gap — it decays smoothly. A fixed-70%-of-top threshold works well when
there's a real cliff between relevant and irrelevant results, but fails on
queries like this one where several files are all *somewhat* relevant
(upgrades touch shop UI, game manager, turrets, and player state all at
once) and the true answer sits partway down a long, gradually-decaying tail
rather than at a clean drop-off point.

### A caveat this benchmark did not originally catch: file-level match ≠ chunk-level match

Looking at the actual chunk text `search_code` returned for `player.gd`
(result #9 above):

```
=========== var max_health : float = 100.0 var health : float = 100.0
var _upgrade_bonuses : Dictionary = { "max_h...
```

That's a variable declaration block, **not the `apply_upgrade()` function
body**. Even the result originally counted as a PASS in the first-pass
recall table only named the right *file* — the specific chunk shown to an
LLM didn't contain the function that actually answers the query. This is
exactly why the chunk-level check was added: it distinguishes "found the
file AND the right chunk" from "found the file but a different, irrelevant
chunk from it" — and the results above show this distinction matters
enormously (75% file-level vs 25% chunk-level on ground truth, for both
tools equally).

This is exactly the predicted failure mode from `token_reduction_findings.md`'s
"Honest caveats" section: *"Dropping results below 70% of the top score and
collapsing to one chunk per file means query_codebase will sometimes miss a
relevant second match in the same file, or a legitimately-relevant but
lower-scoring result search_code would have surfaced."* This benchmark
confirms that caveat is not hypothetical — it reproduces on a real query
against this codebase, with an exact, traceable numeric cause. The
chunk-level pass additionally reveals a **second, larger problem that is not
specific to query_codebase at all**: the underlying chunking/embedding
search frequently surfaces the correct file with the wrong snippet, for both
tools, well before compression is even a factor.

## Combined result

| | search_code | query_codebase |
|---|---|---|
| Combined recall — file-level | 6/7 (86%) | 5/7 (71%) |
| Combined recall — chunk-level | 2/7 (29%) | 1/7 (14%) |

## What this means for using query_codebase safely

**The token savings are not free, and the file-level number understates the
real cost.** At the file level query_codebase trades ~15 percentage points
of recall (86% → 71%) for a 66.3% token reduction. At the chunk level — the
metric that actually predicts whether an LLM gets the right context to
answer correctly — the gap is proportionally similar (29% → 14%, roughly
half) but both absolute numbers are far lower than either tool's file-level
score suggests. Whether the file-vs-chunk gap matters in practice depends on
the use case:

- **Good fit:** broad exploratory searches, scanning many queries in one
  session, or when you'll follow up with a second more targeted query if
  the first doesn't find what you need.
- **Bad fit:** one-shot precision lookups where a file has multiple
  distinct relevant functions and you need the specific non-dominant one
  (e.g. "find the *secondary* effect of X" queries), or safety-critical
  code review where missing a match has real cost.

**Practical mitigation for the query_codebase-specific gap:** if
query_codebase returns fewer than ~4-5 results for a query that should
plausibly hit multiple files, that's itself a signal the relevance cutoff
may be over-pruning — falling back to search_code for that specific query
costs little (both are already fast local calls) and recovers the missed
file.

**Practical mitigation for the shared chunking gap (both tools):** neither
tool's chunk boundaries are guaranteed to include a full function body
starting at its signature — a chunk can start mid-function or capture
adjacent declarations instead of the actual logic. Widening the preview
window (more lines of context per result) or re-chunking on function
boundaries rather than fixed-size windows would improve both tools' true
chunk-level accuracy, independent of any compression mechanism.

## Does fixing the cutoff actually help? Tested, not assumed.

The obvious cheap fix for query_codebase's file-level regression is lowering
the 70%-of-top-score threshold. Tested directly with `trit_cutoff_sweep_test.py`
— a standalone reimplementation of the dedup+cutoff logic at 5 threshold
values, run against the same 7 queries, measuring both recall levels and
total tokens at each setting.

| Threshold | File recall | Chunk recall | Tokens | Token cost vs 0.70 |
|---|---|---|---|---|
| 0.70 (current) | 5/7 (71%) | 1/7 (14%) | 933 | baseline |
| 0.60 | 5/7 (71%) | 1/7 (14%) | 1,476 | +58% |
| **0.50** | **6/7 (86%)** | 1/7 (14%) | 1,539 | **+65%** |
| 0.40 | 6/7 (86%) | 1/7 (14%) | 1,539 | +65% |
| 0.30 | 6/7 (86%) | 1/7 (14%) | 1,609 | +73% |

**File-level recall recovers, at real cost.** Threshold 0.6 does nothing —
the missed `player.gd` chunk scored 4.129 against a 0.7-threshold cutoff of
4.881; it needed the bar to drop to 0.5 (cutoff 3.486) before clearing. At
that setting file recall matches search_code's 86% — but at the cost of
giving back roughly two-thirds of the token savings that motivated
query_codebase in the first place.

**Chunk-level recall does not move at all — 1/7 (14%) at every threshold
tested, including 0.3.** This is the key result: the cutoff only controls
*which chunks pass the filter*, not *what content is inside the chunks that
do pass*. A chunk that never contained the target function's name in the
first place still won't contain it after the threshold is loosened — loosening
the filter can only admit more chunks, not fix what's written inside them.

**Conclusion: cutoff-tuning is not worth pursuing on its own.** It is a real,
working lever for the narrow file-level regression, but a bad trade (65%+
more tokens for one recovered miss) and it is provably blind to the larger,
shared chunk-content problem. The fix that would actually move the number
that matters is re-chunking the index on function/class boundaries instead
of whatever fixed-size window it currently uses — see the chunker
investigation below.

Script: `trit_cutoff_sweep_test.py`

## Does re-chunking on function boundaries fix it? Tested — yes, dramatically.

The real chunker behind the live index (`trit_app.py:362-363`, inside
`build_index()`) is a blind character-window slide:

```python
for i in range(0, len(text), 700):
    chunk = text[i:i+800]
```

800 characters per chunk, 100-character overlap, zero awareness of function,
class, or line boundaries. The preview shown to an LLM at query time is not
cached — it's regenerated lazily as `text[offset:offset+120]` read live from
disk (`trit_app.py:288-292`). This means **the preview quality is entirely
determined by where the chunk's offset happens to land** — mid-function,
on a blank line, or (with luck) right at a signature.

### The test

`trit_rechunk_test.py` builds two fully in-memory indexes over
horde-beta-version-1 only (no disk writes, the live production index at
`~/.trit-search/index` is never touched):

- **OLD** — exact reproduction of the live chunker (blind 800-char / 700-stride windows)
- **NEW** — anchors chunk offsets at detected function/class signature lines
  (reusing the regex already written in `trit_embed_train.py`'s pair
  extraction), falling back to blind windowing for any text with no detected
  functions (configs, docs, leading imports)

Both use the same embedding model, same cosine search, same dedup/cutoff
logic, same 7 queries as the quality benchmark above. The only variable
changed is where chunk boundaries fall.

### Result

| | OLD (blind windows) | NEW (function-anchored) |
|---|---|---|
| File recall | 6/7 (86%) | 6/7 (86%) — unchanged |
| **Chunk recall** | **2/7 (29%)** | **6/7 (86%)** |
| Chunk count | 355 | 605 (+70%) |

| Query | OLD file | OLD chunk | NEW file | NEW chunk |
|---|---|---|---|---|
| player health and damage handling | PASS | MISS | PASS | **PASS** |
| weapon firing and projectile logic | PASS | MISS | PASS | **PASS** |
| shop UI and upgrade purchasing | PASS | PASS | PASS | PASS |
| enemy AI state machine | MISS | MISS | MISS | MISS |
| heal and restore player health | PASS | PASS | PASS | PASS |
| spend gold currency | PASS | MISS | PASS | **PASS** |
| apply purchased upgrade stat to player | PASS | MISS | PASS | **PASS** |

**Four of five previously-failing chunk-level queries are fixed, zero
regressions.** This is a dramatically larger effect than the cutoff sweep —
compare +57 percentage points here against +0 percentage points of chunk
recall from adjusting the threshold across five values. The one remaining
failure ("enemy AI state machine") fails identically on both chunkers — the
`AIMode` enum and `set_ai_mode()` don't score highly enough for this query
phrasing to reach the top of either index. That's an embedding/query-phrasing
limitation, not a chunking problem, and re-chunking correctly leaves it
unchanged rather than accidentally "fixing" it through noise.

**Why it works exactly as predicted:** since the preview is generated live
from `text[offset:offset+120]`, anchoring the offset at `func take_damage(`
means the preview *is* the function signature. Blind windowing has no such
guarantee — the same file, chunked differently, could show the signature,
the middle of the body, or an unrelated declaration a few lines above,
depending purely on where an arbitrary 700-character stride happens to land.

**The cost:** chunk count grew 70% (355→605) — every detected function now
gets its own anchor point instead of being folded into wider blind windows.
More embeddings to compute at index-build time, more index storage. Zero
cost at query time — same `k`, same cutoff, same tokens returned to the LLM
per query, since the fix only changes chunk *placement*, not how many
results are returned or how they're filtered.

**Conclusion:** this is the fix that actually moves the number that matters.
Cutoff-tuning (previous section) is a narrow, costly patch for one specific
symptom. Function-boundary chunking addresses the actual root cause — chunk
boundaries misaligned with the code structure being searched — and improves
both tools simultaneously, since neither's dedup/cutoff logic needs to
change at all. Not yet applied to the live production index; this was
validated as an isolated, reversible experiment first.

Script: `trit_rechunk_test.py`

## Reproducibility

Ground truth was independently re-verified via a second full codebase search
(GDScript search for take_damage/shoot/purchase_upgrade/AIMode/save patterns)
and matched the original findings exactly — file paths and line numbers
consistent across both searches.

Script: `trit_quality_benchmark.py`
