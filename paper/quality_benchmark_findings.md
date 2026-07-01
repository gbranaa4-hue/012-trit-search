# Quality Cost of query_codebase's Token Savings

`token_reduction_findings.md` measured tokens saved (66.3% fewer than
search_code) but never measured whether that reduction costs you the actual
answer. This benchmark closes that gap with two objective (non-LLM-judged)
tests against the real horde-beta-version-1 codebase.

## Method

Ground truth was hand-verified against the actual source (not guessed) via
direct code search. Two test types:

1. **Ground-truth recall** — for queries with a known correct file, check
   whether that file appears anywhere in each tool's output.
2. **Secondary-match stress test** — targets a function that is *not* the
   most prominent one in its file (e.g. `heal()` living alongside the far
   more search-relevant `take_damage()` in the same file). This directly
   stresses query_codebase's two token-saving mechanisms: per-file dedup
   (keeps only the best-scoring chunk per file) and the 70%-of-top-score
   relevance cutoff.

## Results

### Ground-truth recall

| Query | search_code | query_codebase |
|---|---|---|
| player health and damage handling | PASS | PASS |
| weapon firing and projectile logic | PASS | PASS |
| shop UI and upgrade purchasing | PASS | PASS |
| enemy AI state machine | MISS | MISS |
| save and load game state | n/a (no true positive) | n/a (no true positive) |

**search_code: 3/4 (75%)  query_codebase: 3/4 (75%) — tied.**

Both tools missed "enemy AI state machine" equally — the ground truth
(`zombie.gd`'s `AIMode` enum and `set_ai_mode()`) apparently doesn't score
highly enough semantically for either tool to surface it in the top-10/top-8.
This is a shared limitation of the underlying embedding search, not something
query_codebase's compression introduced.

The "save and load game state" query has **no true positive in this codebase**
— verified directly: no `save_game()`/`load_game()` exists for player
progress, gold, or upgrades anywhere in gameplay code. The only persistence
code is in the third-party `addons/terrain_3d` editor plugin. Kept as an
informational negative control, not scored pass/fail.

### Secondary-match stress test

| Query | Target (secondary function) | search_code | query_codebase |
|---|---|---|---|
| heal and restore player health | `HealthComponent.gd` heal() | FOUND | FOUND |
| spend gold currency | `game_manager.gd` spend_gold() | FOUND | FOUND |
| apply purchased upgrade stat to player | `player.gd` apply_upgrade() | FOUND | **NOT FOUND** |

**search_code: 3/3 (100%)  query_codebase: 2/3 (67%) — real regression.**

### The regression, explained

Query: *"apply purchased upgrade stat to player"*
Target: `player.gd`'s `apply_upgrade()` function (line 282)

`player.gd` is dominated by `take_damage()` (line 237) — damage is a hot
path in this codebase and scores much higher on most semantic queries touching
that file. When the query asks about upgrades specifically, `search_code`
(which returns 10 uncollapsed results including lower-scoring chunks) still
surfaces the `player.gd` file because *some* chunk from it makes the top 10.

`query_codebase` collapsed to only 4 files returned (vs search_code's 10),
and evidently the best-scoring chunk it kept per file, combined with the 70%
relevance cutoff, excluded `player.gd` entirely — the query wasn't a strong
enough semantic match to the file's single best chunk (probably the
damage-related content) to clear the cutoff, even though the actual answer
(`apply_upgrade`) exists in that file.

This is exactly the predicted failure mode from `token_reduction_findings.md`'s
"Honest caveats" section: *"Dropping results below 70% of the top score and
collapsing to one chunk per file means query_codebase will sometimes miss a
relevant second match in the same file, or a legitimately-relevant but
lower-scoring result search_code would have surfaced."* This benchmark
confirms that caveat is not hypothetical — it reproduces on a real query
against this codebase.

## Combined result

| | search_code | query_codebase |
|---|---|---|
| Combined recall | 6/7 (86%) | 5/7 (71%) |

## What this means for using query_codebase safely

**The token savings are not free.** On this benchmark, query_codebase traded
roughly 15 percentage points of recall (86% → 71%) for a 66.3% token
reduction. Whether that trade is worth it depends on the use case:

- **Good fit:** broad exploratory searches, scanning many queries in one
  session, or when you'll follow up with a second more targeted query if
  the first doesn't find what you need.
- **Bad fit:** one-shot precision lookups where a file has multiple
  distinct relevant functions and you need the specific non-dominant one
  (e.g. "find the *secondary* effect of X" queries), or safety-critical
  code review where missing a match has real cost.

**Practical mitigation:** if query_codebase returns fewer than ~4-5 results
for a query that should plausibly hit multiple files, that's itself a signal
the relevance cutoff may be over-pruning — falling back to search_code for
that specific query costs little (both are already fast local calls) and
recovers the missed match.

## Reproducibility

Ground truth was independently re-verified via a second full codebase search
(GDScript search for take_damage/shoot/purchase_upgrade/AIMode/save patterns)
and matched the original findings exactly — file paths and line numbers
consistent across both searches.

Script: `trit_quality_benchmark.py`
