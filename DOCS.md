# 012 Ternary — Project Documentation

A unified system for ternary AI: compress, search, train, and run language models using weights in {-1, 0, +1} instead of float32. 16x smaller, hardware-friendly, fully local.

---

## Quick Start

```
python trit_app.py          Launch OBSERVE desktop search app
python trit_search.py --index --serve   Index files + start HTTP API
python trit_embed_train.py --train      Fine-tune search model on GitHub code
python trit_benchmark.py --compare      Benchmark baseline vs fine-tuned
```

---

## File Reference

### OBSERVE Desktop App
**`trit_app.py`**
Full GUI semantic code search application with Matrix/cyberpunk themes. Has its own independent indexing/search engine (`SearchEngine` class) — does **not** reuse `trit_search.py`, so the two implementations can drift if one is edited without the other (this has caused real bugs — see Known Issues below).

- 5 themes: Matrix (green), Cyberpunk (pink/cyan), Amber, Ice, Ghost
- Add directories, index codebase, search by meaning in real time
- Auto-search after 400ms typing pause
- Copy individual results or all results to clipboard
- Click filename to open in editor
- Config persisted at `~/.trit-search/config.json`

**Ternary compression — active, verified (2026-06-29):**
The index is now genuinely compressed, not just theoretically capable of it:
1. Embedding vectors quantized to {-1,0,+1} (`0.7 × mean(|values|)` threshold)
2. Bit-packed 5 trits/byte (3⁵=243 fits in a byte) → `vectors_ternary.npy`, stored on disk at the true ~20x ratio
3. **Unpacked once at load time** into a float32 array kept in RAM — this is the important part. Unpacking on every search call was tested and found to be 28-32x *slower* than float32 search at scale (measured at 326 / 5k / 50k / 500k synthetic chunks). Unpack-once-at-load gives full compression on disk with zero search-latency cost.
4. Metadata also restructured: paths deduplicated into a small table, full preview text dropped entirely and regenerated live from the original file on disk at display time (original files are never modified, so this is always safe and always fresh)

**Real measured result** (326 chunks, your horde-beta-version-1 game project):
| File | Before | After | Reduction |
|---|---|---|---|
| Vectors | 500,736 bytes (float32 equiv) | 25,230 bytes | 19.85x |
| Metadata | 56,047 bytes | 7,698 bytes | 7.3x |
| **Total** | **556,783 bytes** | **32,940 bytes** | **16.9x** |

Search correctness confirmed identical before/after (same top results, same scores) on every query tested.

**Known issues / gotchas:**
- Asymmetric quantization: the *stored* vectors are ternary, the *query* vector stays float32 — this is intentional (keeps one side full precision for better accuracy) but means raw dot-product scores are no longer 0-1 cosine similarity, they're in a different range (roughly 3-9 observed) — don't compare these scores directly to `trit_search.py`'s or `trit_benchmark.py`'s float32-vs-float32 percentages.
- Multiple app instances: launching `trit_app.py` repeatedly without closing prior windows leaves stale processes running old in-memory code even after the source file is edited — caused a real debugging detour this session. Always close prior windows (check `tasklist | grep python`) before relaunching after a code change.
- Legacy fallback: `load()`/`search()` still detect and load the old `faiss.index` format if `vectors_ternary.npy` doesn't exist, for backward compatibility with indexes built before this change.

Build exe:
```
G:\trit312\Scripts\python -m PyInstaller --onedir --noconsole --name OBSERVE trit_app.py
```

---

### MCP Server (Claude Code / Claude Desktop integration)
**`trit_mcp_server.py`**
Exposes OBSERVE's search as an MCP (Model Context Protocol) tool, so Claude Code, Claude Desktop, or any MCP-compatible client can call it directly from a conversation — no copy-pasting search results. Runs headless over stdio; wraps the exact same `SearchEngine` used by `trit_app.py` (same model, same compressed index, same search logic). Does not build an index itself — build one first with `trit_app.py` (INDEX CODEBASE) or `trit_search.py --index`.

**Install:**
```
pip install mcp
```

**Configure** (e.g. in Claude Code's MCP config, typically `.mcp.json` or via `claude mcp add`):
```json
{
  "mcpServers": {
    "observe": {
      "command": "python",
      "args": ["C:/path/to/012-ternary/trit_mcp_server.py"]
    }
  }
}
```
Verify it's connected with `claude mcp list`.

**Tools exposed:**
- `search_code(query, k=10, project_dir="")` — semantic search by meaning, not exact keywords. `project_dir` (optional, absolute path) scopes results to one indexed project's base directory — important if your index spans multiple codebases, otherwise results from unrelated projects can crowd out the one you actually want.
- `index_status()` — reports chunk count and active model, useful to check before searching whether the index is ready.

**`query_codebase(query, k, project_dir)`** — token-tight variant of `search_code`: dedups to one (best-scoring) chunk per file, drops results scoring below 70% of the top result instead of padding to a fixed k, and formats as one compact line per result. **Measured, not estimated** (see [paper/token_reduction_findings.md](paper/token_reduction_findings.md), `trit_token_benchmark.py`, real `tiktoken` counts on 5 queries against this repo's own index): 66.3% fewer tokens than `search_code` on the same queries, 98.7% fewer than the naive no-search-tool baseline (reading the top-3 matching files in full). This replaces an earlier unverified pitch that credited the savings to ternary compression and consensus-gate ranking — neither actually applies at this layer; the real mechanism is dedup + relevance cutoff + compact formatting, and it has a real cost (reduced recall — can miss a relevant second match in the same file or a legitimate lower-scoring result `search_code` would have surfaced).

**Known issues / recent fixes (2026-06-30):**
- The model/index loads lazily on first tool call via a background thread, with the synchronous tool call blocking (polling) until ready. Originally this could hang indefinitely with no useful feedback if the load thread stalled (e.g. after a session restart raced a partially-dead server process). Fixed: a 60s stall watchdog now detects no-progress loads, returns the last real status message instead of blocking silently, and resets so the *next* call retries with a fresh load rather than rejoining a dead wait.
- If the underlying Python process itself dies (not just a stalled thread — e.g. killed externally), the MCP client's connection is unrecoverable until the whole Claude Code session is restarted; no in-process fix can resurrect a dead process.
- Originally, `search_code` had no way to scope to one project — with multiple codebases indexed, results would mix in matches from unrelated projects (e.g. searching `take_damage` in one game project would surface near-identical code from a completely different game in `Downloads/`). Fixed via the `project_dir` parameter, which filters candidates by `base_dir` *before* top-k selection (filtering after would still let an unrelated, larger codebase crowd out the target project).

```
python trit_mcp_server.py    Standalone test (run headless, stdio transport)
```

---

### Semantic Search Engine
**`trit_search.py`**
Core search engine (CLI/HTTP API). Encodes files as 384-dim sentence-transformer vectors, stores in FAISS, searches by cosine similarity.
- Uses fine-tuned `models/code-minilm` if available, falls back to `all-MiniLM-L6-v2`
- `TritEncoder.encode(quantize=True)` supports ternary compression, but **`quantize` defaults to `False` and no call site overrides it** — this file's own index is still full float32, unlike `trit_app.py` which now has compression genuinely active (see above). The two indexing pipelines are independent and have diverged.
- HTTP API via Flask for external integrations
- Scans: `.py .gd .js .ts .cs .rs .go .c .cpp .h .java .md`
- Skips: `.git`, `node_modules`, FL Studio, StarCraft, AppData

```
python trit_search.py --index          Index your codebase
python trit_search.py --search "query" Search from terminal
python trit_search.py --serve          Start HTTP API on port 5000
```

---

### Embedding Fine-Tuner
**`trit_embed_train.py`**
Fine-tunes `all-MiniLM-L6-v2` on GitHub code using contrastive learning.
- Streams from HuggingFace — no full dataset download needed
- 50,000 pairs per language, 20 languages
- Extracts pairs: `function_name→body`, `comment→code`, `class_name→body`
- Local `.gd` files weighted 5x for GDScript specialization
- Loss: 0.46 → 0.09 over 23 minutes
- Result: `models/code-minilm/`

```
python trit_embed_train.py --train          Train on GitHub code
python trit_embed_train.py --train --local  Also include local .gd files
python trit_embed_train.py --test           Test the trained model
python trit_embed_train.py --install        Patch trit_search to use fine-tuned model
```

---

### Benchmarks

**`trit_benchmark.py`**
25 hard semantic triples where the wrong answer is deliberately similar.
Examples: `heal vs take_damage`, `draw_card vs play_card`, `training loss vs eval loss`.

Results:
| Model | Accuracy | Avg Margin |
|-------|----------|------------|
| Baseline MiniLM | 92% | +0.190 |
| Fine-tuned | 96% | +0.215 |
| Speed | 657ms | 176ms (3.7x faster) |

```
python trit_benchmark.py                Run full benchmark
python trit_benchmark.py --quick        20-query quick test
python trit_benchmark.py --compare      Side-by-side comparison
```

**`trit_oss_test.py`**
Downloads 6 real OSS projects (FastAPI, Godot, Redis, Tokio, Gin, godot-gdext), indexes 23,529 chunks, runs 30 search queries across all languages.

Results:
| Model | Accuracy |
|-------|----------|
| Baseline MiniLM | 76.7% |
| Fine-tuned | 92.0% (+15.3%) |

```
python trit_oss_test.py --download      Clone all OSS projects
python trit_oss_test.py --index         Index with both models
python trit_oss_test.py --benchmark     Run search benchmark
python trit_oss_test.py --all           Download + index + benchmark
```

**`benchmark_3way.py`**
Microsoft baseline vs your fine-tuned model vs a third independent model (`flax-sentence-embeddings/st-codesearch-distilroberta-base`, a community-built code-search model), all run on the same `trit_benchmark.py` triples.

Results (measured live, 2026-06-29):
| Model | Accuracy | Margin | Speed |
|---|---|---|---|
| Your fine-tuned | 96.0% | +0.215 | 34ms |
| Microsoft baseline | 92.0% | +0.190 | 219ms |
| CodeSearch-DistilRoBERTa (purpose-built, community) | 92.0% | +0.157 | 93ms |

Caveat: this is only a 25-triple benchmark you designed yourself, not an independent/blind test. Stronger commercial code-embedding models (Jina v2, Voyage-code-2) were not testable due to a `transformers` library version conflict with their custom model code.
```
python benchmark_3way.py
```

**`gpu_benchmark.py`**
Measures real matrix-multiply throughput (TFLOPS) on your GPU via live `torch.matmul` benchmarking, compared against published Nvidia datacenter specs.

Measured result (RTX 5060, 2026-06-29):
| Hardware | TFLOPS (FP16) |
|---|---|
| RTX 5060 (yours, measured) | 39 |
| A100 80GB | 312 (8x) |
| 200x A100 cluster | 62,400 (1,602x) |
```
python gpu_benchmark.py
```

**`int8_vs_ternary_test.py`**
Real head-to-head test: ternary vs INT8 vs float32, same CNN/CIFAR-10 setup. INT8 is the industry-standard quantization format with native hardware support (unlike ternary).

Measured result (15 epochs, RTX 5060, 2026-06-29):
| Format | Accuracy | Loss vs float32 | Compression |
|---|---|---|---|
| Float32 | 79.14% | — | 1x |
| Ternary | 71.44% | -7.70pp | 20.2x |
| **INT8** | **79.33%** | **+0.19pp (no loss)** | 4x |

**Important finding:** INT8 essentially matched float32 with zero accuracy cost, while ternary lost 7.7 points for 5x more compression. On this task, INT8 is strictly better on every practical axis (accuracy, hardware support) — ternary's only advantage is raw compression ratio, which only matters when storage size specifically is the binding constraint and accuracy loss is acceptable. This is worth weighing honestly against any claim that ternary compression is simply "better" — it isn't, except along one specific axis.
```
python int8_vs_ternary_test.py
```

**`precision_loss_test.py`**
Measures the real accuracy cost of ternary weight quantization (not embedding compression — model weights) by training matched StandardCNN/TernaryStandardCNN/mixed-precision variants on CIFAR-10.

Measured result (15 epochs, RTX 5060, 2026-06-29):
| Model | Accuracy | Lost vs float32 |
|---|---|---|
| Float32 | 79.55% | — |
| Full ternary | 72.20% | 7.35pp |
| Mixed precision (first+last layer float32) | 75.85% | 3.70pp |

Key finding: keeping the first and last layers in float32 cuts the accuracy loss roughly in half, at a small compression cost (~18x vs 20.2x). This is the standard "sensitive layers" technique from quantization literature, confirmed here on this specific architecture.
```
python precision_loss_test.py
```

**`scale_test.py`**
Synthetic benchmark (no download needed) testing ternary compression behavior at 326 / 5,000 / 50,000 / 500,000 chunk scales — both storage size and search latency.

Key finding: storage compression holds at ~19.9x regardless of scale, but **unpacking the bit-packed format on every search call is 28-32x slower than float32 search** at every scale tested. This is why `trit_app.py` unpacks once at load time instead of per-query (see OBSERVE section above) — discovered and fixed in this session.
```
python scale_test.py
```

**Real-scale confirmation (not synthetic), 2026-06-29:**
Cloned FastAPI + Gin (real GitHub repos, 2,570 files) and ran the actual `SearchEngine` from `trit_app.py` end-to-end — 65x larger than the original game-project index test.

| Metric | Result |
|---|---|
| Chunks indexed | 21,126 |
| Vectors: float32 equiv → packed-ternary | 32.45MB → 1.63MB |
| Compression ratio | **19.9x — matches synthetic prediction exactly** |
| Build time | 24.7s |
| Search latency | 6.7-10.9ms per query |
| Search correctness | Confirmed relevant top results (e.g. "middleware chain" → `middleware.md`) |

This confirms the unpack-once-at-load fix and the 19.9x compression both hold on real code at real scale, not just synthetic vectors.

---

### LoRA Fine-Tuning Pipeline

**`trit_lora.py`**
Fine-tunes Qwen2.5-coder:7b on your own data using LoRA adapters.
- Only trains ~1% of weights — adapter is ~50MB
- Base model stays frozen, can't be broken
- VRAM needed: ~4-5GB (fits RTX 5060 8GB)
- Output: `checkpoints/lora/`

**`trit_merge.py`**
Merges LoRA adapters into the base Qwen model, then quantizes.
- `merge_and_unload()` bakes adapter into base weights
- 4-bit NF4 quantization via BitsAndBytes (~4GB final size)
- Optional GGUF export for llama.cpp/Ollama
- Creates Ollama Modelfile for `ollama create`

```
python trit_merge.py --merge             Merge LoRA into base
python trit_merge.py --quantize          Quantize to 4-bit
python trit_merge.py --merge --quantize  Both steps
python trit_merge.py --test              Run 3 test prompts
```

---

### Training Pipelines

**`trit_mega_train.py`**
Continual learning across Wikipedia + 10 major code languages via HuggingFace streaming. Never needs more than ~500MB disk at once. Cycles through: Wikipedia → Python → JavaScript → C/C++ → C# → GDScript → Rust → Go → Java → TypeScript.

**`trit_distill.py`**
Knowledge distillation: Qwen2.5:7b (teacher via Ollama) → TritLM (student).
Generates diverse training text from Qwen, trains your ternary architecture on it.

```
python trit_distill.py --generate    Generate training data from Qwen
python trit_distill.py --train       Train TritLM on generated data
python trit_distill.py --chat        Chat with trained model
```

---

### Ternary Language Model

**`trit_lm.py`**
Character-level language model using triadic attention. Every weight is {-1, 0, +1}.

Architecture:
- Token embedding (ternary)
- N × TritAttentionBlock: triadic QKV projection + consensus attention + TritMemoryCell + TernaryFFN
- TernaryLinear → vocab logits

Trained on Shakespeare. Compared against GPT-mini at same parameter budget.

**`trit_rag.py`**
Retrieval-Augmented Generation using ternary embeddings.
- Query → TritEncoder (256-dim trit vector) → FAISS → retrieved facts → TritLM → answer
- Scales to 10M facts on GPU with FAISS flat index, 1B facts on disk with IVF

**`trit_memory_store.py`**
Hopfield-style content-addressable memory using ternary weights.
- Store facts as trit patterns, query with partial/noisy input
- Consensus gate: `sign(a + b + c)` — majority vote reconstructs stored fact
- Different from TritLM: stores discrete facts, not generative

---

### Quantization & Compression

**`ternary_quant.py`**
Post-training ternary quantization for any HuggingFace model.
- Converts weights to {-1, 0, +1} without retraining
- Default target: Qwen2.5-7B (14GB → 1.4GB VRAM)
- Optional few-hundred-step fine-tune to recover quality

---

### Resonance Hypothesis Test Series (2026-06-29)

A proposed hybrid architecture combined the hardware `consensus_gate.sv` primitive (`sign(v0+v1+v2)`, also used in `trit_os_sim.py`'s scheduler) with a biologically-inspired "resonating cell" — a slow low-pass filter meant to integrate noisy signals over time. A follow-on proposal extended this to a "shaped resonant cavity" for tunable symmetry control. Five honest, falsifiable tests were run to find out where (if anywhere) any of this actually helps. All five are real, measured, run-on-this-machine results, not theoretical claims.

**`trit_cache_eviction.py`** — Consensus-gate vs LRU cache eviction (no resonance yet)
Baseline test: does 3-trit-vote eviction (recency/frequency/fairness) beat standard LRU on a Zipfian access trace?
| Cache size | LRU | Consensus-gate |
|---|---|---|
| 10/20/50 | 48-79% | 47-76% (loses by 0.9-2.9pp at every size) |

Result: **LRU wins outright.** Plain consensus voting doesn't beat a specialized single-signal algorithm when that signal (recency) is already near-optimal for the task.

**`trit_resonant_cache.py`** — Adds resonance (low-pass filtered recency) to the same cache test, plus noise injection (decoy accesses)
| Noise rate | LRU | Consensus-gate | Resonant-consensus |
|---|---|---|---|
| 0-40% | 29-62% | -1.2 to -2.2pp vs LRU | **-4.2 to -5.9pp vs LRU (worse)** |

Result: **Hypothesis disproven for this task.** Resonance made things *worse*, not better, at every noise level — smoothing blurred a signal that was already clean and didn't need filtering.

**`trit_adaptive_scheduler.py`** — Tests resonance on a genuinely time-varying signal (a process becoming urgent for a 100-tick window, then not)
| Scheduler | Before | During urgency | After | Ramp-up |
|---|---|---|---|---|
| Static round-robin | 25.0% | 25.0% | 25.0% | +0.0pp |
| Direct-signal consensus | 25.0% | 41.0% | 25.0% | +16.0pp |
| **Resonant consensus** | 25.0% | **71.0%** | 29.0% | **+46.0pp** |

Result: **Hypothesis confirmed — resonance genuinely wins here.** Sparse, intermittent urgency signals need integration over time to be actionable; direct reaction barely registers, resonant accumulation builds a strong, stable, correctly-decaying bias. (Note: an earlier version of this test had a bug — anti-starvation voting thresholds coincidentally matched the natural round-robin period for 4 processes, forcing identical round-robin output for all three schedulers regardless of urgency. Fixed by raising the starvation threshold so it only fires for genuine neglect, not routine rotation.)

**`trit_npc_consensus_test.py`** — Independent follow-up application: does the bare consensus-gate primitive (no neural net, no compression) improve on a *real game's* existing decision logic? Tested against `tribe/npc.gd`'s actual fight-or-flee code (a 2-signal OR-gate override chain) by adding one new signal and combining via majority vote instead. Result: **clean win**, +1.81pp accuracy (0.7385 vs 0.7204), t=71.69, 30/30 seeds. See [paper/npc_consensus_findings.md](paper/npc_consensus_findings.md) for the full test design and honest caveats (some of the gain is from the extra signal, not the voting rule alone; not yet wired into the live game).
```
python trit_npc_consensus_test.py
```

**`trit_order_acceptance_test.py`** — Round 2 of the same question, on a structurally different decision: `tribe/tribemember.gd`'s order-acceptance logic (`give_order()`), which is already a weighted-sum linear threshold (loyalty + courage vs risk) rather than an OR-gate. Predicted *before running* that consensus voting would lose here (discretizing continuous evidence into booleans throws away the margin information a threshold already uses) — confirmed: **clean loss**, -4.62pp accuracy (0.8270 vs 0.8732), t=-107.31, 30/30 seeds favor the existing linear threshold. See [paper/order_acceptance_findings.md](paper/order_acceptance_findings.md). Together with the fight/flee win, this scopes the consensus-gate precisely: helps when combining genuinely separate signals (especially when adding new information), hurts when replacing a rule that already has the continuous combination in hand.
```
python trit_order_acceptance_test.py
```

**`trit_tmr_test.py`** — Third external check of the same scoping rule, against classical Triple/N-Modular Redundancy (von Neumann's `R_TMR = 3R²-2R³` reliability theory) and weighted-majority/Bayesian sensor fusion. 5 independent binary units, calibrated once, tested stable vs. drifted (silently shifted reliability, unknown to the decoder). Result: **confirmed** — weighted log-odds fusion wins under stable calibration (+2.76pp, t=7.04), and the advantage collapses to statistical noise under drift (t=-0.31), the predicted "wins or closes the gap" pattern. See [paper/tmr_findings.md](paper/tmr_findings.md) — softer than the population-coding test's catastrophic-collapse result, because TMR's drift here is gradual rather than an acute outlier, a real and useful refinement of the rule.
```
python trit_tmr_test.py
```

**`trit_ensemble_test.py`** — Fourth check, against real scikit-learn classifiers (LogisticRegression/DecisionTree/KNN/GaussianNB/RandomForest) under covariate shift, not another synthetic simulation. Result: **weak/inconclusive** — weighted voting wins regime A only marginally (+0.72pp, t=2.14) and the gap does *not* collapse under shift (+0.55pp, t=1.73) as predicted. See [paper/ensemble_ml_findings.md](paper/ensemble_ml_findings.md) — reported as run, no retuning to chase a cleaner result. Sharpens the rule: degradation likely depends on whether miscalibration *reorders relative reliability ranking*, not just whether absolute accuracy drops — this shift may have lowered everyone's accuracy together without reordering who's most trustworthy.
```
python trit_ensemble_test.py
```

**`trit_resonant_memory.py`** — Tests resonance on noisy-glimpse memory retrieval (real `TernaryHopfield`, synthetic random trit patterns standing in for facts, skipping the 500-epoch neural encoder for speed)
| Noise | Single-shot | Instant-vote | Resonant (pre-filter) |
|---|---|---|---|
| 40-60% | 88-96% | **100%** | 94-100% (loses to vote at 50-60%) |

Result: **Resonance loses to simple unweighted averaging.** With K independent equally-reliable glimpses of one static fact, instant majority-vote is close to the statistically optimal estimator — recency-weighting (favoring later glimpses) actively discards good information for no benefit, since there's no real temporal structure to exploit.

**`trit_resonant_hopfield.py`** — Corrected-hypothesis retest: puts resonance *inside* the Hopfield settling process (interleaved EMA + energy-minimization per step) instead of as a pre-filter
| Noise | Standard | Instant-vote | Resonant-settle |
|---|---|---|---|
| 40-50% | 94-96% | 100% | **100% (tied with vote — improved from pre-filter version)** |
| 60% | 88% | 100% | 98% (still 2pp behind) |

Result: **Mechanism placement matters, but doesn't change the underlying conclusion.** Interleaving resonance with settling clearly beat the cruder pre-filter version, but still didn't surpass plain averaging — confirming that when a signal is genuinely static, no amount of resonance sophistication beats simple unweighted pooling.

**`trit_symmetry_cavity_test.py`** — Tests a follow-on proposal: a "shaped resonant cavity" giving the triadic mixing weights as a *tunable* parameter (fixed-symmetric, fixed-asymmetric, adaptive/learned, input-driven) instead of the original fixed formula (`s1*(1-s0)+s2*s0`). Skips the proposal's speculative optical-cavity physics (Fano resonance, complex-valued mode phases, unverifiable citations) and tests only the concrete falsifiable claim. Trained on CIFAR-10, 15 epochs, same rotation-robustness harness as `experiments.py`.
| Mode | Accuracy @0° | Norm. stability (lower=better) |
|---|---|---|
| baseline_original (existing fixed formula) | **76.97%** | 68.97% |
| fixed_symmetric | 72.41% | 68.82% |
| fixed_asymmetric | 70.70% | 66.61% (best stability) |
| adaptive | 74.74% | 69.99% (worst) |
| input_driven | 75.76% | 69.23% |

Result: **the proposal's specific prediction was wrong.** It predicted adaptive/input-driven would beat both fixed presets; instead adaptive had the *worst* stability of all 5 variants, and input-driven also lost to baseline. The only variant that improved stability (`fixed_asymmetric`) did so at a 6.27pp accuracy cost — a real tradeoff, not a clean win. The original fixed formula you already had remains the best on raw accuracy and is competitive on stability — added tunability did not help here, consistent with the overall pattern below.

```
python trit_symmetry_cavity_test.py
```

**Overall finding across all five tests:** resonance/temporal-integration and tunable mode-mixing help specifically and only when the underlying signal **genuinely changes over time** (the scheduler case). They do not help — and sometimes actively hurt — when the signal is static or already clean (cache recency, repeated noisy reads of one fixed fact, the fixed triadic mixing formula on rotation robustness), where the established simple baseline (LRU, unweighted averaging, the original `s1*(1-s0)+s2*s0` formula) is already close to optimal. This is a precise, falsifiable, five-test-deep characterization, not a hand-wave — and it cuts against the original framing that added flexibility/resonance is a generally beneficial addition.

**Cross-substrate follow-up:** [paper/cross_substrate_symmetry_findings.md](paper/cross_substrate_symmetry_findings.md) lines this project's `trit_symmetry_cavity_test.py` result up against an independent acoustic-MEMS-plate reservoir-computing study (separate project, separate codebase) that found a much cleaner symmetry-breaking win — and proposes a third substrate (a Spikeling software resonator bank) to test whether the effect generalizes across coupled-oscillator systems or is specific to each substrate's own mechanism.

---

### Research & Experiments

**`experiments.py`**
The core ternary AI learning system. Trains and benchmarks 7 model variants on CIFAR-10 and STL-10 to isolate exactly what makes ternary triadic networks robust.

**The 3 key classes:**

`TernaryQuantize` — the quantization rule:
```
threshold = 0.7 × mean(|weights|)
weight → +1 if w > threshold
weight →  0 if |w| <= threshold   (skip — no computation)
weight → -1 if w < -threshold
```
Gradient uses Straight-Through Estimator (STE) so backprop still works.

`PredictiveTritBlock` — the triadic processing unit:
```
stream_0 (1×1 conv): Observer gate    → sigmoid [0,1]
stream_1 (3×3 conv): Shadow features  → tanh [-1,1]
stream_2 (5×5 conv): Light context    → tanh [-1,1]
output = stream_1 × (1 - gate) + stream_2 × gate
```
Three parallel streams at different scales, gated by the observer. Hardware equivalent: `consensus(s0, s1, s2) = sign(s0+s1+s2)`.

`TritCognition` — the full model:
- 3 × PredictiveTritBlock (32→64→128 channels)
- Spatial attention: 128→32→1 sigmoid map
- Memory gate: σ(W·features) — learned feature selection
- Predictive coding loss: each block predicts its own input (self-supervision)

**7 ablation variants trained:**

| Model | What it tests |
|-------|--------------|
| ResNet18 | float32 baseline |
| StandardCNN | same depth, float32, no triadic |
| TernaryStandardCNN | ternary weights, no triadic structure |
| TritFull | full 012 stack |
| Trit-NoPredLoss | remove predictive coding (λ=0) |
| Trit-NoAttention | remove spatial attention |
| Trit-NoMemoryGate | remove memory gate |

**Key finding:** TernaryStdCNN ≈ StandardCNN in rotation robustness → ternary weights alone don't cause robustness. TritFull > TernaryStdCNN → the **triadic structure** is the cause.

**Energy numbers (28nm CMOS, Andri et al. 2018):**
- Binary MAC: 4.6 pJ (multiply + add)
- Ternary op: 1.1 pJ (add/subtract only, no multiply)
- ~75% energy saving per MAC at 53.5% active weights

```
python experiments.py    Train all 7 models, print full results table + ablation summary
```

**`memory_model.py`**
Extends TritCognition with TritMemoryCell — persistent state across video frames and sensor streams.

**`noise_recovery.py`**
Tests memory as error correction: clean frame 1 + corrupted frame 2 → does memory help recover accuracy?
Noise types: Gaussian, salt+pepper, blur, occlusion (30%), combined.

**`stream_proof.py`**
Five formal proofs:
1. Fixed memory: TritLM stays 2KB, GPT grows linearly
2. Streaming speed: TritLM constant-time, GPT quadratic
3. Forgetting resistance: recall of A after learning B
4. Noise robustness: corrupted tokens + memory from clean context
5. Memory interpretability: which trit cells encode which patterns

**`personal_lm.py`**
TritLM trained on your own writing — emails, notes, code, chat logs.
Learns your vocabulary and style. 38KB model. No API calls.

```
python personal_lm.py --train
python personal_lm.py --chat
python personal_lm.py --complete "start typing and it finishes"
```

**`trit_transformer.py`**
Applies the triadic Observer/Shadow/Light gate to transformer attention (replaces Q/K/V with `s0/s1/s2` streams), trained with a predictive-coding auxiliary loss. Trains `TritGPT` vs `GPTMini` (standard transformer) on Shakespeare at matched parameter budget — tests whether triadic gating helps *training*, not just compression.
```
python trit_transformer.py --train       Train both, compare final/early-step loss
python trit_transformer.py --ablation    4 ablation variants (full/no-pred/float32/baseline)
python trit_transformer.py --chat        Chat with trained TritGPT
```

**`trit_triadic_encoder.py`**
A from-scratch sentence embedding model using the triadic attention block — no pretraining, unlike MiniLM. Trained on 1,777 local code pairs (60 seconds), then benchmarked against MiniLM baseline/fine-tuned on the same `trit_benchmark.py` triples.

Measured result (2026-06-29): **60.0% accuracy** — well below Microsoft baseline (92%) and the fine-tuned MiniLM (96%). Confirms that triadic architecture alone does not substitute for pretraining scale/data — the gains seen in `experiments.py`'s CIFAR-10 ablations don't automatically transfer to a from-scratch text encoder trained on this little data.
```
python trit_triadic_encoder.py --train
python trit_triadic_encoder.py --benchmark
```

---

### Hardware

**`hardware/`**
SystemVerilog RTL implementation of ternary compute:
- `trit_add.sv` — ternary adder
- `trit_mac.sv` — multiply-accumulate (no multiplications, just additions)
- `trit_not.sv` — ternary NOT gate
- `trit_register.sv` — ternary register
- `triadic_pe.sv` — triadic processing element
- `consensus_gate.sv` — majority vote gate
- `012_top.sv` — top-level chip design
- `testbench.sv` — simulation testbench
- `vivado_synth.tcl` — Xilinx synthesis script
- `ternary_rtl.py` — Python RTL simulator

---

### Build & Distribution

**`build_all.py`**
Cross-platform build script using PyInstaller.
- Windows: `--onefile --noconsole` → `dist/windows/OBSERVE.exe`
- Mac: `--onedir --windowed` → `dist/mac/OBSERVE.app` + DMG
- Linux: `--onefile` → `dist/linux/OBSERVE`
- Generates GitHub Actions workflow for automated CI builds

**`trit_install.sh`**
One-command Mac installer for dad/external users.
- Creates venv at `~/.trit-search/`
- Creates Desktop shortcut `TritSearch.app`
- Runs initial index automatically

**`.github/workflows/build.yml`**
GitHub Actions: triggers on `git tag v*`, builds all 3 platforms, uploads to GitHub Releases.

---

## Architecture Summary

```
Your files
    ↓
trit_search.py  (scan + chunk)
    ↓
trit_embed_train.py  (fine-tuned MiniLM encodes chunks → 384-dim vectors)
    ↓
FAISS index  (cosine similarity search)
    ↓
trit_app.py  (OBSERVE GUI — search by meaning)
```

```
Qwen2.5-coder:7b (base)
    ↓
trit_lora.py  (LoRA fine-tune on your data)
    ↓
trit_merge.py  (merge + 4-bit quantize → ~4GB)
    ↓
Ollama / llama.cpp  (run locally forever)
```

```
TritLM (ternary weights)
    ↓
trit_distill.py  (learn from Qwen teacher)
    ↓
trit_rag.py  (retrieve + generate)
    ↓
trit_memory_store.py  (Hopfield fact storage)
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Search accuracy (fine-tuned), real OSS code | 92.0% (vs baseline 76.7%) |
| Search accuracy, hard-triple benchmark | 96.0% (vs baseline 92.0%, vs community code-search model 92.0%) |
| Search speed (benchmark) | 34ms (fine-tuned) vs 219ms (baseline) |
| Fine-tuning training time | 23 minutes, $0 |
| **OBSERVE index compression (verified, active)** | **16.9x total (19.85x vectors, 7.3x metadata)** |
| Ternary weight quantization accuracy cost | 7.35pp lost (full), 3.70pp lost (mixed precision) |
| Bit-packed-on-every-search penalty | 28-32x slower — fixed via unpack-once-at-load |
| Your GPU raw compute (measured) | 39 TFLOPS FP16 (RTX 5060) — 1,602x less than a 200-A100 cluster |
| LoRA adapter size | ~50MB |
| Final quantized LLM model | ~4GB |
| Triadic-architecture-only (no pretraining) text encoder | 60.0% accuracy — underperforms pretrained baseline |
| Supported languages | 20+ |

*Last verified: 2026-06-29, on this machine (RTX 5060), with live re-runs — not cached/historical claims.*
