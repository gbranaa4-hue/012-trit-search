# 012 Ternary — Complete File Reference

Every file in this project, what it does, and how to run it. For the high-level architecture summary see [DOCS.md](DOCS.md).

---

## Root Directory

### Core Research

**`experiments.py`**
The core ternary AI learning system. Trains 7 model variants (ResNet18, StandardCNN, TernaryStandardCNN, TritFull, and 3 ablations) on CIFAR-10 + STL-10 to isolate what makes triadic ternary networks robust to image rotation. Contains `TernaryQuantize` (the {-1,0,+1} quantization rule), `PredictiveTritBlock` (the Observer/Shadow/Light triadic gate), and `TritCognition` (the full model). Key finding: ternary weights alone don't cause robustness — the triadic structure does.
```
python experiments.py
```

**`trit_transformer.py`**
Applies the triadic Observer/Shadow/Light gate to transformer attention instead of standard Q/K/V. Trains `TritGPT` (triadic) vs `GPTMini` (standard) on Shakespeare character-level next-token prediction at matched parameter budget, with a predictive-coding auxiliary loss. Tests whether triadic gating improves training, not just compression.
```
python trit_transformer.py --train       Train both models, compare
python trit_transformer.py --ablation    Train all 4 ablation variants
python trit_transformer.py --chat        Chat with trained TritGPT
```

**`trit_triadic_encoder.py`**
A from-scratch sentence embedding encoder using the triadic attention block (no pretraining, unlike MiniLM). Trained on local code pairs with contrastive loss, then benchmarked against MiniLM baseline/fine-tuned using `trit_benchmark.py`'s test set. Result: 60% accuracy — confirms architecture alone doesn't beat a pretrained model without comparable data/scale.
```
python trit_triadic_encoder.py --train
python trit_triadic_encoder.py --benchmark
```

**`memory_model.py`**
Extends the ternary architecture with `TritMemoryCell` — persistent recurrent state across a sequence of inputs (video frames, sensor streams). Tests whether memory improves accuracy on sequential/temporal data versus frame-by-frame processing.

**`noise_recovery.py`**
Tests whether memory acts as error correction: if frame 1 is clean and frame 2 is corrupted (Gaussian noise, salt+pepper, blur, occlusion, or all combined), does carrying memory from frame 1 help recover accuracy on frame 2? Simulates real-world sensor degradation (camera noise, transmission errors, motion blur, physical obstruction).

**`stream_proof.py`**
Five formal proofs comparing TritLM against a standard GPT-style model: (1) fixed memory footprint vs linear growth, (2) constant-time streaming vs quadratic slowdown, (3) forgetting resistance — recall A after learning B, (4) noise robustness, (5) interpretability of which memory cells encode which patterns.

**`personal_lm.py`**
A tiny (38KB) TritLM trained on your own writing — emails, notes, code, chat logs. Learns your personal vocabulary and style, runs fully offline, no API calls.
```
python personal_lm.py --train
python personal_lm.py --chat
python personal_lm.py --complete "start typing and it finishes"
```

**`ternary_quant.py`**
Post-training ternary quantization for any HuggingFace model — converts existing pretrained weights to {-1, 0, +1} without retraining from scratch. Default target Qwen2.5-7B-Instruct (14GB → ~1.4GB). No fine-tuning needed, though a short recovery fine-tune is optional.

---

### MCP Server (Claude Code / Claude Desktop integration)

**`trit_mcp_server.py`**
Exposes OBSERVE's search as an MCP tool over stdio, so Claude Code/Desktop can call `search_code` and `index_status` directly from a conversation. Wraps the same `SearchEngine` as `trit_app.py` — same model, same index, no separate build step (index it first via `trit_app.py` or `trit_search.py --index`). `search_code` takes an optional `project_dir` to scope results to one indexed codebase when multiple are indexed together. See DOCS.md for setup, config JSON, and known issues/fixes.
```
pip install mcp
python trit_mcp_server.py   # standalone test
```

---

### Search Engine ("OBSERVE")

**`trit_search.py`**
The core semantic search engine. Scans your files, chunks them, encodes with sentence-transformers (fine-tuned MiniLM if available), indexes in FAISS, and serves results via terminal, HTTP API, or the OBSERVE GUI. Has its own `SKIP_DIRS`/`INCLUDE_ONLY_DIRS` config — note: `trit_app.py` has a **separate, duplicate** skip-list used when indexing through the GUI.
```
python trit_search.py --index
python trit_search.py --search "query"
python trit_search.py --serve
```

**`trit_app.py`**
The OBSERVE desktop GUI. Tkinter app with 5 themes (Matrix, Cyberpunk, Amber, Ice, Ghost), directory management, live search-as-you-type, click-to-open files, copy-to-clipboard. Contains its own file-scanning/indexing logic independent of `trit_search.py` — if you change skip directories, you must update both files.

**Ternary compression is genuinely active here** (verified 2026-06-29, not just available-but-dormant): vectors are quantized to {-1,0,+1} and bit-packed (5 trits/byte) for ~19.85x disk compression, metadata is deduplicated and previews regenerated live from disk instead of cached (~7.3x), for ~16.9x total index size reduction. Critically, the packed vectors are **unpacked once at load time** into RAM — unpacking on every search was tested and found 28-32x slower at scale, so this matters: don't "simplify" by removing the unpack-once caching.
```
python trit_app.py
```

**`trit_embed_train.py`**
Fine-tunes `all-MiniLM-L6-v2` on GitHub code using contrastive learning (`MultipleNegativesRankingLoss`). Streams code from HuggingFace across 20 languages, extracts (function_name→body), (comment→code), (class_name→body) pairs automatically. Local `.gd` files weighted 5x. Produces `models/code-minilm/`.
```
python trit_embed_train.py --train
python trit_embed_train.py --train --local
python trit_embed_train.py --install
```

**`trit_benchmark.py`**
25 hard semantic triples (query, correct code, deliberately-similar-but-wrong code) used to score model quality. Result on this set: Microsoft baseline 92%, your fine-tuned 96%.
```
python trit_benchmark.py --compare
```

**`trit_oss_test.py`**
Downloads 6 real open-source projects (FastAPI, Godot demo, Redis, Tokio, Gin, godot-gdext) spanning Python/GDScript/C/Rust/Go, indexes 23,529 real code chunks, runs 30 queries. Result: baseline 76.7%, fine-tuned 92.0%.
```
python trit_oss_test.py --all
```

**`benchmark_3way.py`**
Standalone comparison script: Microsoft baseline vs your fine-tuned model vs a third community code-search model (`flax-sentence-embeddings/st-codesearch-distilroberta-base`), all on the same `trit_benchmark.py` triples. Confirmed result: your model wins on accuracy, margin, and speed against both.
```
python benchmark_3way.py
```

**`gpu_benchmark.py`**
Measures real matrix-multiply throughput (TFLOPS) on your GPU and compares against published Nvidia datacenter GPU specs (A100, H100) to show the actual raw-compute gap between consumer and cluster hardware.
```
python gpu_benchmark.py
```

**`precision_loss_test.py`**
Measures the real accuracy cost of ternary *weight* quantization (separate from the embedding-vector compression in OBSERVE) by training matched Float32/Ternary/Mixed-precision CNN variants on CIFAR-10. Result: full ternary loses 7.35pp accuracy; keeping first+last layers float32 ("mixed precision") cuts that loss to 3.70pp for almost the same compression ratio (~18x vs 20.2x).
```
python precision_loss_test.py
```

**`scale_test.py`**
Synthetic benchmark (no downloads needed) proving ternary compression behavior at 326/5k/50k/500k chunk scale: storage compression holds at ~19.9x at every scale, but unpacking the bit-packed format on every search is 28-32x *slower* than float32 search. This is why `trit_app.py` unpacks once at load time instead of per-query — found and fixed in this session.
```
python scale_test.py
```

---

### LLM Fine-Tuning Pipeline

**`trit_lora.py`**
LoRA fine-tuning for Qwen2.5-coder:7b on your own data. Trains only ~1% of weights via low-rank adapter matrices — base model stays frozen, adapter is ~50MB, fits in 8GB VRAM.

**`trit_merge.py`**
Merges a trained LoRA adapter into the base Qwen model (`merge_and_unload()`), then optionally quantizes to 4-bit NF4 (BitsAndBytes) or GGUF (for llama.cpp/Ollama). Can generate an Ollama Modelfile for `ollama create`.
```
python trit_merge.py --merge --quantize
python trit_merge.py --test
```

**`trit_mega_train.py`**
Continual learning pipeline streaming Wikipedia + 10 programming languages via HuggingFace streaming (never more than ~500MB on disk at once). Cycles: Wikipedia → Python → JS → C/C++ → C# → GDScript → Rust → Go → Java → TypeScript.

**`trit_distill.py`**
Knowledge distillation: generates training text from Qwen2.5:7b (via local Ollama) as a "teacher," then trains your ternary TritLM "student" on that generated corpus.
```
python trit_distill.py --generate
python trit_distill.py --train
python trit_distill.py --chat
```

**`trit_lm.py`**
The base character-level ternary language model — triadic attention blocks, ternary weights, fixed-size memory cell (no growing KV cache). Trained/benchmarked on Shakespeare against a parameter-matched standard transformer.

**`trit_rag.py`**
Retrieval-Augmented Generation using ternary embeddings: query → TritEncoder (256-dim) → FAISS search → retrieved facts → TritLM → generated answer. Scales to millions of facts on a single GPU.

**`trit_memory_store.py`**
A Hopfield-style associative memory using ternary weights — stores discrete facts as trit patterns, retrieves with a consensus gate (`sign(a+b+c)` majority vote) even from partial/noisy queries. Different from TritLM: stores facts, doesn't generate text.

**Known bug:** the entire demo/training script runs at module level with no `if __name__ == "__main__":` guard — simply `import`-ing this file (e.g. to reuse `TernaryHopfield`) triggers a full 500-epoch encoder training run and the interactive demo. Discovered when building `trit_resonant_memory.py`; worked around by copying the `TernaryHopfield` class directly instead of importing. Worth fixing if this file needs to be reused as a library.

---

### Build & Distribution

**`build_all.py`**
Cross-platform PyInstaller build script. Builds Windows `.exe` (`--onefile`), Mac `.app`/`.dmg` (`--onedir --windowed`), Linux binary (`--onefile`), and generates a GitHub Actions CI workflow for automated multi-platform builds.
```
python build_all.py
```

**`build_mac.sh`** / **`build_linux.sh`**
Generated by `build_all.py` — platform-specific build scripts to run on Mac/Linux directly (install deps, run PyInstaller).

**`trit_install.sh`**
One-command Mac installer — creates an isolated venv at `~/.trit-search/`, installs dependencies, creates a Desktop app shortcut, runs the initial codebase index automatically.

**`OBSERVE.spec`** / **`TritSearch.spec`**
PyInstaller build specs auto-generated during exe builds (define what files/hidden imports get bundled). Safe to delete and regenerate.

**`requirements_app.txt`**
Minimal dependency list for the OBSERVE desktop app specifically (sentence-transformers, faiss-cpu, flask, torch) — used for building a smaller/cleaner venv than the full research environment.

---

### Resonance Hypothesis Test Series

Five honest, falsifiable tests of whether adding a "resonating cell" (slow low-pass filter, biologically inspired) or a "shaped resonant cavity" (tunable mode-mixing) to the hardware consensus-gate/triadic primitives actually improves decision-making. See DOCS.md for full results tables — summary: added flexibility/resonance helps only when the underlying signal genuinely changes over time, hurts or ties when the signal is static.

**`trit_cache_eviction.py`**
Consensus-gate (3-trit-vote: recency/frequency/fairness) vs standard LRU cache eviction on a Zipfian access trace, no resonance yet. Result: LRU wins at every cache size tested.
```
python trit_cache_eviction.py
```

**`trit_resonant_cache.py`**
Adds resonance (low-pass filtered recency) plus injected noise (decoy accesses) to the cache eviction test. Result: resonance made eviction decisions *worse* than plain consensus, which was already worse than LRU — smoothing an already-clean signal hurts.
```
python trit_resonant_cache.py
```

**`trit_os_sim.py`**
A software OS-concept simulator (NOT a real OS — no ternary hardware exists to boot on) demonstrating consensus-gate process scheduling and ternary-compressed process control blocks, reusing `pack_ternary`/`unpack_ternary` from `trit_app.py`. Confirmed 12x compression and correct round-trip on a small process table.
```
python trit_os_sim.py
```

**`trit_adaptive_scheduler.py`**
Tests resonance on a genuinely time-varying signal: a process becomes "urgent" for a 100-tick window, then stops. Compares static round-robin, direct-signal consensus, and resonant consensus on how well each adapts CPU allocation. Result: **resonance wins clearly here** — 71% CPU share during urgency (vs 41% direct-reaction, 25% no-adaptation), correctly relaxing back down after. This is the one test in the series where the hybrid genuinely outperforms the alternatives.
```
python trit_adaptive_scheduler.py
```

**`trit_resonant_memory.py`**
Tests resonance on noisy-glimpse memory retrieval using the real `TernaryHopfield` class (synthetic random trit patterns standing in for facts, skipping the 500-epoch neural encoder for speed). Compares single-shot, instant majority-vote, and resonant (pre-filter) retrieval across noise levels. Result: simple instant-vote beats resonance — with multiple equally-reliable noisy reads of one static fact, recency-weighting throws away good information for no benefit.
```
python trit_resonant_memory.py
```

**`trit_resonant_hopfield.py`**
Retests memory retrieval with a corrected mechanism: resonance interleaved *inside* the Hopfield settling process (EMA blend + energy-minimization step per glimpse) instead of as a pre-filter. Result: clearly improves over the pre-filter version, but still doesn't surpass plain instant-vote — confirms that mechanism placement matters, but doesn't change the underlying finding that static signals don't benefit from resonance.
```
python trit_resonant_hopfield.py
```

**`trit_mcp_server.py`**
Exposes OBSERVE's compressed semantic search as an MCP (Model Context Protocol) server, so AI coding assistants (Claude Code, Claude Desktop, etc.) can call it as a tool directly from a conversation. Wraps the same `SearchEngine` as `trit_app.py` — same model, same compressed index — running headless over stdio (no GUI). Two tools exposed: `search_code(query, k)` and `index_status()`. Verified working through the real MCP protocol layer (`list_tools()`/`call_tool()`), not just direct function calls — returns identical results/scores to the GUI search. Requires an index to already exist (build one first via `trit_app.py` or `trit_search.py --index`); this server only reads, doesn't build.
```
pip install mcp
python trit_mcp_server.py
```
Add to an MCP client config (e.g. Claude Code) with:
```json
{"mcpServers": {"observe": {"command": "python", "args": ["path/to/trit_mcp_server.py"]}}}
```

**`trit_emulator.py`**
A balanced-ternary CPU emulator (software simulation, same honesty framing as `trit_os_sim.py` — no real ternary hardware required or implied). 9-trit words (3^9=19,683 states), 8 registers, flat memory, and an instruction set including `CONSENSUS` as a *native* instruction (per-trit `sign(a+b+c)`, not built from comparisons) — the same gate as `hardware/consensus_gate.sv`. Verified: balanced ternary arithmetic handles negative numbers with no separate sign bit (47 + -12 = 35, 47 - -12 = 59, both correct); consensus voting and loop control confirmed working.
```
python trit_emulator.py
```

**`trit_emulator_tests.py`**
Five concrete, falsifiable tests on `trit_emulator.py`:
1. Native `CONSENSUS` vs manual `ADD+ADD+SGN` reconstruction — **1.4x fewer instructions** (modest, real, much smaller than earlier "10x faster" scheduler claims)
2. Balanced ternary's symmetric range vs binary two's-complement's negation-overflow bug at INT_MIN — **confirmed real advantage**, balanced ternary avoids this bug class by construction
3. Ports `trit_os_sim.py`'s scheduler consensus logic to real emulator assembly (not Python) — confirmed working correctly
4. Word overflow/wraparound boundary testing — confirmed `clamp_word` wraps correctly and predictably (caught and fixed a bug in the test's own verification formula along the way, not in the emulator)
5. Emulator throughput: 256K ops/sec vs raw Python's 17.1M ops/sec — **67x slower**, expected and disclosed honestly (interpreted simulation, not a performance claim)
```
python trit_emulator_tests.py
```

**`trit_assembler.py`**
A text-based assembler for `trit_emulator.py`'s TritChip — write programs as readable assembly (`LOAD R0, 5`, `loop:` labels, `JNZ R0, loop`) instead of raw Python opcode tuples. Two-pass assembler (resolve labels, then parse instructions), with real error checking (unknown opcodes, wrong operand counts both raise `AssemblerError` with line numbers). Verified: assembled/ran the same consensus-vote loop as `trit_emulator.py`'s demo 3, produced identical output.
```
python trit_assembler.py program.tasm     Assemble and run a file
python trit_assembler.py                  Run the built-in demo
```

**`trit_symmetry_cavity_test.py`**
Tests a follow-on "shaped resonant cavity" proposal: tunable triadic mode-mixing weights (fixed-symmetric, fixed-asymmetric, adaptive/learned, input-driven) vs the original fixed formula (`s1*(1-s0)+s2*s0`), on real CIFAR-10 rotation robustness (same harness as `experiments.py`). Skips the proposal's speculative optical-cavity physics (Fano resonance, unverifiable citations) and tests only the concrete claim. Result: the proposal's specific prediction (adaptive/input-driven should win) was wrong — adaptive had the *worst* stability of all 5 variants tested; the original fixed formula remained best on accuracy and competitive on stability.
```
python trit_symmetry_cavity_test.py
```

---

### Documentation

**`README.md`**
Original project README — covers the ternary/triadic research stack (`experiments.py`, hardware) and headline benchmark numbers (28x fewer params, 20x compression, stability gains vs ResNet18).

**`DOCS.md`**
Full architecture documentation — every file's purpose, the three pipelines (search, LLM fine-tuning, language model), and key numbers in one place.

**`FILES.md`** (this file)
Per-file reference with run commands.

**`paper/012_paper.md`** / **`paper/012_paper.tex`**
Full research paper draft on the 012 triadic architecture.

**`paper/figures.py`**
Generates the figures used in the paper (e.g. rotation robustness plots).

---

### Hardware (SystemVerilog RTL)

Physical chip implementation of ternary compute — proves the architecture maps to real silicon, not just simulation.

| File | Purpose |
|---|---|
| `hardware/trit_pkg.sv` | Ternary type definitions |
| `hardware/trit_register.sv` | Ternary D flip-flop (storage) |
| `hardware/trit_not.sv` | Ternary NOT gate |
| `hardware/trit_add.sv` | Ternary adder with carry |
| `hardware/trit_mac.sv` | Multiply-accumulate — no multipliers needed, just adders |
| `hardware/consensus_gate.sv` | 3-input majority-vote gate (the hardware form of the Observer/Shadow/Light consensus) |
| `hardware/triadic_pe.sv` | Triadic processing element (combines the above into one compute unit) |
| `hardware/012_top.sv` | Top-level chip design wiring everything together |
| `hardware/testbench.sv` | Simulation testbench — 36/36 test cases passing |
| `hardware/vivado_synth.tcl` | Xilinx Vivado synthesis script for FPGA deployment |
| `hardware/ternary_rtl.py` | Python reference simulator validating the RTL logic |

```bash
iverilog -g2012 -o sim hardware/trit_pkg.sv hardware/trit_register.sv \
    hardware/trit_not.sv hardware/trit_add.sv \
    hardware/consensus_gate.sv hardware/testbench.sv
vvp sim
```

---

### Generated / Runtime Directories (not source files)

These are created by running the scripts above — not hand-written, safe to delete and regenerate:

- **`models/`** — downloaded/fine-tuned model weights (`code-minilm`, `triadic-encoder`)
- **`checkpoints/`** — LoRA adapters and training checkpoints
- **`data/`** — downloaded datasets (CIFAR-10, STL-10, Shakespeare text)
- **`results/`** — benchmark JSON outputs
- **`oss_results/`** — `trit_oss_test.py` output (`oss_benchmark.json`)
- **`search_index/`** — FAISS index + metadata from `trit_search.py --index`
- **`index/`** — saved TritLM/TritEncoder model weights from early experiments
- **`dist/`** / **`build/`** — PyInstaller build output
- **`__pycache__/`** — Python bytecode cache

---

### Misc

**`.gitignore`**
Excludes all generated/large files above from git (models, datasets, checkpoints, indexes, binary build artifacts).

**`2.0`**, **`Node`**, **`void`**
Empty or placeholder files — no functional content, safe to ignore/delete.
