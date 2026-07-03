# Porting OBSERVE to another machine or software implementation

The analysis framework (semantic search + code entanglement + reference
scanning + AST matching) is not tied to this specific codebase. Here is
exactly what is portable and what needs setup, verified against the code.

## Already portable (no changes needed)

- **Core pipeline** — `observe_pipeline.py`, `trit_entanglement.py`,
  `code_references.py`, `ast_entanglement.py`, `run_full_pipeline.py`,
  and the GUI (`trit_app.py`) work off `base_dirs` read from the loaded
  index at runtime, not hardcoded paths.
- **Index location** — `INDEX_DIR` uses `Path.home() / ".trit-search"`,
  already user-agnostic.
- **Model** — `MODEL_PATH` falls back to downloading `all-MiniLM-L6-v2`
  from Hugging Face if no local fine-tuned model is present.
- **Project grouping config** — `CONTAINER_PREFIXES` is now DERIVED from
  `Path.home()` automatically (was hardcoded to one username). Multi-drive
  paths (C:/, D:/, G:/, ...) are handled correctly.

## Setup needed for a new deployment

1. **Build an index for the new codebase.** Launch `trit_app.py`, use
   "+ ADD DIR" to point at the directories you want, then "INDEX
   CODEBASE". This writes `~/.trit-search/index`. Everything downstream
   reads from there.

2. **(Optional) Configure project grouping.** If your codebases live
   somewhere the auto-derived home-relative prefixes don't cover (e.g. a
   dedicated `D:/code/` root), copy `observe_config.example.json` to
   `observe_config.json` and set `container_prefixes`. Also set
   `non_project_hints` to your own non-code folders (e.g. `node_modules`,
   `venv`) or `[]` to disable that distinction. No code edits required.

3. **Ollama + models** (only for the LLM summarization stage). Install
   Ollama and pull the models the pipeline uses:
   `qwen2.5-coder:7b`, `deepseek-r1:7b`, `llama3.2`. The reference
   scanner and AST matcher need none of this -- they are pure static
   analysis and run without Ollama.

## Still machine-specific (secondary, not core)

These are calibration/benchmark harnesses with hardcoded ground-truth
paths -- they validate the framework against THIS machine's files and
would need their example paths updated (or simply aren't needed) on a new
deployment:

- `calibrate_cross_language.py` — hardcoded confirmed-genuine/coincidental
  file pairs.
- `retune_likely_genuine.py`, `apply_compound_signal.py` — hardcoded
  `BASE_DIR_CANDIDATES`.
- `trit_*_test.py`, `trit_quality_benchmark.py` — hardcoded `PROJECT_DIR`.

None of these are in the runtime path of the actual tools; they are
one-off measurement scripts.
