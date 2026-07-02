"""
Corpus-wide document-frequency index for distinctive tokens.

calibrate_cross_language.py found that raw token overlap fails as a
cross-language relationship signal: common English programming vocabulary
(append, array, before, between) appears in nearly every file, so large
files share MORE raw tokens with unrelated files than a genuinely related
pair shares with its real counterpart. A fixed stopword list can't capture
this -- "neuron"/"threshold" need to count for more than "append"/"array"
specifically because they're much rarer across the actual corpus, which is
a corpus-level fact, not something a hand-picked blocklist can know.

This builds that corpus-level fact once: for every distinctive token,
how many of the ~1800 indexed FILES (not chunks -- file-level presence,
so a word used many times in one file doesn't get overweighted) contain
it. Cached to disk since it requires reading every indexed file's full
text once (~1800 files) -- expensive to redo per query, cheap to
precompute and reuse.

Usage:
    python corpus_idf.py          Build and cache the document-frequency index
"""
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE_PATH = Path(__file__).resolve().parent / "corpus_df_cache.json"

_STOPWORDS = {
    "self", "return", "false", "true", "none", "null", "print", "import",
    "class", "public", "private", "static", "const", "float", "int",
    "string", "bool", "void", "function", "def", "var", "let", "extends",
    "export", "value", "index", "count", "error", "result", "data",
}

# Framework/stdlib API surface, not user-written vocabulary. Found via a
# real failure: apply_compound_signal.py's first full run at scale flagged
# `acoustic-3d/base.gd <-> 3d-simulation/main.gd` sharing 29 "compound
# identifiers" (compound_idf=114.4) as its top match -- every one of them
# was a built-in Godot Node method or property (_ready, _process, add_child,
# queue_free, get_tree, global_position, material_override...), which every
# single GDScript file in the corpus uses. IDF computed across a MIXED
# multi-language corpus still scored these as "rare," because rarity was
# measured globally (few non-.gd files use them) rather than conditioned on
# "rare within GDScript specifically" -- a language-conditional stopword
# problem IDF alone can't see. This denylist is the concrete fix: known
# engine/stdlib identifiers are excluded before computing compound overlap,
# regardless of their global corpus rarity.
_ENGINE_API_DENYLIST = {
    # Godot virtual callbacks
    "_ready", "_process", "_physics_process", "_input", "_unhandled_input",
    "_unhandled_key_input", "_draw", "_enter_tree", "_exit_tree", "_init",
    "_notification", "_gui_input", "_integrate_forces", "_get_configuration_warnings",
    # Godot Node/Node2D/Node3D/Control common API
    "add_child", "remove_child", "get_child", "get_children", "get_node",
    "get_node_or_null", "get_tree", "get_parent", "queue_free", "is_in_group",
    "add_to_group", "remove_from_group", "global_position", "global_transform",
    "global_rotation", "rotation_degrees", "look_at", "look_at_from_position",
    "distance_to", "distance_squared_to", "set_anchors_preset",
    "add_theme_font_size_override", "add_theme_color_override",
    "custom_minimum_size", "material_override", "albedo_color",
    "emission_enabled", "emission_energy_multiplier", "background_color",
    "background_mode", "transparency_alpha", "mouse_button_left",
    "mouse_button_right", "mouse_button_wheel_up", "mouse_button_wheel_down",
    "mouse_button_middle", "key_escape", "key_space", "key_enter",
    "deg_to_rad", "rad_to_deg", "randf_range", "randi_range", "randf", "randi",
    "create_timer", "process_frame", "physics_frame", "cull_disabled",
    "cull_mode", "horizontal_alignment", "horizontal_alignment_center",
    "vertical_alignment", "font_size", "bottom_radius", "top_radius",
    "billboard_enabled", "get_instance_id", "button_index", "world_pos",
    "get_viewport", "get_window", "queue_redraw", "set_process",
    "set_physics_process", "instantiate", "preload", "load", "connect",
    "disconnect", "emit_signal", "is_instance_valid", "class_name",
    # Python dunder/stdlib
    "__main__", "__name__", "__init__", "__repr__", "__str__", "__len__",
    "__eq__", "__hash__", "__iter__", "__next__", "__enter__", "__exit__",
    "__call__", "__getitem__", "__setitem__",
}


def identifier_only_tokens(text: str) -> set:
    idents = {m.lower() for m in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{4,}\b", text)}
    idents -= _STOPWORDS
    strings = {m.lower() for m in re.findall(r'"([^"]{4,40})"', text)}
    return idents | strings


def compound_identifier_overlap(text_a: str, text_b: str) -> set:
    """Shared compound (snake_case) identifiers between two files, with
    known engine/stdlib API excluded -- see _ENGINE_API_DENYLIST for why
    that exclusion is necessary, not optional."""
    toks_a = identifier_only_tokens(text_a)
    toks_b = identifier_only_tokens(text_b)
    shared = toks_a & toks_b
    return {t for t in shared if "_" in t and t not in _ENGINE_API_DENYLIST}


def build_document_frequency(engine, base_dirs: list) -> dict:
    """df[token] = number of distinct files containing that token at least once."""
    seen_files = set()
    df = Counter()
    n_files = 0

    for p in engine.path_table:
        rel = p["rel_path"]
        key = (p["base_dir"], rel)
        if key in seen_files:
            continue
        seen_files.add(key)

        full = None
        for b in base_dirs:
            cand = Path(b) / rel
            if cand.exists() and cand.is_file():
                full = cand
                break
        if full is None:
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not text:
            continue

        n_files += 1
        for tok in identifier_only_tokens(text):
            df[tok] += 1

        if n_files % 200 == 0:
            print(f"  ...{n_files} files processed")

    return dict(df), n_files


def load_or_build():
    if CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return cached["df"], cached["n_files"]

    from trit_app import SearchEngine
    from trit_entanglement import INDEX_DIR, MODEL_PATH
    engine = SearchEngine()
    engine.load(INDEX_DIR, MODEL_PATH, lambda m: print(f"  {m}"))
    while not engine.ready:
        time.sleep(0.2)
    base_dirs = sorted({p["base_dir"] for p in engine.path_table})

    print(f"Building document-frequency index across {len(engine.path_table)} path entries...")
    df, n_files = build_document_frequency(engine, base_dirs)
    print(f"Done: {n_files} files, {len(df)} distinct tokens")

    CACHE_PATH.write_text(json.dumps({"df": df, "n_files": n_files}), encoding="utf-8")
    print(f"Cached: {CACHE_PATH}")
    return df, n_files


def idf(token: str, df: dict, n_files: int) -> float:
    d = df.get(token, 0)
    return math.log((n_files + 1) / (d + 1))


if __name__ == "__main__":
    load_or_build()
