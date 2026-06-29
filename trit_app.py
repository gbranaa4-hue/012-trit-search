"""
012 Trit Search — Desktop App
Standalone GUI with Matrix/cyberpunk themes.
Packages into a single .exe with PyInstaller.

Install deps:
  pip install sentence-transformers faiss-cpu flask torch tkinter pyinstaller

Build exe:
  pyinstaller --onefile --noconsole --name TritSearch trit_app.py

Usage:
  python trit_app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading, subprocess, sys, os, json, time, random, webbrowser
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# THEMES
# ══════════════════════════════════════════════════════════════════════════════

THEMES = {
    "Matrix": {
        "bg":           "#0D0D0D",
        "bg2":          "#111111",
        "fg":           "#00FF41",
        "fg_dim":       "#005C13",
        "fg_bright":    "#AFFFBC",
        "accent":       "#00FF41",
        "accent2":      "#00CC33",
        "error":        "#FF4444",
        "border":       "#003B00",
        "font_main":    ("Courier New", 11),
        "font_title":   ("Courier New", 22, "bold"),
        "font_small":   ("Courier New", 9),
        "rain_chars":   "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ01",
    },
    "Cyberpunk": {
        "bg":           "#0A0014",
        "bg2":          "#120020",
        "fg":           "#FF00FF",
        "fg_dim":       "#5C005C",
        "fg_bright":    "#FFB3FF",
        "accent":       "#00FFFF",
        "accent2":      "#FF00FF",
        "error":        "#FF4444",
        "border":       "#3D0060",
        "font_main":    ("Courier New", 11),
        "font_title":   ("Courier New", 22, "bold"),
        "font_small":   ("Courier New", 9),
        "rain_chars":   "▓▒░█▄▀■□●○◆◇★☆♦♠♣♥0123456789ABCDEF",
    },
    "Amber": {
        "bg":           "#0D0800",
        "bg2":          "#150E00",
        "fg":           "#FFB000",
        "fg_dim":       "#5C3E00",
        "fg_bright":    "#FFD966",
        "accent":       "#FF8C00",
        "accent2":      "#FFB000",
        "error":        "#FF4444",
        "border":       "#3D2800",
        "font_main":    ("Courier New", 11),
        "font_title":   ("Courier New", 22, "bold"),
        "font_small":   ("Courier New", 9),
        "rain_chars":   "01▌▐░▒▓█ABCDEF0123456789",
    },
    "Ice": {
        "bg":           "#00050F",
        "bg2":          "#000A1A",
        "fg":           "#00CFFF",
        "fg_dim":       "#004466",
        "fg_bright":    "#AAEEFF",
        "accent":       "#0088FF",
        "accent2":      "#00CFFF",
        "error":        "#FF4444",
        "border":       "#003355",
        "font_main":    ("Courier New", 11),
        "font_title":   ("Courier New", 22, "bold"),
        "font_small":   ("Courier New", 9),
        "rain_chars":   "❄✦✧⬡⬢◈◉○●◦•·˙0123456789",
    },
    "Ghost": {
        "bg":           "#080808",
        "bg2":          "#0F0F0F",
        "fg":           "#CCCCCC",
        "fg_dim":       "#444444",
        "fg_bright":    "#FFFFFF",
        "accent":       "#888888",
        "accent2":      "#CCCCCC",
        "error":        "#FF4444",
        "border":       "#222222",
        "font_main":    ("Courier New", 11),
        "font_title":   ("Courier New", 22, "bold"),
        "font_small":   ("Courier New", 9),
        "rain_chars":   "░▒▓01·˙",
    },
}

CURRENT_THEME = "Matrix"

# ══════════════════════════════════════════════════════════════════════════════
# MATRIX RAIN CANVAS
# ══════════════════════════════════════════════════════════════════════════════

class MatrixRain:
    def __init__(self, canvas, theme):
        self.canvas  = canvas
        self.theme   = theme
        self.columns = []
        self.running = False
        self.after_id = None

    def start(self, width, height):
        self.width   = width
        self.height  = height
        self.cols    = max(1, width // 14)
        self.rows    = max(1, height // 16)
        self.columns = [random.randint(0, self.rows) for _ in range(self.cols)]
        self.running = True
        self._draw()

    def stop(self):
        self.running = False
        if self.after_id:
            try: self.canvas.after_cancel(self.after_id)
            except: pass

    def _draw(self):
        if not self.running:
            return
        t   = self.theme
        chars = t["rain_chars"]
        self.canvas.delete("rain")

        for col, row in enumerate(self.columns):
            x = col * 14 + 7
            # Tail — dim
            for r in range(max(0, row-8), row):
                y    = r * 16 + 8
                char = random.choice(chars)
                self.canvas.create_text(x, y, text=char, fill=t["fg_dim"],
                                        font=t["font_small"], tags="rain")
            # Head — bright
            if row < self.rows:
                y    = row * 16 + 8
                char = random.choice(chars)
                self.canvas.create_text(x, y, text=char, fill=t["fg_bright"],
                                        font=t["font_small"], tags="rain")

            self.columns[col] = (row + 1) % (self.rows + random.randint(4, 16))

        self.after_id = self.canvas.after(80, self._draw)


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH ENGINE (runs in background thread)
# ══════════════════════════════════════════════════════════════════════════════

class SearchEngine:
    def __init__(self):
        self.index    = None
        self.metadata = []
        self.model    = None
        self.ready    = False
        self.status   = "Not initialized"

    def load(self, index_dir, model_path, on_status):
        def _load():
            try:
                import faiss
                import numpy as np
                from sentence_transformers import SentenceTransformer

                on_status("Loading model...")
                self.model = SentenceTransformer(model_path)

                idx_path  = os.path.join(index_dir, "faiss.index")
                meta_path = os.path.join(index_dir, "metadata.json")

                if os.path.exists(idx_path) and os.path.exists(meta_path):
                    on_status("Loading index...")
                    self.index    = faiss.read_index(idx_path)
                    self.metadata = json.load(open(meta_path, encoding="utf-8"))
                    on_status(f"✓ Ready — {len(self.metadata):,} chunks indexed. Start searching!")
                else:
                    on_status("⟳ No index yet — add a directory then click INDEX CODEBASE")
                self.ready = True
            except Exception as e:
                on_status(f"Error: {e}")
        threading.Thread(target=_load, daemon=True).start()

    def search(self, query, k=10):
        if not self.ready or self.index is None or self.model is None:
            return []
        import numpy as np
        vec    = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(vec, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                m = self.metadata[idx]
                results.append({
                    "score":   float(score),
                    "path":    m.get("rel_path", m.get("path", "?")),
                    "preview": m.get("preview", ""),
                })
        return results

    def build_index(self, scan_dirs, index_dir, on_status, on_done):
        def _build():
            try:
                import faiss, numpy as np
                from sentence_transformers import SentenceTransformer

                if self.model is None:
                    on_status("Loading model...")

                EXTS = {".py",".gd",".js",".ts",".cs",".rs",".go",
                        ".c",".cpp",".h",".java",".lua",".rb",".php",
                        ".swift",".kt",".dart",".zig",".md",".sh",".ps1"}
                SKIP = {".git","__pycache__","node_modules",".venv","venv",
                        "dist","build","target","models","search_index"}

                chunks = []
                files  = 0
                on_status("Scanning files...")

                for base in scan_dirs:
                    for root, dirs, fnames in os.walk(base):
                        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
                        for fname in fnames:
                            if Path(fname).suffix.lower() not in EXTS:
                                continue
                            fpath = os.path.join(root, fname)
                            try:
                                text = open(fpath, encoding="utf-8", errors="ignore").read()
                                if len(text.strip()) < 100:
                                    continue
                                rel = os.path.relpath(fpath, base)
                                # Chunk
                                for i in range(0, len(text), 700):
                                    chunk = text[i:i+800]
                                    if len(chunk.strip()) > 50:
                                        chunks.append({
                                            "text":     f"file:{rel}\n{chunk}",
                                            "rel_path": rel,
                                            "preview":  chunk[:120].replace("\n", " "),
                                        })
                                files += 1
                            except: pass

                on_status(f"Encoding {len(chunks):,} chunks from {files:,} files...")
                os.makedirs(index_dir, exist_ok=True)

                texts = [c["text"] for c in chunks]
                bs    = 128
                vecs  = []
                for i in range(0, len(texts), bs):
                    batch = texts[i:i+bs]
                    v = self.model.encode(batch, normalize_embeddings=True,
                                          show_progress_bar=False)
                    vecs.append(v)
                    pct = (i+bs) / len(texts) * 100
                    on_status(f"Encoding... {min(pct,100):.0f}%  ({min(i+bs, len(texts)):,}/{len(texts):,})")

                vecs = np.vstack(vecs).astype("float32")

                index = faiss.IndexFlatIP(vecs.shape[1])
                index.add(vecs)
                faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

                meta = [{"rel_path": c["rel_path"], "preview": c["preview"]} for c in chunks]
                json.dump(meta, open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8"), ensure_ascii=False)

                self.index    = index
                self.metadata = meta
                self.ready    = True
                on_status(f"Index complete — {len(chunks):,} chunks, {files:,} files")
                on_done()
            except Exception as e:
                on_status(f"Index error: {e}")
        threading.Thread(target=_build, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

class TritSearchApp:
    def __init__(self, root):
        self.root         = root
        self.theme_name   = CURRENT_THEME
        self.t            = THEMES[self.theme_name]
        self.engine       = SearchEngine()
        self.rain         = None
        self.scan_dirs    = []
        self.index_dir    = str(Path.home() / ".trit-search" / "index")
        self.model_path   = self._find_model()
        self.typing_job   = None
        self._last_results = []

        self.root.title("012 — Trit Search")
        self.root.geometry("1000x720")
        self.root.minsize(800, 600)
        self.root.configure(bg=self.t["bg"])

        self._build_ui()
        self._apply_theme()
        self._start_rain()
        self._load_config()
        self._init_engine()

    def _find_model(self):
        # Look for fine-tuned model next to script, then fallback to base
        candidates = [
            Path(__file__).parent / "models" / "code-minilm",
            Path.home() / ".trit-search" / "models" / "code-minilm",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return "all-MiniLM-L6-v2"

    # ── UI BUILD ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        t = self.t

        # ── Top bar ──
        self.topbar = tk.Frame(self.root, bg=t["bg"], height=50)
        self.topbar.pack(fill="x", padx=0, pady=0)
        self.topbar.pack_propagate(False)

        self.title_lbl = tk.Label(self.topbar, text="◈ 012 TRIT SEARCH",
                                   font=t["font_title"], bg=t["bg"], fg=t["accent"])
        self.title_lbl.pack(side="left", padx=20, pady=8)

        # Theme selector
        theme_frame = tk.Frame(self.topbar, bg=t["bg"])
        theme_frame.pack(side="right", padx=16, pady=8)

        tk.Label(theme_frame, text="THEME:", font=t["font_small"],
                 bg=t["bg"], fg=t["fg_dim"]).pack(side="left", padx=(0,6))

        self.theme_var = tk.StringVar(value=self.theme_name)
        self.theme_menu = ttk.Combobox(theme_frame, textvariable=self.theme_var,
                                        values=list(THEMES.keys()), width=10,
                                        state="readonly", font=t["font_small"])
        self.theme_menu.pack(side="left")
        self.theme_menu.bind("<<ComboboxSelected>>", self._on_theme_change)

        # ── Divider ──
        self.div1 = tk.Frame(self.root, bg=t["border"], height=1)
        self.div1.pack(fill="x")

        # ── Main pane ──
        self.main = tk.Frame(self.root, bg=t["bg"])
        self.main.pack(fill="both", expand=True)

        # Left sidebar
        self.sidebar = tk.Frame(self.main, bg=t["bg2"], width=220)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="CODEBASE", font=self.t["font_small"],
                 bg=t["bg2"], fg=t["fg_dim"]).pack(pady=(16,4), padx=12, anchor="w")

        self.dir_listbox = tk.Listbox(self.sidebar, bg=t["bg"], fg=t["fg"],
                                       font=t["font_small"], selectbackground=t["border"],
                                       selectforeground=t["fg_bright"],
                                       relief="flat", highlightthickness=0,
                                       height=6)
        self.dir_listbox.pack(fill="x", padx=8)

        btn_frame = tk.Frame(self.sidebar, bg=t["bg2"])
        btn_frame.pack(fill="x", padx=8, pady=4)

        self.add_dir_btn = self._btn(btn_frame, "+ ADD DIR", self._add_dir)
        self.add_dir_btn.pack(side="left", padx=(0,4))
        self.rem_dir_btn = self._btn(btn_frame, "− REMOVE", self._remove_dir)
        self.rem_dir_btn.pack(side="left")

        tk.Frame(self.sidebar, bg=t["border"], height=1).pack(fill="x", pady=8, padx=8)

        self.index_btn = self._btn(self.sidebar, "⟳ INDEX CODEBASE", self._start_index,
                                    width=18)
        self.index_btn.pack(pady=4, padx=8, fill="x")

        self.serve_btn = self._btn(self.sidebar, "⊕ OPEN IN BROWSER", self._open_browser,
                                    width=18)
        self.serve_btn.pack(pady=4, padx=8, fill="x")

        tk.Frame(self.sidebar, bg=t["border"], height=1).pack(fill="x", pady=8, padx=8)

        # Stats
        tk.Label(self.sidebar, text="STATS", font=self.t["font_small"],
                 bg=t["bg2"], fg=t["fg_dim"]).pack(pady=(4,2), padx=12, anchor="w")
        self.stats_lbl = tk.Label(self.sidebar, text="—", font=t["font_small"],
                                   bg=t["bg2"], fg=t["fg"], wraplength=190, justify="left")
        self.stats_lbl.pack(padx=12, anchor="w")

        # Rain canvas in sidebar
        self.rain_canvas = tk.Canvas(self.sidebar, bg=t["bg"], highlightthickness=0)
        self.rain_canvas.pack(fill="both", expand=True, padx=0, pady=(8,0))

        # ── Right panel ──
        self.right = tk.Frame(self.main, bg=t["bg"])
        self.right.pack(side="left", fill="both", expand=True)

        # Search bar
        search_frame = tk.Frame(self.right, bg=t["bg"])
        search_frame.pack(fill="x", padx=16, pady=12)

        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                                      font=("Courier New", 14),
                                      bg=t["bg2"], fg=t["fg"],
                                      insertbackground=t["accent"],
                                      relief="flat", highlightthickness=2,
                                      highlightcolor=t["accent"],
                                      highlightbackground=t["border"])
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0,8))
        self.search_entry.bind("<Return>", lambda e: self._search())
        self.search_entry.bind("<KeyRelease>", self._on_key)
        self.search_entry.insert(0, "Search your codebase by meaning...")
        self.search_entry.bind("<FocusIn>", self._clear_placeholder)
        self.search_entry.bind("<FocusOut>", self._restore_placeholder)

        self.search_btn = self._btn(search_frame, "SEARCH", self._search, width=10)
        self.search_btn.pack(side="left", padx=(0,4))

        self.copy_all_btn = self._btn(search_frame, "⎘ COPY ALL", self._copy_all, width=12)
        self.copy_all_btn.pack(side="left")

        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_lbl = tk.Label(self.right, textvariable=self.status_var,
                                    font=t["font_small"], bg=t["bg"],
                                    fg=t["fg_dim"], anchor="w")
        self.status_lbl.pack(fill="x", padx=16)

        # Results
        self.results_frame = tk.Frame(self.right, bg=t["bg"])
        self.results_frame.pack(fill="both", expand=True, padx=16, pady=8)

        self.results_canvas = tk.Canvas(self.results_frame, bg=t["bg"],
                                         highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical",
                                        command=self.results_canvas.yview)
        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.results_canvas.pack(side="left", fill="both", expand=True)

        self.results_inner = tk.Frame(self.results_canvas, bg=t["bg"])
        self.results_window = self.results_canvas.create_window(
            (0,0), window=self.results_inner, anchor="nw"
        )
        self.results_inner.bind("<Configure>", self._on_results_configure)
        self.results_canvas.bind("<Configure>", self._on_canvas_configure)
        self.results_canvas.bind("<MouseWheel>", self._on_mousewheel)

        # Bottom bar
        self.botbar = tk.Frame(self.root, bg=t["bg2"], height=28)
        self.botbar.pack(fill="x", side="bottom")
        self.botbar.pack_propagate(False)
        tk.Label(self.botbar, text="012 TERNARY SEARCH  ·  LOCAL  ·  PRIVATE  ·  FREE",
                 font=t["font_small"], bg=t["bg2"], fg=t["fg_dim"]).pack(side="left", padx=12)
        tk.Label(self.botbar, text="v1.0",
                 font=t["font_small"], bg=t["bg2"], fg=t["fg_dim"]).pack(side="right", padx=12)

    def _btn(self, parent, text, cmd, width=None):
        t   = self.t
        kw  = dict(text=text, command=cmd, font=t["font_small"],
                   bg=t["bg2"], fg=t["accent"], relief="flat",
                   activebackground=t["border"], activeforeground=t["fg_bright"],
                   cursor="hand2", bd=1, highlightthickness=1,
                   highlightbackground=t["border"])
        if width: kw["width"] = width
        b = tk.Button(parent, **kw)
        b.bind("<Enter>", lambda e: b.config(bg=t["border"], fg=t["fg_bright"]))
        b.bind("<Leave>", lambda e: b.config(bg=t["bg2"], fg=t["accent"]))
        return b

    # ── THEME ─────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        t = self.t
        # Restyle combobox
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TCombobox",
                         fieldbackground=t["bg2"],
                         background=t["bg2"],
                         foreground=t["fg"],
                         selectbackground=t["border"],
                         selectforeground=t["fg_bright"])
        style.configure("TScrollbar",
                         background=t["bg2"],
                         troughcolor=t["bg"],
                         arrowcolor=t["fg_dim"])

    def _on_theme_change(self, event=None):
        name = self.theme_var.get()
        self.theme_name = name
        self.t = THEMES[name]
        self._save_config()

        # Restart rain
        if self.rain:
            self.rain.stop()

        # Rebuild UI with new theme
        for widget in self.root.winfo_children():
            widget.destroy()
        self._build_ui()
        self._apply_theme()
        self._start_rain()
        self._restore_dirs()
        self._update_stats()
        self.theme_var.set(name)
        self.theme_menu.set(name)

    # ── RAIN ──────────────────────────────────────────────────────────────────

    def _start_rain(self):
        self.rain_canvas.update()
        w = self.rain_canvas.winfo_width()
        h = self.rain_canvas.winfo_height()
        if w < 10:
            self.root.after(200, self._start_rain)
            return
        self.rain = MatrixRain(self.rain_canvas, self.t)
        self.rain.start(w, h)

    # ── SEARCH ────────────────────────────────────────────────────────────────

    def _clear_placeholder(self, event):
        if self.search_entry.get() == "Search your codebase by meaning...":
            self.search_entry.delete(0, "end")
            self.search_entry.config(fg=self.t["fg"])

    def _restore_placeholder(self, event):
        if not self.search_entry.get():
            self.search_entry.insert(0, "Search your codebase by meaning...")
            self.search_entry.config(fg=self.t["fg_dim"])

    def _on_key(self, event):
        if self.typing_job:
            self.root.after_cancel(self.typing_job)
        q = self.search_var.get().strip()
        if len(q) > 3 and q != "Search your codebase by meaning...":
            self.typing_job = self.root.after(400, self._search)

    def _search(self):
        q = self.search_var.get().strip()
        if not q or q == "Search your codebase by meaning...":
            return
        if not self.engine.ready:
            self._set_status("Still loading — wait a moment...")
            return
        if self.engine.index is None:
            self._set_status("⟳ No index yet — add a directory then click INDEX CODEBASE")
            return

        self._set_status(f"Searching... index={self.engine.index.ntotal} chunks  model={'loaded' if self.engine.model else 'NONE'}")
        t0      = time.time()
        try:
            results = self.engine.search(q, k=12)
        except Exception as e:
            self._set_status(f"Search error: {e}")
            return
        elapsed = (time.time() - t0) * 1000

        self._set_status(f"  {len(results)} results  ·  {elapsed:.0f}ms  ·  query: \"{q}\"")
        self._last_results = results
        self._show_results(results)

    def _show_results(self, results):
        t = self.t
        for w in self.results_inner.winfo_children():
            w.destroy()

        if not results:
            tk.Label(self.results_inner, text="No results found.",
                     font=t["font_main"], bg=t["bg"], fg=t["fg_dim"]).pack(pady=20)
            return

        for i, r in enumerate(results):
            card = tk.Frame(self.results_inner, bg=t["bg2"],
                            highlightthickness=1, highlightbackground=t["border"])
            card.pack(fill="x", pady=3)

            # Score bar
            score_pct = max(0, min(1, r["score"]))
            bar_w     = int(score_pct * 80)
            bar_color = t["accent"] if score_pct > 0.5 else t["fg_dim"]

            header = tk.Frame(card, bg=t["bg2"])
            header.pack(fill="x", padx=8, pady=(6,2))

            score_bar = tk.Frame(header, bg=bar_color, width=bar_w, height=3)
            score_bar.pack(side="left", pady=(4,0))

            tk.Label(header, text=f"  {r['score']:.3f}",
                     font=t["font_small"], bg=t["bg2"], fg=bar_color).pack(side="left")

            path_lbl = tk.Label(header, text=r["path"],
                                 font=("Courier New", 10, "bold"),
                                 bg=t["bg2"], fg=t["fg_bright"],
                                 cursor="hand2")
            path_lbl.pack(side="left", padx=8)
            path_lbl.bind("<Button-1>", lambda e, p=r["path"]: self._open_file(p))

            copy_btn = tk.Label(header, text="⎘ COPY", font=t["font_small"],
                                bg=t["bg2"], fg=t["fg_dim"], cursor="hand2")
            copy_btn.pack(side="left", padx=4)
            copy_btn.bind("<Button-1>", lambda e, p=r["path"]: self._copy(p))

            preview = tk.Label(card, text=r["preview"][:200].replace("\n", "  "),
                                font=t["font_small"], bg=t["bg2"], fg=t["fg"],
                                anchor="w", justify="left", wraplength=700)
            preview.pack(fill="x", padx=12, pady=(0,6))

            # Hover
            for w in [card, header, preview]:
                w.bind("<Enter>", lambda e, c=card: c.config(
                    highlightbackground=self.t["accent"]))
                w.bind("<Leave>", lambda e, c=card: c.config(
                    highlightbackground=self.t["border"]))

    def _copy(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status(f"Copied: {text}")

    def _copy_all(self):
        results = getattr(self, "_last_results", [])
        if not results:
            self._set_status("No results to copy.")
            return
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] [{r['score']:.3f}] {r['path']}")
            lines.append(f"     {r['preview'][:120].strip()}")
            lines.append("")
        text = "\n".join(lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status(f"Copied {len(results)} results to clipboard.")

    def _open_file(self, rel_path):
        for base in self.scan_dirs:
            full = os.path.join(base, rel_path)
            if os.path.exists(full):
                if sys.platform == "darwin":
                    subprocess.run(["open", full])
                elif sys.platform == "win32":
                    os.startfile(full)
                else:
                    subprocess.run(["xdg-open", full])
                return

    # ── INDEX ─────────────────────────────────────────────────────────────────

    def _add_dir(self):
        d = filedialog.askdirectory(title="Select codebase directory")
        if d and d not in self.scan_dirs:
            self.scan_dirs.append(d)
            self.dir_listbox.insert("end", d)
            self._save_config()

    def _remove_dir(self):
        sel = self.dir_listbox.curselection()
        if sel:
            idx = sel[0]
            self.scan_dirs.pop(idx)
            self.dir_listbox.delete(idx)
            self._save_config()

    def _restore_dirs(self):
        self.dir_listbox.delete(0, "end")
        for d in self.scan_dirs:
            self.dir_listbox.insert("end", d)

    def _start_index(self):
        if not self.scan_dirs:
            self._set_status("Add a codebase directory first.")
            return
        self.index_btn.config(state="disabled")
        self.engine.build_index(
            self.scan_dirs, self.index_dir,
            on_status=self._set_status,
            on_done=self._on_index_done
        )

    def _on_index_done(self):
        self.root.after(0, lambda: self.index_btn.config(state="normal"))
        self.root.after(0, self._update_stats)

    def _open_browser(self):
        webbrowser.open(f"http://localhost:5050")

    # ── ENGINE ────────────────────────────────────────────────────────────────

    def _init_engine(self):
        self.engine.load(self.index_dir, self.model_path, self._set_status)

    # ── STATUS / STATS ────────────────────────────────────────────────────────

    def _set_status(self, msg):
        try:
            self.root.after(0, lambda: self.status_var.set(f"  {msg}"))
        except: pass

    def _update_stats(self):
        n = len(self.engine.metadata)
        if n:
            self.stats_lbl.config(
                text=f"{n:,} chunks\n{len(self.scan_dirs)} dir(s)\nModel: {Path(self.model_path).name}"
            )

    # ── SCROLL ────────────────────────────────────────────────────────────────

    def _on_results_configure(self, event):
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.results_canvas.itemconfig(self.results_window, width=event.width)

    def _on_mousewheel(self, event):
        self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    # ── CONFIG PERSIST ────────────────────────────────────────────────────────

    def _config_path(self):
        return Path.home() / ".trit-search" / "config.json"

    def _save_config(self):
        cfg = {"theme": self.theme_name, "scan_dirs": self.scan_dirs}
        self._config_path().parent.mkdir(parents=True, exist_ok=True)
        self._config_path().write_text(json.dumps(cfg))

    def _load_config(self):
        try:
            cfg = json.loads(self._config_path().read_text())
            self.scan_dirs = cfg.get("scan_dirs", [])
            theme = cfg.get("theme", "Matrix")
            if theme in THEMES:
                self.theme_name = theme
                self.t = THEMES[theme]
            self._restore_dirs()
        except: pass


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app  = TritSearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
