#!/bin/bash
# 012 Trit Search — Mac Installer
# Run this once on any Mac to set up the semantic search tool.
#
# Usage:
#   curl -O https://your-host/trit_install.sh  (or copy this file)
#   chmod +x trit_install.sh
#   ./trit_install.sh /path/to/your/codebase

set -e

CODEBASE_DIR="${1:-$HOME}"
INSTALL_DIR="$HOME/.trit-search"
VENV_DIR="$INSTALL_DIR/venv"
PORT=5050

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║           012 Trit Search — Mac Installer            ║"
echo "║     Local semantic code search. Free. Private.       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Codebase : $CODEBASE_DIR"
echo "  Install   : $INSTALL_DIR"
echo "  Port      : $PORT"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "Python3 not found. Install from https://python.org"
    exit 1
fi

PYTHON=$(command -v python3)
PY_VERSION=$($PYTHON --version 2>&1)
echo "  Python: $PY_VERSION"

# ── Create install directory ──────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/search_index"

# ── Create virtual environment ────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "  Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

# ── Install dependencies ──────────────────────────────────────────────────────
echo ""
echo "  Installing dependencies (first time takes 2-3 minutes)..."
$PIP install --quiet --upgrade pip
$PIP install --quiet sentence-transformers faiss-cpu flask torch

echo "  Dependencies installed."

# ── Copy trit_search.py ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/trit_search.py" ]; then
    cp "$SCRIPT_DIR/trit_search.py" "$INSTALL_DIR/trit_search.py"
    echo "  Copied trit_search.py"
else
    echo ""
    echo "  ERROR: trit_search.py not found next to this installer."
    echo "  Put trit_search.py in the same folder as trit_install.sh"
    exit 1
fi

# ── Copy fine-tuned model ─────────────────────────────────────────────────────
MODEL_SRC="$SCRIPT_DIR/models/code-minilm"

if [ -d "$MODEL_SRC" ]; then
    echo "  Copying fine-tuned model (~90MB)..."
    cp -r "$MODEL_SRC" "$INSTALL_DIR/models/"
    echo "  Model copied."
else
    echo "  Fine-tuned model not found — will use base MiniLM (still works, slightly less accurate)"
fi

# ── Patch INCLUDE_ONLY_DIRS for this user's codebase ─────────────────────────
SEARCH_SCRIPT="$INSTALL_DIR/trit_search.py"

# Replace the INCLUDE_ONLY_DIRS with the user's actual codebase path
python3 - <<EOF
import re

path = "$INSTALL_DIR/trit_search.py"
code = open(path).read()

# Replace INCLUDE_ONLY_DIRS content
new_dirs = '''INCLUDE_ONLY_DIRS = [
    r"$CODEBASE_DIR",
]'''

code = re.sub(
    r'INCLUDE_ONLY_DIRS\s*=\s*\[.*?\]',
    new_dirs,
    code,
    flags=re.DOTALL
)

# Point model path to local install
code = code.replace(
    r'C:\Users\gbran\OneDrive\Documents\012-ternary\models\code-minilm',
    '$INSTALL_DIR/models/code-minilm'
)

open(path, 'w').write(code)
print("  Config patched for: $CODEBASE_DIR")
EOF

# ── Create launcher script ────────────────────────────────────────────────────
cat > "$INSTALL_DIR/search.sh" << 'LAUNCHER'
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/venv/bin/python"
cd "$DIR"

case "$1" in
    index)
        echo "Indexing codebase..."
        $PYTHON trit_search.py --index
        ;;
    reindex)
        echo "Force reindexing..."
        rm -rf search_index
        $PYTHON trit_search.py --index
        ;;
    serve)
        echo "Starting search server at http://localhost:5050"
        $PYTHON trit_search.py --serve
        ;;
    search)
        $PYTHON trit_search.py --search "$2" --k 8
        ;;
    *)
        echo ""
        echo "012 Trit Search"
        echo "  ./search.sh index         Index your codebase"
        echo "  ./search.sh serve         Start browser UI at localhost:5050"
        echo "  ./search.sh search 'query'  Search from terminal"
        echo "  ./search.sh reindex       Rebuild index from scratch"
        echo ""
        ;;
esac
LAUNCHER

chmod +x "$INSTALL_DIR/search.sh"

# ── Create macOS app shortcut ─────────────────────────────────────────────────
APP_DIR="$HOME/Desktop/TritSearch.app"
mkdir -p "$APP_DIR/Contents/MacOS"

cat > "$APP_DIR/Contents/MacOS/TritSearch" << APPSCRIPT
#!/bin/bash
cd "$INSTALL_DIR"
"$VENV_DIR/bin/python" trit_search.py --serve &
sleep 2
open "http://localhost:$PORT"
APPSCRIPT

chmod +x "$APP_DIR/Contents/MacOS/TritSearch"

cat > "$APP_DIR/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>TritSearch</string>
    <key>CFBundleExecutable</key>
    <string>TritSearch</string>
    <key>CFBundleIdentifier</key>
    <string>com.012.trit-search</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
PLIST

# ── Run initial index ─────────────────────────────────────────────────────────
echo ""
echo "  Running initial index of: $CODEBASE_DIR"
echo "  (This takes 1-5 minutes depending on codebase size)"
echo ""
cd "$INSTALL_DIR"
$PYTHON trit_search.py --index

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  To search from terminal:"
echo "    $INSTALL_DIR/search.sh search 'your query'"
echo ""
echo "  To start browser UI:"
echo "    $INSTALL_DIR/search.sh serve"
echo "    then open http://localhost:$PORT"
echo ""
echo "  Or double-click TritSearch on your Desktop"
echo ""
echo "  To reindex after code changes:"
echo "    $INSTALL_DIR/search.sh index"
echo ""
