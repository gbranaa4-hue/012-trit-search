#!/bin/bash
# Run this on Linux to build TritSearch binary

cd "$(dirname "$0")"

echo "Installing system deps (Ubuntu/Debian)..."
sudo apt-get install -y python3-tk python3-pip 2>/dev/null || true

echo "Installing Python deps..."
pip3 install sentence-transformers faiss-cpu flask torch pyinstaller

echo "Building TritSearch..."
python3 build_all.py

echo ""
echo "Done! Find your binary at dist/linux/TritSearch"
echo "Make it executable: chmod +x dist/linux/TritSearch"
