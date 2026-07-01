#!/bin/bash
# Run this on Linux to build OBSERVE binary

cd "$(dirname "$0")"

echo "Installing system deps (Ubuntu/Debian)..."
sudo apt-get install -y python3-tk python3-pip 2>/dev/null || true

echo "Installing Python deps..."
pip3 install sentence-transformers faiss-cpu flask torch pyinstaller

echo "Building OBSERVE..."
python3 build_all.py --linux

echo ""
echo "Done! Find your binary at dist/linux/OBSERVE"
echo "Make it executable: chmod +x dist/linux/OBSERVE"
