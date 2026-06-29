#!/bin/bash
# Run this on a Mac to build TritSearch.app
# Copy the whole 012-ternary folder to the Mac first, then run this.

cd "$(dirname "$0")"

echo "Installing dependencies..."
pip3 install sentence-transformers faiss-cpu flask torch pyinstaller

echo "Building TritSearch.app..."
python3 build_all.py

echo ""
echo "Done! Find your app in dist/mac/TritSearch.app"
echo "Or the DMG installer at dist/mac/TritSearch.dmg"
