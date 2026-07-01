#!/bin/bash
# Run this on a Mac to build OBSERVE.app
# Copy the whole 012-ternary folder to the Mac first, then run this.

cd "$(dirname "$0")"

echo "Installing dependencies..."
pip3 install sentence-transformers faiss-cpu flask torch pyinstaller

echo "Building OBSERVE.app..."
python3 build_all.py --mac

echo ""
echo "Done! Find your app in dist/mac/OBSERVE.app"
echo "Or the DMG installer at dist/mac/OBSERVE.dmg"
