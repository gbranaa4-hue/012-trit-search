"""
OBSERVE — Cross-Platform Build Script
Builds OBSERVE for Windows, Mac, and Linux.

On Windows: builds Windows exe, creates Mac/Linux build scripts to run remotely
On Mac:     builds Mac .app
On Linux:   builds Linux binary

Usage:
  python build_all.py          Build for current platform
  python build_all.py --all    Build + create scripts for other platforms
"""

import os, sys, shutil, subprocess, platform
from pathlib import Path

ROOT       = Path(__file__).parent
DIST_DIR   = ROOT / "dist"
BUILD_DIR  = ROOT / "build"
MODEL_DIR  = ROOT / "models" / "code-minilm"
ICON_ICO   = ROOT / "icon.ico"    # Windows icon (optional)
ICON_ICNS  = ROOT / "icon.icns"   # Mac icon (optional)
APP_NAME   = "OBSERVE"

def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def ensure_pyinstaller():
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        run([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build_windows():
    print(f"\n[Windows] Building {APP_NAME}.exe...")
    ensure_pyinstaller()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--noconsole",
        "--name", APP_NAME,
        "--distpath", str(DIST_DIR / "windows"),
        "--workpath", str(BUILD_DIR / "windows"),
        "--specpath", str(BUILD_DIR),
    ]

    # Bundle fine-tuned model if it exists
    if MODEL_DIR.exists():
        cmd += ["--add-data", f"{MODEL_DIR};models/code-minilm"]
        print(f"  Bundling model: {MODEL_DIR}")
    else:
        print("  No fine-tuned model found — bundling base model name only")

    if ICON_ICO.exists():
        cmd += ["--icon", str(ICON_ICO)]

    cmd.append(str(ROOT / "trit_app.py"))
    run(cmd)

    exe = DIST_DIR / "windows" / f"{APP_NAME}.exe"
    size = exe.stat().st_size / 1e6 if exe.exists() else 0
    print(f"\n  Windows build: {exe}  ({size:.0f} MB)")
    return exe

def build_mac():
    print(f"\n[Mac] Building {APP_NAME}.app...")
    ensure_pyinstaller()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onedir",          # .app bundle (better on Mac than onefile)
        "--windowed",
        "--name", APP_NAME,
        "--distpath", str(DIST_DIR / "mac"),
        "--workpath", str(BUILD_DIR / "mac"),
        "--specpath", str(BUILD_DIR),
        "--osx-bundle-identifier", "com.observe.code-search",
    ]

    if MODEL_DIR.exists():
        cmd += ["--add-data", f"{MODEL_DIR}:models/code-minilm"]

    if ICON_ICNS.exists():
        cmd += ["--icon", str(ICON_ICNS)]

    cmd.append(str(ROOT / "trit_app.py"))
    run(cmd)

    app = DIST_DIR / "mac" / f"{APP_NAME}.app"
    print(f"\n  Mac build: {app}")

    # Create DMG for easy distribution
    _create_dmg(app)
    return app

def _create_dmg(app_path):
    """Create a .dmg installer for Mac."""
    dmg_path = DIST_DIR / "mac" / f"{APP_NAME}.dmg"
    try:
        run(["hdiutil", "create", "-volname", APP_NAME,
             "-srcfolder", str(app_path),
             "-ov", "-format", "UDZO",
             str(dmg_path)])
        print(f"  DMG: {dmg_path}")
    except Exception as e:
        print(f"  DMG creation skipped: {e}")

def build_linux():
    print(f"\n[Linux] Building {APP_NAME} binary...")
    ensure_pyinstaller()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", APP_NAME,
        "--distpath", str(DIST_DIR / "linux"),
        "--workpath", str(BUILD_DIR / "linux"),
        "--specpath", str(BUILD_DIR),
        # Linux needs display — use tk
        "--hidden-import", "tkinter",
    ]

    if MODEL_DIR.exists():
        cmd += ["--add-data", f"{MODEL_DIR}:models/code-minilm"]

    cmd.append(str(ROOT / "trit_app.py"))
    run(cmd)

    binary = DIST_DIR / "linux" / APP_NAME
    if binary.exists():
        binary.chmod(0o755)
        size = binary.stat().st_size / 1e6
        print(f"\n  Linux build: {binary}  ({size:.0f} MB)")
    return binary

def create_mac_build_script():
    """Script to run on a Mac to build the Mac version."""
    script = ROOT / "build_mac.sh"
    script.write_text(f"""#!/bin/bash
# Run this on a Mac to build {APP_NAME}.app
# Copy the whole 012-ternary folder to the Mac first, then run this.

cd "$(dirname "$0")"

echo "Installing dependencies..."
pip3 install sentence-transformers faiss-cpu flask torch pyinstaller

echo "Building {APP_NAME}.app..."
python3 build_all.py

echo ""
echo "Done! Find your app in dist/mac/{APP_NAME}.app"
echo "Or the DMG installer at dist/mac/{APP_NAME}.dmg"
""")
    script.chmod(0o755)
    print(f"\n  Mac build script: {script}")
    print("  Copy 012-ternary folder to Mac and run: ./build_mac.sh")

def create_linux_build_script():
    """Script to run on Linux."""
    script = ROOT / "build_linux.sh"
    script.write_text(f"""#!/bin/bash
# Run this on Linux to build {APP_NAME} binary

cd "$(dirname "$0")"

echo "Installing system deps (Ubuntu/Debian)..."
sudo apt-get install -y python3-tk python3-pip 2>/dev/null || true

echo "Installing Python deps..."
pip3 install sentence-transformers faiss-cpu flask torch pyinstaller

echo "Building {APP_NAME}..."
python3 build_all.py

echo ""
echo "Done! Find your binary at dist/linux/{APP_NAME}"
echo "Make it executable: chmod +x dist/linux/{APP_NAME}"
""")
    script.chmod(0o755)
    print(f"\n  Linux build script: {script}")

def create_github_actions():
    """
    GitHub Actions workflow — builds all 3 platforms automatically
    on every release. Push to GitHub and it builds for free.
    """
    workflows_dir = ROOT / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    workflow = workflows_dir / "build.yml"
    workflow.write_text(f"""name: Build {APP_NAME}

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install deps
        run: pip install sentence-transformers faiss-cpu flask torch pyinstaller
      - name: Build
        run: python build_all.py --windows
      - name: Upload
        uses: actions/upload-artifact@v4
        with:
          name: {APP_NAME}-Windows
          path: dist/windows/{APP_NAME}.exe

  build-mac:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install deps
        run: pip install sentence-transformers faiss-cpu flask torch pyinstaller
      - name: Build
        run: python build_all.py --mac
      - name: Upload
        uses: actions/upload-artifact@v4
        with:
          name: {APP_NAME}-Mac
          path: dist/mac/{APP_NAME}.dmg

  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install deps
        run: |
          sudo apt-get install -y python3-tk
          pip install sentence-transformers faiss-cpu flask torch pyinstaller
      - name: Build
        run: python build_all.py --linux
      - name: Upload
        uses: actions/upload-artifact@v4
        with:
          name: {APP_NAME}-Linux
          path: dist/linux/{APP_NAME}

  release:
    needs: [build-windows, build-mac, build-linux]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    steps:
      - uses: actions/download-artifact@v4
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            {APP_NAME}-Windows/{APP_NAME}.exe
            {APP_NAME}-Mac/{APP_NAME}.dmg
            {APP_NAME}-Linux/{APP_NAME}
          body: |
            ## {APP_NAME} ${{{{ github.ref_name }}}}
            Local semantic code search. Free. Private. No cloud.

            ### Download
            - **Windows**: {APP_NAME}.exe
            - **Mac**: {APP_NAME}.dmg
            - **Linux**: {APP_NAME} (chmod +x first)

            ### Usage
            1. Open {APP_NAME}
            2. Click + ADD DIR and select your codebase
            3. Click INDEX CODEBASE
            4. Search by meaning
""")
    print(f"\n  GitHub Actions workflow: {workflow}")
    print("  Push to GitHub + tag a release -> all 3 builds happen automatically")

def print_summary():
    print(f"""
+==============================================================+
|                  {APP_NAME} — Build Summary                       |
+==============================================================+

  dist/
  |-- windows/{APP_NAME}.exe    Windows 10/11
  |-- mac/{APP_NAME}.app        macOS 12+
  |-- mac/{APP_NAME}.dmg        Mac installer
  `-- linux/{APP_NAME}          Ubuntu/Debian/Arch

  To build all platforms automatically (free):
  1. Push this folder to a GitHub repo
  2. Run: git tag v1.0 && git push --tags
  3. GitHub Actions builds all 3 in ~10 minutes
  4. Download from GitHub Releases page

  Or build manually on each platform:
  - Windows: python build_all.py --windows
  - Mac:     ./build_mac.sh
  - Linux:   ./build_linux.sh
""")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",     action="store_true", help="Build + create all scripts")
    parser.add_argument("--windows", action="store_true", help="Build Windows only")
    parser.add_argument("--mac",     action="store_true", help="Build Mac only")
    parser.add_argument("--linux",   action="store_true", help="Build Linux only")
    parser.add_argument("--ci",      action="store_true", help="Create GitHub Actions workflow")
    args = parser.parse_args()

    DIST_DIR.mkdir(exist_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)

    system = platform.system()

    if args.ci:
        create_github_actions()

    elif args.windows or (not any([args.mac, args.linux, args.all]) and system == "Windows"):
        build_windows()
        create_mac_build_script()
        create_linux_build_script()
        create_github_actions()
        print_summary()

    elif args.mac or (not any([args.windows, args.linux, args.all]) and system == "Darwin"):
        build_mac()

    elif args.linux or (not any([args.windows, args.mac, args.all]) and system == "Linux"):
        build_linux()

    elif args.all:
        if system == "Windows":
            build_windows()
        elif system == "Darwin":
            build_mac()
        elif system == "Linux":
            build_linux()
        create_mac_build_script()
        create_linux_build_script()
        create_github_actions()
        print_summary()
