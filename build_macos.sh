#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-.venv/bin/python}
OUT=${OUT:-build/macos}

echo "Using python: $PY"
mkdir -p "$OUT"

echo "Installing/ensuring Nuitka..."
$PY -m pip install --upgrade pip setuptools wheel
$PY -m pip install nuitka

echo "Building macOS app bundle (onefile app)..."
LTO_ARG=""
if [ -n "${LTO:-}" ]; then
  LTO_ARG="--lto=${LTO}"
fi

$PY -m nuitka --onefile --macos-create-app-bundle --output-dir="$OUT" --output-filename=mhrqi-gui \
  $LTO_ARG --python-flag=no_docstrings --python-flag=no_asserts --include-data-dir=resources=resources run_gui.py

# codesign / notarize steps must be done manually using your Apple Developer account

echo "Build finished. Check $OUT for artifacts."
