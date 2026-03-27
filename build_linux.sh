#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-.venv/bin/python}
OUT=${OUT:-build/linux}

echo "Using python: $PY"
mkdir -p "$OUT"

echo "Installing/ensuring Nuitka..."
$PY -m pip install --upgrade pip setuptools wheel
$PY -m pip install nuitka

echo "Building mhrqi CLI (onefile)..."
$LTO_ARG=""
if [ -n "${LTO:-}" ]; then
  LTO_ARG="--lto=${LTO}"
fi

$PY -m nuitka --onefile --remove-output --output-dir="$OUT" --output-filename=mhrqi-cli \
  $LTO_ARG --python-flag=no_docstrings --python-flag=no_asserts \
  --nofollow-import-to=qiskit --nofollow-import-to=qiskit_aer --nofollow-import-to=torch \
  --include-data-dir=resources=resources mhrqi/cli/main.py

# strip and compress if available
if command -v strip >/dev/null 2>&1; then
  strip "$OUT/mhrqi-cli" || true
fi
if command -v upx >/dev/null 2>&1; then
  upx --best --lzma "$OUT/mhrqi-cli"
fi

echo "Build finished. Check $OUT for artifacts."
