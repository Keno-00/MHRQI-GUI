param(
  [string]$PY = "c:/Users/Admin/Documents/source_repos/MHRQI-0.1.0/.venv/Scripts/python.exe",
  [string]$OUT = "build/windows"
)

Set-StrictMode -Version Latest
Write-Host "Using Python: $PY"

# Ensure build dir
if (!(Test-Path $OUT)) { New-Item -ItemType Directory -Path $OUT | Out-Null }

Write-Host "Installing/ensuring Nuitka..."
& $PY -m pip install --upgrade pip setuptools wheel
& $PY -m pip install nuitka

Write-Host "Building run_gui (onefile, GUI mode)..."
& $PY -m nuitka --onefile --remove-output --output-dir=$OUT --output-filename=mhrqi-gui.exe --windows-console-mode=disable --python-flag=no_docstrings --python-flag=no_asserts --enable-plugin=numpy --include-data-dir=resources=resources run_gui.py

if (Get-Command upx -ErrorAction SilentlyContinue) {
  Write-Host "Compressing with UPX..."
  upx --best --lzma "$OUT\mhrqi-gui.exe"
} else {
  Write-Host "UPX not found; skipping compression"
}

Write-Host "Build finished. Check $OUT for artifacts."
