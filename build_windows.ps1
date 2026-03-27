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
& $PY -m pip install "nuitka[onefile]" zstandard

Write-Host "Building run_gui (onefile, GUI mode)..."
# Check for Dependency Walker (depends.exe). Nuitka requires it on Windows for onefile/standalone.
$depends = Get-Command depends.exe -ErrorAction SilentlyContinue
if ($depends) {
  Write-Host "Dependency Walker found: $($depends.Path)"
  & $PY -m nuitka --onefile --remove-output --output-dir=$OUT --output-filename=mhrqi-gui.exe --windows-console-mode=disable --python-flag=no_docstrings --python-flag=no_asserts --enable-plugin=numpy --include-data-dir=resources=resources run_gui.py
} else {
  Write-Host "Dependency Walker not found. Attempting choco install dependencywalker.portable..."
  try {
    choco install dependencywalker.portable -y -r
  } catch {
    Write-Host "choco install failed: $_"
  }

  $depends = Get-Command depends.exe -ErrorAction SilentlyContinue
  if ($depends) {
    Write-Host "Dependency Walker installed at: $($depends.Path)"
    $nuitkaCmd = "`"$PY`" -m nuitka --onefile --remove-output --output-dir=$OUT --output-filename=mhrqi-gui.exe --windows-console-mode=disable --python-flag=no_docstrings --python-flag=no_asserts --enable-plugin=numpy --include-data-dir=resources=resources run_gui.py"
    Write-Host "Running: cmd /c echo Y | $nuitkaCmd"
    cmd.exe /c "echo Y | $nuitkaCmd"
  } else {
    Write-Host "Dependency Walker still not found. Piping 'Y' to Nuitka to allow automatic download (non-interactive)."
    $nuitkaCmd = "`"$PY`" -m nuitka --onefile --remove-output --output-dir=$OUT --output-filename=mhrqi-gui.exe --windows-console-mode=disable --python-flag=no_docstrings --python-flag=no_asserts --enable-plugin=numpy --include-data-dir=resources=resources run_gui.py"
    Write-Host "Running: cmd /c echo Y | $nuitkaCmd"
    cmd.exe /c "echo Y | $nuitkaCmd"
  }
}

if (Get-Command upx -ErrorAction SilentlyContinue) {
  Write-Host "Compressing with UPX..."
  upx --best --lzma "$OUT\mhrqi-gui.exe"
} else {
  Write-Host "UPX not found; skipping compression"
}

Write-Host "Build finished. Check $OUT for artifacts."
