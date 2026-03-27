# COMPILE.Md - Nuitka compile templates and notes

Purpose
-------
Store reproducible Nuitka compile commands, per-platform prerequisites, and notes to produce small, self-contained executables for this project.

Quick audit (repo-specific notes)
---------------------------------
- Heavy / large dependencies present in this codebase: `qiskit`, `qiskit-aer`, `torch`, `opencv-python`, `dearpygui`, `bm3d`, `brisque`, `scikit-video`.
- These will dramatically increase binary size. If you need small artifacts, build a *minimal* variant that excludes simulator/GUI/ML packages and only includes core `numpy`/`scipy`/`pillow` etc.
- Strategies to reduce size:
  - Create two build targets: `minimal` (CLI/core features) and `full` (everything). Build `minimal` for distribution where possible.
  - Move heavy imports into function scope (lazy import) so Nuitka can omit them when unused.
  - Use `--nofollow-import-to` to explicitly avoid following heavy optional modules when building a minimal binary (danger: runtime errors if you actually use those modules).

General recommendations
----------------------
- Primary tool: `Nuitka` (onefile or standalone mode). Use `--onefile` to bundle into a single binary, then `UPX` for optional compression.
- Minimization flags we typically use:
  - `--onefile` (bundle into one executable)
  - `--remove-output` (clear previous outputs)
  - `--output-dir=build/<platform>` (where artifacts go)
  - `--output-filename=<name>` (control binary name)
  - `--lto` (link-time optimization)
  - `--python-flag=no_docstrings` and `--python-flag=no_asserts` (reduce size)
  - `--enable-plugin=numpy` (help for numpy-based packages)
  - `--include-data-dir=resources=resources` (include images/resources folder)
  - `--nofollow-import-to=<module>` (exclude heavy optional imports)

Important: always use the Python interpreter from the venv you intend to ship with. Example venv python on Windows for this workspace:

    c:/Users/Admin/Documents/source_repos/MHRQI-0.1.0/.venv/Scripts/python.exe

Windows (PowerShell) - prototype (GUI: `run_gui.py`)
---------------------------------------------------
Steps (one-time):

1. Ensure a usable C compiler is installed. Recommended: Visual Studio 2022 (C++ build tools).
2. Install UPX (optional, for compression). Example: install manually or with Chocolatey: `choco install upx`.
3. From project root (PowerShell):

```powershell
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install nuitka
```

Build (minimal / example):

```powershell
$PY = "c:/Users/Admin/Documents/source_repos/MHRQI-0.1.0/.venv/Scripts/python.exe"
$PY -m nuitka --onefile --remove-output --output-dir=build/windows --output-filename=mhrqi-gui.exe \
  --windows-console-mode=disable --lto --python-flag=no_docstrings --python-flag=no_asserts \
  --enable-plugin=numpy --include-data-dir=resources=resources run_gui.py
```

Optional: exclude heavy packages for a *minimal* build (use with care):

```powershell
$PY -m nuitka --onefile --remove-output --output-dir=build/windows --output-filename=mhrqi-cli.exe \
  --lto --python-flag=no_docstrings --python-flag=no_asserts \
  --nofollow-import-to=qiskit --nofollow-import-to=qiskit_aer --nofollow-import-to=torch \
  --nofollow-import-to=dearpygui --include-data-dir=resources=resources mhrqi/cli/main.py
```

After build (optional compression):

```powershell
# compress with UPX if installed (may break signing/notarization)
if (Get-Command upx -ErrorAction SilentlyContinue) { upx --best --lzma build\windows\mhrqi-gui.exe }
```

Linux (bash) - prototype (CLI / minimal)
----------------------------------------
Prereqs: `gcc`/`g++`, `python3-dev` matching your Python, `strip`, `upx` (optional).

```bash
PY=.venv/bin/python
$PY -m pip install --upgrade pip setuptools wheel
$PY -m pip install nuitka

# Build minimal CLI (example)
$PY -m nuitka --onefile --remove-output --output-dir=build/linux --output-filename=mhrqi-cli \
  --lto --python-flag=no_docstrings --python-flag=no_asserts \
  --nofollow-import-to=qiskit --nofollow-import-to=qiskit_aer --nofollow-import-to=torch \
  --include-data-dir=resources=resources mhrqi/cli/main.py

# strip and compress
strip build/linux/mhrqi-cli || true
if command -v upx >/dev/null 2>&1; then
  upx --best --lzma build/linux/mhrqi-cli
fi
```

macOS (bash / zsh) - prototype (GUI as app bundle)
--------------------------------------------------
Prereqs: Xcode (command-line tools), system Python or Homebrew Python (be careful with portability), `upx` (optional).

```bash
PY=.venv/bin/python
$PY -m pip install --upgrade pip setuptools wheel
$PY -m pip install nuitka

# Create macOS app bundle (useful for GUI apps)
$PY -m nuitka --onefile --macos-create-app-bundle --output-dir=build/macos --output-filename=mhrqi-gui \
  --lto --python-flag=no_docstrings --python-flag=no_asserts --include-data-dir=resources=resources run_gui.py

# Optional: codesign/notarize the created bundle before distribution (see Apple docs)
# no need sign or notarize because it is for professor eyes only.
```

Notes & troubleshooting
-----------------------
- If module/import errors occur at compile time, re-run with `--report=compilation-report.xml` to inspect what Nuitka included/excluded.
- To force-include a module: use `--include-module=<modulename>` or `--include-package-data`/`--include-data-files` as needed.
- Some binary deps (e.g., `qiskit-aer`, `torch`) bring native libraries (DLL/.so/.dylib). Use `--include-dll` or package config if Nuitka does not pick them up automatically.
- UPX compression may reduce size but can break code signing and sometimes runtime; test thoroughly.

Templates are intentionally conservative. Adjust `--nofollow-import-to` and data-file inclusion to suit the variant you want to distribute.

Files added by this change
-------------------------
- `build_windows.ps1` - PowerShell build helper
- `build_linux.sh` - Linux build helper
- `build_macos.sh` - macOS build helper

--
Last updated: 2026-03-27
