#!/usr/bin/env bash
set -euo pipefail

# publish_portable.sh
# Build a portable BoneAgeX package (Python 3.6, PyInstaller)
# Usage: ./publish_portable.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Detect OS and path separator for PyInstaller --add-data
UNAME=$(uname -s 2>/dev/null || echo "Windows")
if [[ "$UNAME" == "Linux" || "$UNAME" == "Darwin" ]]; then
  SEP=":"
else
  SEP=";"
fi

# Find a Python 3.6 executable
PY_CMD=""
if command -v py >/dev/null 2>&1 && py -3.6 -c 'import sys' >/dev/null 2>&1; then
  PY_CMD="py -3.6"
elif command -v python3.6 >/dev/null 2>&1 && python3.6 -c 'import sys' >/dev/null 2>&1; then
  PY_CMD="python3.6"
elif command -v python >/dev/null 2>&1 && python -c 'import sys; v=sys.version_info; sys.exit(0 if (v.major==3 and v.minor==6) else 1)' >/dev/null 2>&1; then
  PY_CMD="python"
else
  echo "ERROR: Python 3.6 not found on PATH. Install Python 3.6 or make it available as 'py -3.6' or 'python3.6'."
  exit 1
fi

echo "Using Python command: $PY_CMD"

# Ensure PyInstaller is available for that Python
if ! $PY_CMD -m pip show pyinstaller >/dev/null 2>&1; then
  echo "PyInstaller not found for $PY_CMD. Installing into user site-packages..."
  $PY_CMD -m pip install --user pyinstaller
fi

DIST_DIR="BoneAgeX"
WORK_DIR="build_temp"
SPEC_FILE="BoneAgeX.spec"
LOGFILE="build.log"

# Clean old artifacts
echo "Cleaning old build artifacts..."
rm -rf "$DIST_DIR" "$WORK_DIR" "$SPEC_FILE" "$LOGFILE" 2>/dev/null || true

# Build with PyInstaller
echo "Starting PyInstaller build (this may take several minutes)..."

# Build command components
ADD_DATA=(
  "data${SEP}data"
  "graphs${SEP}graphs"
  "eval_gui_data${SEP}eval_gui_data"
  "utils${SEP}utils"
  "ssh-final_models${SEP}ssh-final_models"
  "Atlas_of_Hand_Bone_Age.pdf${SEP}."
)

PYINSTALLER_CMD=( $PY_CMD -m PyInstaller eval_gui.py -w -n BoneAgeX --distpath="$DIST_DIR" --workpath="$WORK_DIR" -y --log-level=INFO )
for d in "${ADD_DATA[@]}"; do
  PYINSTALLER_CMD+=( --add-data "$d" )
done

# Execute and tee output to build.log
# Use eval to allow $PY_CMD with space (e.g. "py -3.6") to run correctly
echo "Running: ${PYINSTALLER_CMD[*]}"
eval "${PYINSTALLER_CMD[*]}" 2>&1 | tee "$LOGFILE"

# Post-build: verify
if [ -d "$DIST_DIR" ] && [ -f "$DIST_DIR/BoneAgeX.exe" ]; then
  echo "Build finished successfully. Dist folder: $ROOT_DIR/$DIST_DIR"
else
  echo "Build finished but executable not found. See $LOGFILE for details."
  exit 2
fi

# Copy or generate deployment README if not present
if [ -f README_DEPLOYMENT.md ]; then
  cp README_DEPLOYMENT.md "$DIST_DIR/" 2>/dev/null || true
fi

# Create zip archive for distribution (use zip if available; otherwise PowerShell Compress-Archive on Windows)
ZIP_NAME="BoneAgeX_portable_$(date +%Y%m%d%H%M%S).zip"
if command -v zip >/dev/null 2>&1; then
  echo "Creating $ZIP_NAME with zip..."
  zip -r "$ZIP_NAME" "$DIST_DIR" >/dev/null
elif command -v powershell >/dev/null 2>&1; then
  echo "Creating $ZIP_NAME using PowerShell Compress-Archive..."
  powershell -NoProfile -Command "Compress-Archive -Path '$DIST_DIR\*' -DestinationPath '$ZIP_NAME' -Force"
else
  echo "No zip utility found. Skipping archive creation. Dist is at: $DIST_DIR"
  echo "Build log: $LOGFILE"
  exit 0
fi

echo "Package created: $ROOT_DIR/$ZIP_NAME"
echo "Done."
