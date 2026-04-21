#!/usr/bin/env bash
# run_exp7_rudiger.sh — Ejecuta exp7 VRAM profiling en servidor Rudiger.
# Uso (en Rudiger):
#   cd ~/sam && bash scripts/run_exp7_rudiger.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_PYTHON="$(pwd)/venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "venv no encontrado. Corre antes: ./run.sh install" >&2
    exit 1
fi
mkdir -p outputs/exp7_vram
echo "Ejecutando exp7 VRAM profiling (puede tardar ~30 min)..."
"$VENV_PYTHON" experiments/exp7_vram_profiling.py 2>&1 | tee outputs/exp7_vram_profile.txt
echo "Output en outputs/exp7_vram_profile.txt"
