#!/usr/bin/env bash
# cleanup_old_runs.sh — Borra runs antiguos del directorio outputs/
#
# Uso:
#   scripts/cleanup_old_runs.sh [--days N] [--dry-run]
#
# Por defecto borra runs de más de 30 días.
# Con --dry-run solo lista qué borraría.

set -euo pipefail

DAYS=30
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --days)
            DAYS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Uso: $0 [--days N] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

OUTPUTS_DIR="$(dirname "$0")/../outputs"
OUTPUTS_DIR="$(realpath "$OUTPUTS_DIR")"

if [[ ! -d "$OUTPUTS_DIR" ]]; then
    echo "Directorio outputs/ no existe: $OUTPUTS_DIR" >&2
    exit 0
fi

echo "Limpiando runs con más de $DAYS días en $OUTPUTS_DIR"

find "$OUTPUTS_DIR" -maxdepth 1 -type d -name "run_*" -mtime +"$DAYS" -print0 | \
    while IFS= read -r -d '' dir; do
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[DRY] Borraría: $dir"
        else
            echo "Borrando: $dir"
            rm -rf "$dir"
        fi
    done
