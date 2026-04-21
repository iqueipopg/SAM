#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# SAM Rudiger runner (T16.4)
#
# Lanza la pipeline SAM (A1..A9) contra rudiger/config_rudiger.yaml y
# rudiger/queries.json. Pensado para que Eduardo lo ejecute en el
# servidor Rudiger tras haber rellenado los placeholders <RELLENAR_EDUARDO>
# del config_rudiger.yaml (modelos locales + r_library_path).
#
# Uso:
#   ./rudiger/run_rudiger.sh all                 # todas las queries
#   ./rudiger/run_rudiger.sh <query_id>          # una sola (ej: paula_nutrition)
#   ./rudiger/run_rudiger.sh list                # lista las query IDs disponibles
#   ./rudiger/run_rudiger.sh help                # ayuda
#
# Requiere:
#   - venv en <repo>/venv/ con SAM instalado
#   - rudiger/config_rudiger.yaml (T16.2) sin placeholders pendientes
#   - rudiger/queries.json (T16.3)
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Rutas absolutas (independientes del cwd) ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$REPO_ROOT/venv/bin/python"
CONFIG_YAML="$SCRIPT_DIR/config_rudiger.yaml"
QUERIES_JSON="$SCRIPT_DIR/queries.json"
RUNNER_PY="$SCRIPT_DIR/run_queries.py"
LOGS_DIR="$SCRIPT_DIR/logs"

# Versión: leer VERSION del repo, fallback a 0.6.0.
if [[ -f "$REPO_ROOT/VERSION" ]]; then
    SAM_VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/VERSION")"
else
    SAM_VERSION="0.6.0"
fi

# ── Colores ───────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'
    CYAN=$'\033[0;36m'
    NC=$'\033[0m'
else
    RED=""; GREEN=""; YELLOW=""; CYAN=""; NC=""
fi

log()  { printf '%s[rudiger]%s %s\n' "$GREEN"  "$NC" "$1"; }
warn() { printf '%s[rudiger]%s %s\n' "$YELLOW" "$NC" "$1"; }
err()  { printf '%s[rudiger]%s %s\n' "$RED"    "$NC" "$1" >&2; }

# ── Banner ────────────────────────────────────────────────────────────
print_banner() {
    printf '%s' "$CYAN"
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  SAM — Systematic Agent for Meta-analysis  (v${SAM_VERSION})"
    echo "  Ejecución en servidor: $(hostname)"
    echo "  Fecha: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Config:  $CONFIG_YAML"
    echo "  Queries: $QUERIES_JSON"
    echo "══════════════════════════════════════════════════════════════════════"
    printf '%s' "$NC"
}

# ── Validaciones previas ──────────────────────────────────────────────
# validate_files: checks que pueden hacerse para cualquier subcomando.
validate_files() {
    if [[ ! -x "$VENV_PY" ]]; then
        err "No encuentro el venv en $VENV_PY"
        err "Crea el entorno con './run.sh install' desde la raíz del repo."
        exit 3
    fi
    if [[ ! -f "$CONFIG_YAML" ]]; then
        err "Falta $CONFIG_YAML (esperado por T16.2)."
        exit 3
    fi
    if [[ ! -f "$QUERIES_JSON" ]]; then
        err "Falta $QUERIES_JSON (esperado por T16.3)."
        exit 3
    fi
    if [[ ! -f "$RUNNER_PY" ]]; then
        err "Falta $RUNNER_PY (wrapper Python T16.4)."
        exit 3
    fi
}

# validate_config_filled: sólo para subcomandos que lanzan pipeline.
# Aborta si el config aún tiene placeholders sin rellenar.
validate_config_filled() {
    if grep -q "<RELLENAR_EDUARDO>" "$CONFIG_YAML"; then
        err "config_rudiger.yaml contiene placeholders <RELLENAR_EDUARDO>."
        err "Rellena models.* y synthesis.r_library_path antes de lanzar."
        err "Ruta: $CONFIG_YAML"
        exit 4
    fi
}

# ── Subcomandos ───────────────────────────────────────────────────────
show_help() {
    cat <<EOF
Uso: $0 <subcomando>

Subcomandos:
  all            Ejecuta todas las queries de queries.json
  <query_id>     Ejecuta una sola query (ver ./run_rudiger.sh list)
  list           Lista las IDs disponibles en queries.json
  help           Muestra esta ayuda

Ejemplos:
  $0 all
  $0 paula_nutrition
  $0 cbt_depression_baseline

Logs: $LOGS_DIR/run_<timestamp>.log
Outputs: $REPO_ROOT/outputs/batch_rudiger_<timestamp>/
EOF
}

cmd_list() {
    # Lista las IDs usando el venv python (sin depender de jq).
    "$VENV_PY" -c "
import json, sys
with open('$QUERIES_JSON', encoding='utf-8') as f:
    data = json.load(f)
for q in data.get('queries', []):
    print(f\"  {q.get('id','?'):30s} {q.get('framework','?'):8s} {q.get('query','')[:80]}\")
"
}

cmd_run() {
    local query_id_arg="$1"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOGS_DIR"
    local log_file="$LOGS_DIR/run_${ts}.log"

    # Construcción de args al runner Python.
    local py_args=(
        "$RUNNER_PY"
        --config "$CONFIG_YAML"
        --queries "$QUERIES_JSON"
    )
    if [[ "$query_id_arg" != "all" ]]; then
        py_args+=(--query-id "$query_id_arg")
    fi

    log "Lanzando runner Python..."
    log "  Log completo: $log_file"
    local start_epoch
    start_epoch="$(date +%s)"

    # tee para que se vea en consola y a disco al mismo tiempo.
    # `|| true` para no abortar el script antes de imprimir el resumen:
    # el runner Python maneja sus propios errores y devuelve 0 en flujo normal.
    set +e
    (cd "$REPO_ROOT" && "$VENV_PY" "${py_args[@]}") 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    set -e

    local end_epoch
    end_epoch="$(date +%s)"
    local elapsed=$(( end_epoch - start_epoch ))

    echo ""
    log "Resumen final:"
    log "  Subcomando:    $query_id_arg"
    log "  Duración:      ${elapsed}s"
    log "  Exit code:     $rc"
    log "  Log:           $log_file"
    log "  Outputs:       $REPO_ROOT/outputs/batch_rudiger_*/"

    if [[ $rc -ne 0 ]]; then
        warn "El runner devolvió código $rc. Revisa el log arriba."
    fi
    exit "$rc"
}

# ── Main ──────────────────────────────────────────────────────────────
CMD="${1:-help}"

case "$CMD" in
    help|--help|-h)
        show_help
        ;;
    list)
        validate_files
        print_banner
        cmd_list
        ;;
    all)
        print_banner
        validate_files
        validate_config_filled
        cmd_run "all"
        ;;
    *)
        # Cualquier otro argumento se trata como query_id.
        # Validamos existencia en queries.json antes de lanzar.
        print_banner
        validate_files
        validate_config_filled
        if ! "$VENV_PY" -c "
import json, sys
with open('$QUERIES_JSON', encoding='utf-8') as f:
    data = json.load(f)
ids = [q.get('id') for q in data.get('queries', [])]
sys.exit(0 if '$CMD' in ids else 1)
"; then
            err "query_id '$CMD' no existe en queries.json."
            err "Usa '$0 list' para ver las IDs disponibles."
            exit 2
        fi
        cmd_run "$CMD"
        ;;
esac
