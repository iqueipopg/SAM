#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# SAM — Systematic Agent for Meta-analysis
# Script de instalación y ejecución para servidores Ubuntu (Rudiger)
#
# Uso:
#   chmod +x run.sh
#   ./run.sh install        # Instala todo (primera vez)
#   ./run.sh pipeline       # Ejecuta pipeline batch (5 queries)
#   ./run.sh single "pregunta clínica aquí"  # Una sola query
#   ./run.sh ui             # Lanza interfaz Streamlit
#   ./run.sh test           # Ejecuta tests
# ══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuración ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
R_LIB="$HOME/R/library"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[SAM]${NC} $1"; }
warn() { echo -e "${YELLOW}[SAM]${NC} $1"; }
err()  { echo -e "${RED}[SAM]${NC} $1" >&2; }

# ── Funciones de instalación ──────────────────────────────────────

install_system_deps() {
    log "Instalando dependencias del sistema..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 python3-venv python3-pip python3-dev \
        r-base r-base-dev \
        texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
        texlive-fonts-extra texlive-latex-recommended \
        libxml2-dev libcurl4-openssl-dev libssl-dev \
        libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
        libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
        git curl wget \
        2>/dev/null
    log "Dependencias del sistema instaladas."
}

install_python_env() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Creando entorno virtual Python..."
        python3 -m venv "$VENV_DIR"
    fi

    log "Instalando paquetes Python..."
    "$PIP" install --upgrade pip wheel setuptools -q
    "$PIP" install -r requirements.txt -q
    log "Paquetes Python instalados."
}

install_r_packages() {
    log "Instalando paquetes R (metafor)..."
    mkdir -p "$R_LIB"
    Rscript -e "
        lib <- Sys.getenv('HOME')
        lib <- file.path(lib, 'R', 'library')
        dir.create(lib, recursive=TRUE, showWarnings=FALSE)
        .libPaths(c(lib, .libPaths()))
        if (!require('metafor', quietly=TRUE)) {
            install.packages('metafor', repos='https://cloud.r-project.org', lib=lib)
        }
        cat('metafor version:', as.character(packageVersion('metafor')), '\n')
    "
    log "Paquetes R instalados."
}

setup_env_file() {
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        log "Creando archivo .env desde plantilla..."
        if [ -f "$SCRIPT_DIR/.env.example" ]; then
            cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        else
            cat > "$SCRIPT_DIR/.env" << 'ENVEOF'
# PubMed / NCBI (obligatorio)
ENTREZ_EMAIL=meta-analysis@example.com

# Unpaywall (recomendado — email para acceso full-text)
UNPAYWALL_EMAIL=meta-analysis@example.com

# Semantic Scholar (opcional — mayor rate limit con API key)
# SEMANTIC_SCHOLAR_API_KEY=

# Lens.org (opcional — requiere registro gratuito)
# LENS_API_TOKEN=

# CORE (opcional — requiere registro gratuito en core.ac.uk)
# CORE_API_KEY=
ENVEOF
        fi
        warn "Edita .env con tus credenciales: nano $SCRIPT_DIR/.env"
    fi
}

download_model() {
    # Descarga el modelo si no existe localmente
    # En Rudiger puede que ya esté descargado o queramos uno más grande
    local MODEL_DIR="$SCRIPT_DIR/models/qwen2.5-7b"
    if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
        log "Modelo Qwen2.5-7B ya existe en $MODEL_DIR"
        return 0
    fi

    warn "El modelo Qwen2.5-7B no está descargado."
    warn "Para descargarlo ejecuta:"
    echo "  $PYTHON -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='$MODEL_DIR'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='$MODEL_DIR')\""
    warn ""
    warn "Para usar un modelo más grande en Rudiger, edita config.yaml:"
    warn "  models.agent1_protocol: './models/qwen3-30b'  (o el que prefieras)"
}

check_gpu() {
    log "Comprobando GPU..."
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        "$PYTHON" -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || true
    else
        warn "nvidia-smi no encontrado. ¿Están instalados los drivers CUDA?"
    fi
}

# ── Funciones de ejecución ────────────────────────────────────────

run_pipeline_batch() {
    log "Ejecutando pipeline batch (5 queries)..."
    "$PYTHON" batch_run.py
}

run_single_query() {
    local QUERY="$1"
    log "Ejecutando pipeline para: $QUERY"

    "$PYTHON" -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from dotenv import load_dotenv
load_dotenv()

import time
from datetime import datetime
from batch_run import run_single_pipeline, analyze_run, clean_r_objects, log as blog

query = '''$QUERY'''
ts = time.strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join('outputs', f'run_{ts}')
os.makedirs(run_dir, exist_ok=True)

print(f'[SAM] Query: {query}')
print(f'[SAM] Output: {run_dir}')

run_log = []
start = time.time()
result = run_single_pipeline(query, run_dir, run_log)
elapsed = round(time.time() - start, 1)

# Analyze and print report
report = analyze_run(run_dir, result)
report['elapsed_s'] = elapsed

print(f'\n--- SUMMARY ---')
print(f'  Framework: {report[\"framework\"]}')
print(f'  Papers found: {report.get(\"papers_found\", 0)}')
print(f'  A3 screening: {report.get(\"a3_included\", 0)} included / {report.get(\"a3_total\", 0)} total')
print(f'  Final included: {report.get(\"final_included\", 0)}')
print(f'  Analyses: {report.get(\"n_analyses\", 0)}')
for detail in report.get('synthesis_details', []):
    print(f'    {detail[\"measure\"]}: pooled={detail[\"pooled\"]} [{detail[\"ci_lower\"]}, {detail[\"ci_upper\"]}], k={detail[\"k_studies\"]}, GRADE={detail[\"grade\"]}')
print(f'  Verdict: {report[\"verdict\"]}')
if report.get('issues'):
    for issue in report['issues']:
        print(f'  ! {issue}')
print(f'  Time: {elapsed}s')
print(f'  Results: {run_dir}/')
"
}

run_ui() {
    log "Lanzando interfaz Streamlit..."
    "${VENV_DIR}/bin/streamlit" run app.py --server.headless true --server.port 8501
}

run_tests() {
    log "Ejecutando tests..."
    "${VENV_DIR}/bin/pytest" tests/ -v --tb=short 2>&1 || true
}

# ── Cambiar modelo (para Rudiger con más VRAM) ───────────────────

configure_model() {
    local MODEL_PATH="${1:-}"
    if [ -z "$MODEL_PATH" ]; then
        echo "Uso: ./run.sh model <ruta-al-modelo>"
        echo ""
        echo "Ejemplos:"
        echo "  ./run.sh model ./models/qwen3-30b"
        echo "  ./run.sh model Qwen/Qwen2.5-32B-Instruct"
        echo ""
        echo "Esto actualiza config.yaml para que todos los agentes usen el modelo indicado."
        return 1
    fi

    log "Configurando modelo: $MODEL_PATH"
    "$PYTHON" -c "
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
for key in cfg.get('models', {}):
    if key != 'agent7_synthesis':
        cfg['models'][key] = '$MODEL_PATH'
# Si el modelo es grande, podemos desactivar cuantización
cfg['quantization'] = 'none'
with open('config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print('config.yaml actualizado.')
for k, v in cfg['models'].items():
    print(f'  {k}: {v}')
print(f'  quantization: {cfg[\"quantization\"]}')
"
    log "Modelo configurado. Ejecuta './run.sh pipeline' para probar."
}

# ── Main ──────────────────────────────────────────────────────────

show_help() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════"
    echo "  SAM — Systematic Agent for Meta-analysis"
    echo "═══════════════════════════════════════════════════════════"
    echo -e "${NC}"
    echo "Uso: ./run.sh <comando> [argumentos]"
    echo ""
    echo "Comandos:"
    echo "  install          Instala todo (sistema + Python + R + .env)"
    echo "  pipeline         Ejecuta batch de 5 queries predefinidas"
    echo "  single \"query\"   Ejecuta pipeline para una pregunta clínica"
    echo "  ui               Lanza interfaz web Streamlit"
    echo "  test             Ejecuta suite de tests"
    echo "  nutrition        Test local con PDFs de nutrición (A5→A8)"
    echo "  nutrition-a5     Solo extracción A5 (rápido, para diagnóstico)"
    echo "  model <ruta>     Cambia el modelo LLM en config.yaml"
    echo "  gpu              Muestra info de GPU disponible"
    echo "  help             Muestra esta ayuda"
    echo ""
    echo "Ejemplo rápido:"
    echo "  ./run.sh install"
    echo "  ./run.sh single \"Is exercise effective for depression in elderly?\""
    echo ""
    echo "Para usar un modelo más grande en Rudiger:"
    echo "  ./run.sh model ./models/qwen3-30b"
    echo "  ./run.sh pipeline"
}

CMD="${1:-help}"

case "$CMD" in
    install)
        install_system_deps
        install_python_env
        install_r_packages
        setup_env_file
        download_model
        check_gpu
        log "Instalación completa. Ejecuta './run.sh pipeline' para probar."
        ;;
    pipeline)
        run_pipeline_batch
        ;;
    single)
        if [ -z "${2:-}" ]; then
            err "Falta la query. Uso: ./run.sh single \"pregunta clínica\""
            exit 1
        fi
        run_single_query "$2"
        ;;
    ui)
        run_ui
        ;;
    test)
        run_tests
        ;;
    nutrition)
        log "Ejecutando test local de nutrición (A5→A8)..."
        "$PYTHON" tests/test_local_nutrition.py ${2:+--max-papers "$2"}
        ;;
    nutrition-a5)
        log "Ejecutando extracción A5 sobre PDFs de nutrición..."
        "$PYTHON" tests/test_local_nutrition.py --a5-only ${2:+--max-papers "$2"}
        ;;
    model)
        configure_model "${2:-}"
        ;;
    gpu)
        check_gpu
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        err "Comando desconocido: $CMD"
        show_help
        exit 1
        ;;
esac
