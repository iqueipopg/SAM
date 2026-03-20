#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# SAM — Test de Meta-Análisis de Nutrición (Papers de Paula)
#
# Para ejecutar en Rudiger con modelo grande:
#
#   chmod +x run_nutrition_test.sh
#   ./run_nutrition_test.sh                     # modelo por defecto (config.yaml)
#   ./run_nutrition_test.sh --model Qwen/Qwen2.5-32B-Instruct
#   ./run_nutrition_test.sh --max-papers 5      # test rápido con 5 papers
#   ./run_nutrition_test.sh --resume <dir>      # retomar desde checkpoint
#
# Requisitos:
#   - PDFs en docs/meta-nutricion-paula/
#   - pip install pdfplumber PyMuPDF python-docx (ya en requirements.txt)
#   - GPU con suficiente VRAM (7B → 8GB, 32B → 24GB+)
#
# Eduardo: si usas un modelo grande, primero:
#   ./run.sh model ./models/tu-modelo
# o pasa --model aquí directamente.
# ══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
CONFIG="$SCRIPT_DIR/config.yaml"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[SAM-NUTRITION]${NC} $1"; }
warn() { echo -e "${YELLOW}[SAM-NUTRITION]${NC} $1"; }
err()  { echo -e "${RED}[SAM-NUTRITION]${NC} $1" >&2; }

# ── Parse args ──────────────────────────────────────────────────
MODEL_PATH=""
MAX_PAPERS=0
RESUME_DIR=""
A5_ONLY=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"; shift 2 ;;
        --max-papers)
            MAX_PAPERS="$2"; shift 2 ;;
        --resume)
            RESUME_DIR="$2"; shift 2 ;;
        --a5-only)
            A5_ONLY=true; shift ;;
        *)
            err "Argumento desconocido: $1"
            echo "Uso: $0 [--model <path>] [--max-papers N] [--resume <dir>] [--a5-only]"
            exit 1 ;;
    esac
done

# ── Verificar entorno ───────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    err "No se encontró el venv. Ejecuta primero: ./run.sh install"
    exit 1
fi

# ── Verificar PDFs ──────────────────────────────────────────────
PDF_DIR="$SCRIPT_DIR/docs/meta-nutricion-paula"
if [ ! -d "$PDF_DIR" ]; then
    err "No se encontró el directorio de PDFs: $PDF_DIR"
    err "Copia los PDFs de Paula ahí primero."
    exit 1
fi

N_PDFS=$(find "$PDF_DIR" -name "*.pdf" -type f | wc -l)
log "PDFs encontrados: ${BOLD}$N_PDFS${NC}"

# ── GPU check ───────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU")
    log "GPU: ${CYAN}$GPU_INFO${NC}"
else
    warn "nvidia-smi no encontrado — se usará CPU (muy lento)"
fi

# ── Configurar modelo (si se pasó --model) ──────────────────────
ORIGINAL_CONFIG=""
if [ -n "$MODEL_PATH" ]; then
    log "Configurando modelo: ${BOLD}$MODEL_PATH${NC}"

    # Guardar config original
    ORIGINAL_CONFIG=$(cat "$CONFIG")

    # Actualizar config.yaml con el modelo
    "$PYTHON" -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
for key in cfg.get('models', {}):
    if key != 'agent7_synthesis':
        cfg['models'][key] = '$MODEL_PATH'
# Si el modelo es grande (>14B), quitar quantization
model_lower = '$MODEL_PATH'.lower()
if any(s in model_lower for s in ['32b', '30b', '72b', '70b', '14b']):
    cfg['quantization'] = '4bit'  # aún 4bit para modelos grandes
    print(f'[SAM] Modelo grande detectado, manteniendo 4bit quantization')
with open('$CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f'[SAM] Config actualizada: todos los agentes usan $MODEL_PATH')
"
fi

# ── Construir comando ───────────────────────────────────────────
CMD="$PYTHON tests/test_local_nutrition.py"

if [ "$MAX_PAPERS" -gt 0 ]; then
    CMD="$CMD --max-papers $MAX_PAPERS"
fi

if [ -n "$RESUME_DIR" ]; then
    CMD="$CMD --resume $RESUME_DIR"
fi

if [ "$A5_ONLY" = true ]; then
    CMD="$CMD --a5-only"
fi

# ── Ejecutar ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  SAM — Test de Nutrición (Papers de Paula)${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "  PDFs:        $N_PDFS"
echo -e "  Max papers:  $([ "$MAX_PAPERS" -gt 0 ] && echo "$MAX_PAPERS" || echo "todos")"
echo -e "  Modelo:      $([ -n "$MODEL_PATH" ] && echo "$MODEL_PATH" || echo "(config.yaml default)")"
echo -e "  Dual extr.:  $(grep 'dual_extraction' "$CONFIG" | head -1 | awk '{print $2}')"
echo -e "  Comando:     $CMD"
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""

START_TIME=$(date +%s)

# Ejecutar con output unbuffered para ver progreso en tiempo real
PYTHONUNBUFFERED=1 $CMD
EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}  TEST COMPLETADO EN ${MINS}m ${SECS}s${NC}"
else
    echo -e "${RED}  TEST FALLIDO (exit code $EXIT_CODE) — ${MINS}m ${SECS}s${NC}"
fi
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"

# ── Mostrar resultados ──────────────────────────────────────────
LATEST_RUN=$(ls -td outputs/nutrition_test_* 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
    echo ""
    log "Output: ${BOLD}$LATEST_RUN/${NC}"
    echo ""

    # Mostrar resumen de A5
    if [ -f "$LATEST_RUN/checkpoint_a5_extraction.json" ]; then
        "$PYTHON" -c "
import json
with open('$LATEST_RUN/checkpoint_a5_extraction.json') as f:
    d = json.load(f)
s = d.get('stats', {})
print(f'  A5 Extracción:')
print(f'    Complete:  {s.get(\"complete_extraction\", 0)}')
print(f'    Partial:   {s.get(\"partial_extraction\", 0)}')
print(f'    Failed:    {s.get(\"failed_extraction\", 0)}')
print(f'    Numérico:  {s.get(\"numeric_complete\", 0)}')
print(f'    Recovered: {s.get(\"numeric_recovered\", 0)}')
print(f'    Dual:      {s.get(\"dual_extraction\", False)}')
print(f'    Tiempo:    {s.get(\"elapsed_s\", 0)}s')
" 2>/dev/null || true
    fi

    # Mostrar resumen de A7
    if [ -f "$LATEST_RUN/checkpoint_a7_synthesis.json" ]; then
        "$PYTHON" -c "
import json
with open('$LATEST_RUN/checkpoint_a7_synthesis.json') as f:
    d = json.load(f)
s = d.get('stats', {})
print(f'  A7 Síntesis:')
print(f'    Papers con datos: {s.get(\"papers_with_usable_data\", 0)}')
print(f'    Análisis:         {s.get(\"analyses_performed\", 0)}')
for m, a in d.get('analyses', {}).items():
    model = a.get('model', {})
    grade = a.get('grade', {})
    print(f'    {m}: pooled={model.get(\"pooled_estimate\", \"?\")} '
          f'[{model.get(\"ci_lower\", \"?\")}, {model.get(\"ci_upper\", \"?\")}] '
          f'I²={model.get(\"I2\", \"?\")}% GRADE={grade.get(\"final_level\", \"?\")}')
" 2>/dev/null || true
    fi

    # Mostrar resumen de A8
    if [ -f "$LATEST_RUN/checkpoint_a8_manuscript.json" ]; then
        "$PYTHON" -c "
import json
with open('$LATEST_RUN/checkpoint_a8_manuscript.json') as f:
    d = json.load(f)
s = d.get('stats', {})
coh = d.get('coherence_check', {})
cv = d.get('citation_verification', {})
print(f'  A8 Manuscrito:')
print(f'    Papers citados:  {s.get(\"total_papers_cited\", 0)}')
print(f'    RAG chunks:      {s.get(\"total_chunks\", 0)}')
print(f'    PDF:             {d.get(\"pdf_path\", \"N/A\")}')
print(f'    Coherencia:      {\"OK\" if coh.get(\"passed\", True) else str(coh.get(\"n_warnings\", 0)) + \" warnings\"}')
print(f'    Citas:           {cv.get(\"total_verified\", 0)}/{cv.get(\"total_citations\", 0)} verificadas')
" 2>/dev/null || true
    fi

    echo ""
    log "Archivos generados:"
    ls -lh "$LATEST_RUN/" 2>/dev/null | grep -v "^total\|^d" | awk '{print "    " $NF " (" $5 ")"}'

    # Listar plots
    if [ -d "$LATEST_RUN/plots" ] && [ "$(ls -A "$LATEST_RUN/plots" 2>/dev/null)" ]; then
        echo ""
        log "Plots:"
        ls -lh "$LATEST_RUN/plots/" 2>/dev/null | grep -v "^total\|^d" | awk '{print "    " $NF " (" $5 ")"}'
    fi
fi

# ── Restaurar config original si fue modificada ─────────────────
if [ -n "$ORIGINAL_CONFIG" ]; then
    echo "$ORIGINAL_CONFIG" > "$CONFIG"
    log "Config restaurada al estado original."
fi

exit $EXIT_CODE
