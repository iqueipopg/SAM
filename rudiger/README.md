# SAM en Servidor Rudiger — Guía de Instalación y Uso

Este documento describe cómo poner en marcha **SAM** (Systematic Agent for Meta-analysis) en el servidor Rudiger del laboratorio, utilizando la configuración especializada `config_rudiger.yaml`.

## ¿Qué es SAM?

SAM es una pipeline de 9 agentes IA que automatiza el ciclo completo de una revisión sistemática:

1. **A1** – Protocolo (framework PICO/SPIDER + criteria + query PubMed)
2. **A2** – Búsqueda multi-base de datos (PubMed, Europe PMC, Semantic Scholar, Google Scholar)
3. **A3** – Screening título/abstract (ensemble voting)
4. **A4** – Recuperación full-text + screening (cascada 10 fuentes)
5. **A5** – Extracción de datos (3 prompts focalizados)
6. **A6** – Evaluación de calidad (RoB 2.0 / ROBINS-I / JBI)
7. **A7** – Síntesis estadística (R/metafor: REML, I², tau², Egger, Fail-Safe N, subgrupos, meta-regresión, GRADE)
8. **A8** – Generación de manuscrito (RAG + LaTeX PDF)
9. **A9** – Explainability (xlsx + informe markdown)

En Rudiger, tendrás VRAM de sobra (a diferencia de la laptop RTX 4060 del desarrollo). Puedes usar modelos más grandes sin cuantizar.

---

## Prerequisitos

- **Python 3.10+** (verificar: `python --version`)
- **Entorno virtual venv** (se crea con `python -m venv`)
- **R con metafor** (`Rscript -e "install.packages('metafor')"`)
- **GPU con CUDA** (NVIDIA, verificar: `nvidia-smi`)
- **pdflatex** (para generación de PDFs: `apt-get install texlive-latex-base`)
- **Espacio en disco** (~5-10 GB recomendado para outputs)
- **Acceso a .env** con tokens de APIs (ENTREZ_EMAIL mínimo, opcional: UNPAYWALL_EMAIL, GROQ_API_KEY, etc.)

---

## Contenido de `rudiger/`

| Archivo | Descripción |
|---------|-------------|
| `config_rudiger.yaml` | Configuración self-contained para Rudiger (hereda de `config.yaml` raíz con overrides). Contiene placeholders `<RELLENAR_EDUARDO>` en rutas de modelos y R library. |
| `queries.json` | 3 queries predefinidas: `paula_nutrition`, `paula_sport`, `cbt_depression_baseline`. Cada una con framework, max_results, y notas de validación. |
| `run_rudiger.sh` | Script bash ejecutable. Subcomandos: `help`, `list`, `all`, `<query_id>`. Valida que no haya placeholders sin rellenar (exit 4). |
| `run_queries.py` | Wrapper Python que lee `queries.json` y llama `batch_run.run_single_pipeline()` por cada query. Maneja config override vía `--config`. |

---

## Paso 1: Rellenar Placeholders en `config_rudiger.yaml`

Antes de lanzar cualquier pipeline, **OBLIGATORIO** rellenar estas 8 líneas en `config_rudiger.yaml`:

### Modelos LLM (7 entradas)

Abre el archivo y busca la sección `models:`. Reemplaza cada `<RELLENAR_EDUARDO>` con la ruta local (absoluta o HuggingFace hub id):

```yaml
models:
  agent1_protocol:   "/opt/models/qwen2.5-7b"        # o "Qwen/Qwen2.5-7B-Instruct"
  agent2_search:     "/opt/models/qwen2.5-7b"
  agent3_screening:  "/opt/models/qwen2.5-7b"
  agent4_validation: "/opt/models/qwen2.5-7b"
  agent5_extraction: "/opt/models/qwen2.5-7b"
  agent6_quality:    "/opt/models/qwen2.5-7b"
  agent7_synthesis:  "R/metafor"  # ← este NO se cambia
  agent8_manuscript: "/opt/models/llama-3.3-70b"     # o el modelo que prefieras
```

**Opciones recomendadas para Rudiger:**

- **Modelo pequeño (rápido, ≤7B):** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct
- **Modelo grande (mejor calidad, 70B):** Llama-3.3-70B, Mistral-Large, Qwen2.5-72B (requiere `quantization: none`)

### Cuantización (ya configurado, pero revisar)

En la sección `generation:`, el valor está pre-configurado:

```yaml
quantization: "none"  # Opciones: "4bit" | "8bit" | "none"
                      # Rudiger puede usar "none" (sin cuantizar)
                      # para modelos 70B si tiene 40+ GB VRAM
```

### Ruta de R Library (1 entrada)

Busca la sección `synthesis:` y rellena:

```yaml
synthesis:
  r_library_path: "~/R/library"  # o "/opt/R/library" o donde esté metafor
```

Verifica que metafor está instalado:

```bash
Rscript -e "library(metafor); packageVersion('metafor')"
```

---

## Paso 2: Elegir Perfil de Tokens (Opcional)

En `config/agents.json` (raíz del repo, NO en rudiger/), el agente A8 (manuscrito) tiene dos perfiles de tamaño de sección:

- **`compact`** (default): secciones pequeñas (400-1200 tokens), rápido, output conciso
- **`extended`** (nuevo): secciones 3-4× más grandes (1200-4800 tokens), mejor para modelos poderosos

Para activar `extended`, edita `config/agents.json`:

```json
"agent8_manuscript": {
  "section_max_tokens_profile": "extended",  // cambiar de "compact"
  // ... resto de parámetros
}
```

**Cuándo usar:**

- `compact`: modelo ≤7B, time constraints, GPU con VRAM limitada
- `extended`: modelo 70B, tiempo disponible, VRAM >40 GB

---

## Paso 3 (Opcional): Ensemble Multi-Modelo

Si quieres que A3 (screening) o A4 (full-text) usen múltiples modelos en secuencia (mejor consenso, mejor calidad), puedes descomentar la sección `ensemble_models` en `config_rudiger.yaml`:

```yaml
agent3_screening:
  ensemble_models:
    - path: "/opt/models/qwen2.5-7b"
      quantization: "4bit"
      votes: 1
    - path: "/opt/models/llama-3.3-70b"
      quantization: "none"
      votes: 2

agent4_validation:
  ensemble_models:
    - path: "/opt/models/qwen2.5-7b"
      quantization: "4bit"
      votes: 1
    - path: "/opt/models/llama-3.3-70b"
      quantization: "none"
      votes: 2
```

**Cómo funciona:**

- A3/A4 cargan el primer modelo, emiten `votes` votos, liberan VRAM (`free_vram()`)
- Cargan el segundo modelo, emiten `votes` votos, liberan VRAM
- Y así sucesivamente
- Los votos se agregan: si `voting_threshold: 0.60` y 3 votos totales, necesitas ≥2 INCLUDE
- Solo UN modelo en VRAM a la vez

**Impacto:** tiempo ~2-3× más lento (depende de #modelos), mejor precisión.

**Regla importante:** suma de `votes` ≥ `ensemble_size` (que por defecto es 3).

---

## Paso 4: Lanzar la Pipeline

### Verificar Queries Disponibles

```bash
cd ~/meta-analisis-automatizado  # repo raíz
./rudiger/run_rudiger.sh list
```

Output:

```
  paula_nutrition          PICO   In community-dwelling adults of retirement...
  paula_sport              PICO   In community-dwelling adults around retirement...
  cbt_depression_baseline  PICO   Is cognitive behavioral therapy more effective...
```

### Ejecutar Una Query Concreta

```bash
./rudiger/run_rudiger.sh paula_nutrition
```

- Carga `config_rudiger.yaml`
- Lanza `batch_run.py` internamente con esa config
- Genera checkpoints bajo `rudiger/outputs/`
- Escribe log en `rudiger/logs/run_<timestamp>.log`

### Ejecutar Todas las Queries

```bash
./rudiger/run_rudiger.sh all
```

Itera paula_nutrition → paula_sport → cbt_depression_baseline. Duración estimada: 4-8 horas (depende de modelos y VRAM).

### Ver Ayuda

```bash
./rudiger/run_rudiger.sh help
```

---

## Paso 5: Outputs y Logs

**Estructura de directorios:**

```
rudiger/
  outputs/
    run_paula_nutrition/
      checkpoint_a1_protocol.json
      checkpoint_a2_search.json
      ...
      checkpoint_a8_manuscript.json
      checkpoint_a9_explainer.json
      pipeline.json                  # resultado final
      manuscript.pdf
      explainability.md              # informe en español
  logs/
    run_20260421_142530.log          # log completo
```

**Archivos importantes en pipeline final:**

- `pipeline.json` – todo: protocolo, búsqueda, screening, full-text, extracción, calidad, síntesis, manuscrito
- `manuscript.pdf` – PDF listo para publicación/revisión
- `explainability.md` – trazabilidad: qué papers incluyo/excluyo, por qué, decisiones LLM
- `papers_raw.xlsx` – tabla per-paper (title, authors, year, extraction data, quality score)

---

## Paso 5b (Alternativo): API Server

Si prefieres lanzar la UI web o un servidor REST:

```bash
# Terminal 1: FastAPI backend con SSE streaming
export SAM_CONFIG=rudiger/config_rudiger.yaml
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2: React web UI (requiere Node.js)
cd web
npm install
npm run dev
```

Abre `http://localhost:5173` (web UI) o `http://localhost:8000/docs` (FastAPI swagger).

---

## Troubleshooting

### Error: "config_rudiger.yaml contiene placeholders `<RELLENAR_EDUARDO>`"

**Causa:** No rellenaste los modelos o r_library_path.

**Solución:** Abre `rudiger/config_rudiger.yaml` y reemplaza todos `<RELLENAR_EDUARDO>` con rutas válidas. Luego reintenta.

### Error: "No module named 'R'" o "metafor not found"

**Causa:** R no está instalado o metafor no está accesible.

**Solución:**

```bash
# Instalar R
apt-get install r-base

# Instalar metafor
Rscript -e "install.packages('metafor', lib='~/R/library')"

# Verificar
Rscript -e "library(metafor); print('OK')"
```

### Error: "CUDA out of memory"

**Causa:** VRAM insuficiente para el modelo elegido.

**Soluciones:**

1. Cambiar modelo a uno más pequeño: agent8_manuscript → Qwen2.5-7B en lugar de 70B
2. Activar cuantización: `quantization: "4bit"` en lugar de `"none"`
3. Reducir `batch_size` (en config_rudiger.yaml): `batch_size: 1` → funciona pero lento
4. Desactivar ensemble multi-modelo si lo usabas

### Error: Google Scholar CAPTCHA / Rate limit 429

**Causa:** Demasiadas consultas a Google Scholar muy rápido.

**Solución:** En `config_rudiger.yaml`, aumenta el delay:

```yaml
search:
  google_scholar_rate_limit_seconds: 5.0  # aumentar de 3.0
```

### Error: PubMed rate limit (Entrez)

**Causa:** ENTREZ_EMAIL no configurado en `.env` o rate limit de 0.4s insuficiente.

**Solución:**

```bash
# 1. Verificar .env
cat .env | grep ENTREZ_EMAIL

# 2. Si falta, añadir:
echo "ENTREZ_EMAIL=tu_email@example.com" >> .env

# 3. El delay ya está bien configurado en config_rudiger.yaml (0.4s)
```

---

## Queries Incluidas

### `paula_nutrition`

**Research question:** En adultos comunitarios edad jubilación (media 55-65), ¿intervenciones con componentes psicológicos (entrevista motivacional, TCC, autogestión, goal-setting, modificación comportamental) que promuevan cambio dietético son más efectivas que comparador para mejorar ingesta dietética (verduras, frutas, calorías, grasas, calidad) y resultados antropométricos (peso, BMI, perímetro cintura, grasa visceral)?

**Gold standard:** 90 PDFs incluidos (meta-análisis Paula Collazo, SAMA). Población: jubilación, media 55-65.

**Limitaciones:** Excluir outcomes glicémicos/lipídicos (HbA1c, medidas biológicas).

### `paula_sport`

**Research question:** En adultos comunitarios edad jubilación (media 55-65), ¿intervenciones de actividad física con componentes psicológicos intencionales (técnicas cambio comportamental, action planning, teoría autodeterminación) son más efectivas que comparador para mejorar comportamiento actividad física medida directa (fuerza, caminata, reducción sedentarismo)?

**Gold standard:** 1552 estudios screening + MA publicada Paula (data/gold_standard/deporte/, 2026-04-21). Solo RCTs individuales (excluir cluster-RCTs). Excluir pacientes con enfermedades serias.

**Limitaciones:** Solo outcomes conductuales medidos directamente (no intenciones). Componente psicológico intencional evaluado por expertos.

### `cbt_depression_baseline`

**Research question:** ¿TCC es más efectiva que medicación para depresión en adultos?

**Propósito:** Baseline de referencia. Reproduce `outputs/run_20260311_155651/` (10 included).

**Uso:** Smoke-test de regresión entre versiones/hardware. Duración: ~30 min.

---

## Contacto y Autor

**SAM** — Trabajo de Fin de Grado, Universidad Pontificia Comillas (ICAI), iMAT.

GitHub: https://github.com/iqueipopg/SAM

Autor: iqueipopg

Última actualización: 2026-04-21

---

## Siguientes Pasos (post-instalación)

1. **Hacer un smoke-test:** ejecuta `./rudiger/run_rudiger.sh cbt_depression_baseline` (rápido, ~30 min)
2. **Revisar outputs:** abre `rudiger/outputs/run_cbt_depression_baseline/manuscript.pdf`
3. **Comparar vs baseline:** verifica que el resultado coincida con `outputs/run_20260311_155651/`
4. **Ejecutar paula_sport:** más grande (1552 screening), gold standard disponible para validación
5. **Ejecutar paula_nutrition:** gold standard parcial, útil para tunear A5 extraction numeric recovery

---

## Notas Técnicas

- **Venv:** Siempre usar `~/meta-analisis-automatizado/venv/bin/python` (no `source activate`)
- **Config:** `--config` flag soportado en `batch_run.py` y `api.py` (T16.9)
- **Checkpointing:** Activado por defecto. Si la pipeline falla en A5, puedes reanudar desde A6.
- **Groq cloud:** Si quieres comparar con Llama-70B en la nube, cambia `groq.provider: "local"` → `"groq"` y configura `GROQ_API_KEY` en `.env`
- **Idioma:** Outputs en español (informes, explainability, logs). Código en inglés.
