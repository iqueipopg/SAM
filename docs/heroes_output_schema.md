# Heroes Output Schema — Interfaz Uniforme de Evaluadores A3-A6

**Última actualización:** 2026-05-15  
**Sprint:** T16.D.5 (Evaluación de 4 agentes contra gold standard de Paula)  
**Estado:** Especificación canónica para evaluadores hero

---

## 1. Propósito

Este documento define el **schema JSON uniforme** que los 4 evaluadores "hero" deben emitir tras correr sobre muestras del gold standard de Paula (deporte, 1552 estudios). Los heroes son:

- **Hero 1 (A3):** Agent3 Title/Abstract Screening
- **Hero 2 (A4):** Agent4 Full-Text Screening  
- **Hero 3 (A5):** Agent5 Data Extraction
- **Hero 4 (A6):** Agent6 Quality Assessment (RoB)

La interfaz común permite:
- ✓ Comparabilidad directa entre agentes
- ✓ Agregación en `run_heroes_micro.py`
- ✓ Procesamiento uniforme en test suite
- ✓ Debugging y auditoría de predicciones
- ✓ Exportación a reportes de evaluación

---

## 2. Schema JSON Canónico

```json
{
  "hero_id": "hero1" | "hero2" | "hero3" | "hero4",
  "agent": "A3" | "A4" | "A5" | "A6",
  "sample_path": "data/gold_standard/deporte/samples/micro/sample_50.csv",
  "sample_size": 50,
  "model_used": "Qwen2.5-7B-Instruct (4-bit BitsAndBytes)",
  "llm_config": {
    "temperature": 0,
    "top_p": 1.0,
    "max_tokens": 150,
    "repetition_penalty": 1.3,
    "seed": 42
  },
  "timestamp_utc": "2026-05-15T14:23:01Z",
  "elapsed_seconds": 187.4,
  "environment": {
    "python_version": "3.12.x",
    "ollama_version": null,
    "gpu_info": "RTX 4060 8GB"
  },
  "metrics": {
    "...": "ver Sección 3 (hero-específico)"
  },
  "confusion_matrix": {
    "...": "tabla 2x2 (hero1/2), 3x3 (hero4), null (hero3)"
  },
  "raw_predictions": [
    {
      "input_id": "string (ej: 'PMC123456' o 'study_001')",
      "gold": "valor gold standard (tipo depende de hero)",
      "predicted": "valor predicho por el agente",
      "match": true,
      "details": {
        "raw_llm_response": "respuesta bruta del LLM (sin procesar)",
        "field_scores": {}
      }
    }
  ],
  "errors": [
    {
      "input_id": "string",
      "error_type": "ParseError" | "TimeoutError" | "ModelError" | "ValidationError",
      "message": "descripción del error"
    }
  ],
  "notes": "observaciones cualitativas, limitations, detalles del run"
}
```

### Campos obligatorios (todos):
- `hero_id`, `agent`, `sample_path`, `sample_size`
- `model_used`, `llm_config`, `timestamp_utc`, `elapsed_seconds`, `environment`
- `metrics`, `confusion_matrix`, `raw_predictions`, `errors`

### Reglas generales:
- **Timestamps:** ISO 8601 UTC (`"2026-05-15T14:23:01Z"`)
- **Duración:** en segundos, float, incluyendo setup + inference + I/O
- **GPU info:** ej. `"RTX 4060 8GB"`, `"A100 40GB"`, o `null` si CPU-only
- **raw_llm_response:** debe capturarse SIEMPRE (sin truncar) para debugging. Usar `\n` para saltos de línea.
- **Seed:** fijo a 42 para reproducibilidad de evaluadores

---

## 3. Tabla de Métricas por Hero

### 3.1 Hero 1 — A3 Title/Abstract Screening

**Entrada:** abstract + título; **Salida:** Include/Exclude (EXCLUDE por defecto si parse falla)

**Métricas obligatorias:**

```json
"metrics": {
  "sensitivity_overall": 0.92,
  "sensitivity_abstract_yes": 0.88,
  "sensitivity_abstract_no": 0.95,
  "tp": 46,
  "fn": 4,
  "tn": null,
  "fp": null,
  "false_negatives": ["PMC123", "PMC124", "PMC125", "PMC126"]
}
```

**Definiciones:**
- `sensitivity_overall` = TP / (TP + FN) — capacidad de detectar estudios incluídos
- `sensitivity_abstract_yes` = TP / (TP + FN) **filtrando solo abstracts con gold="Include"**
- `sensitivity_abstract_no` = FN / (TP + FN) **filtrando solo abstracts con gold="Exclude"**
- `tp`, `fn`: recuentos absolutos
- `tn`, `fp`: null (muestra 100% relevante, sin estudios true-exclude de oro)
- `false_negatives`: lista de input_ids incorrectamente predichos como Exclude (debugging)

**Matriz confusión:** 2×2 (Predicted Include/Exclude × Gold Include/Exclude)

**Caveat:** Sample de oro contiene solo estudios potencialmente relevantes → **sin especificidad** (TP+TN no suma a N). El cálculo trata FN como errores críticos (falsos negativos de screening).

---

### 3.2 Hero 2 — A4 Full-Text Screening

**Entrada:** full-text completo; **Salida:** Include/Exclude (o una de 8 razones de exclusión)

**Métricas obligatorias:**

```json
"metrics": {
  "accuracy": 0.88,
  "sensitivity": 0.91,
  "specificity": 0.85,
  "f1": 0.89,
  "cohens_kappa": 0.76,
  "stratified_matrix_8x8": {
    "include": {"include": 30, "exclude_population": 0, "exclude_intervention": 1, ...},
    "exclude_population": {"include": 0, ...},
    ...
  },
  "mismatches": [
    {"input_id": "PMC789", "gold": "Include", "predicted": "Exclude (population mismatch)", "reason": "..."}
  ]
}
```

**Definiciones:**
- `accuracy` = (TP + TN) / N
- `sensitivity` = TP / (TP + FN) — capacidad de detectar incluídos
- `specificity` = TN / (TN + FP) — capacidad de detectar excluídos
- `f1` = 2 × (precision × recall) / (precision + recall)
- `cohens_kappa` = acuerdo ponderado (Cohen's kappa simple, no weighted)
- `stratified_matrix_8x8`: desagregación por los 8 códigos de decisión (Include + 7 razones de exclusión). Cada celda `[gold][predicted]` es recuento.
- `mismatches`: lista de predicciones incorrectas con motivo (para auditoría)

**Matriz confusión (resumida):** 2×2 (Include × Exclude), derivada de `stratified_matrix_8x8`

---

### 3.3 Hero 3 — A5 Data Extraction

**Entrada:** full-text + metadatos; **Salida:** diccionario de campos extraídos (N, X, SD, etc. por outcome)

**Métricas obligatorias:**

```json
"metrics": {
  "pct_fields_correct_overall": 0.84,
  "by_field_type": {
    "N": {"correct": 42, "total": 50, "pct_correct": 0.84},
    "X": {"correct": 38, "total": 50, "pct_correct": 0.76},
    "SD": {"correct": 35, "total": 50, "pct_correct": 0.70}
  },
  "by_outcome_category": {
    "objective": {"correct": 30, "total": 32, "pct_correct": 0.9375},
    "self_report": {"correct": 35, "total": 43, "pct_correct": 0.8139}
  },
  "by_timepoint": {
    "baseline": {"correct": 20, "total": 20, "pct_correct": 1.0},
    "post_intervention": {"correct": 35, "total": 45, "pct_correct": 0.7777},
    "followup": {"correct": 10, "total": 12, "pct_correct": 0.8333}
  },
  "extractions_detail": [
    {
      "input_id": "study_001",
      "gold_outcomes": [{"measure": "BDI", "n": 50, "x": 12.3, "sd": 4.1}],
      "predicted_outcomes": [{"measure": "BDI", "n": 50, "x": 12.3, "sd": 4.1}],
      "outcomes_match": "exact" | "partial" | "no_match",
      "failed_fields": []
    }
  ]
}
```

**Definiciones:**
- `pct_fields_correct_overall`: % de campos extraídos correctamente (sobre todas las muestras y campos)
- `by_field_type.{N|X|SD}.pct_correct`: % de aciertos para ese tipo de campo. **Tolerancias:**
  - N (sample size): exacto
  - X (mean): ±5%
  - SD (std dev): ±10%
- `by_outcome_category`: split por si la medida es objetiva (ej: hemoglobina) o auto-reporte (ej: DASS-21)
- `by_timepoint`: split por timepoint (baseline, post-intervention, follow-up, etc.)
- `extractions_detail`: listado de cada estudio con outcomes dorados vs predichos
  - `outcomes_match`: "exact" (todos los campos coinciden dentro tolerancia), "partial" (algunos coinciden), "no_match"
  - `failed_fields`: lista de campos que no pasaron la tolerancia

**Matriz confusión:** null (problema de regresión, no clasificación)

**Nota:** Hero 3 es el más flexible en tolerancia; el oro es valores numéricos puntuales.

---

### 3.4 Hero 4 — A6 Quality Assessment (Risk of Bias)

**Entrada:** full-text + estudio + diseño; **Salida:** RoB 2.0 (RCT, 5 dominios), ROBINS-I (observacional, 7 dominios), o JBI (cualitativo, 10 items)

**Métricas obligatorias:**

```json
"metrics": {
  "cohens_kappa_quadratic": 0.68,
  "cohens_kappa_linear": 0.72,
  "accuracy_flat": 0.71,
  "error_distribution": {
    "off_by_0": 0.71,
    "off_by_1": 0.22,
    "off_by_2": 0.07,
    "mean_abs_error": 0.36
  },
  "by_tool": {
    "rob2": {"kappa_quad": 0.75, "n": 20, "studies": ["rct_001", ...]},
    "robins_i": {"kappa_quad": 0.65, "n": 22, "studies": [...]},
    "jbi": {"kappa_quad": 0.62, "n": 8, "studies": [...]}
  },
  "by_domain": {
    "bias_of_selection": {"kappa_quad": 0.72, "n": 22},
    "bias_of_detection": {"kappa_quad": 0.68, "n": 42},
    ...
  }
}
```

**Definiciones:**
- `cohens_kappa_quadratic`: **métrica primaria**. Cohen's kappa con pesos cuadráticos (penaliza más los errores grandes).
  - Escala ordinal: L(0) < SC(1) < H(2) → distancia² = (0-1)²=1, (0-2)²=4, (1-2)²=1
  - Fórmula: κ_w = (P_o - P_e) / (1 - P_e) donde se usan pesos cuadráticos en P_o y P_e
- `cohens_kappa_linear`: **métrica secundaria**. Kappa con pesos lineales (más leniente).
  - Distancia: |0-1|=1, |0-2|=2, |1-2|=1
- `accuracy_flat`: % de exactos (predicho == oro), sin pesos
- `error_distribution`: histograma de distance(predicted, gold)
  - `off_by_0`: exacto
  - `off_by_1`: error de 1 paso (ej: L→SC o SC→H)
  - `off_by_2`: error máximo (ej: L→H)
  - `mean_abs_error`: promedio de distancias
- `by_tool`: desglose por tipo de herramienta (RoB 2.0, ROBINS-I, JBI)
  - Cada subgrupo calcula su propio kappa, recuento de estudios
- `by_domain`: desglose por dominio de riesgo (ej: "Bias of Selection", "Bias of Detection", ...)

**Matriz confusión:** 3×3 (predicted Low/SC/High × gold Low/SC/High)

**Nota:** A6 detecta automáticamente el diseño (RCT → RoB 2.0; observacional → ROBINS-I; cualitativo → JBI). El evaluador tabula cuáles estudios caen en cada categoría.

---

## 4. Configuración LLM (`llm_config`)

El campo `llm_config` **debe heredar de `config/agents.json`** (regla #9 del proyecto). No se hardcodean.

**Fuente:**
1. Lee `config/agents.json` → `agents.{hero_id}.{parámetro}`
2. Override de evaluador: temperature=0, top_p=1.0, seed=42 (determinismo)
3. max_tokens: desde agents.json por agente (típicamente: A3=150, A4=200, A5=500, A6=300)
4. repetition_penalty: desde agents.json (típicamente 1.3 en todos)

**Ejemplo de agents.json (referencia):**

```json
{
  "agent3_screening": {
    "temperature": 0.2,
    "top_p": 1.0,
    "max_tokens": 150,
    "repetition_penalty": 1.3
  },
  "agent4_fulltext": {
    "temperature": 0.2,
    "top_p": 1.0,
    "max_tokens": 200,
    "repetition_penalty": 1.3
  },
  ...
}
```

**Nota D6':** Sprint actual (T16.D) sobreescribe Ollama → **Transformers local + Qwen2.5-7B 4-bit** (stack actual de BaseAgent en `agents/base.py`). El evaluador força temperature=0 para reproducibilidad.

---

## 5. Array `raw_predictions`

Cada entrada registra una predicción individual, necesaria para debugging y auditoría.

**Esquema por entrada:**

```json
{
  "input_id": "PMC123456" | "study_001",
  "gold": "Include" | "Exclude" | {...número exacto, hero 3} | "Low" | {...},
  "predicted": "valor predicho",
  "match": true | false,
  "details": {
    "raw_llm_response": "respuesta bruta sin procesar del LLM, ej: '{...}' o 'Include: Study meets inclusion criteria'",
    "field_scores": {
      "hero1": null,
      "hero2": {"primary_reason": "...", "confidence": 0.9},
      "hero3": {"n_match": true, "x_match": true, "sd_match": false},
      "hero4": {"domain_scores": {...}}
    }
  }
}
```

**Reglas:**
- `input_id`: identificador único del estudio/muestra en el oro (ej: PMC ID, PMID, study_001, etc.)
- `gold`: valor exacto del oro (tipo depende de hero)
- `predicted`: valor predicho por el agente (mismo tipo que `gold` idealmente)
- `match`: boolean, `true` si coinciden (dentro tolerancia hero3)
- `raw_llm_response`: **siempre capturar**, sin truncar, con escapes de línea (`\n`)
- `field_scores`: hero-específico; puede ser null si no aplica. Permite debugging detallado.

---

## 6. Array `errors`

Registra fallos, timeouts, errores de parsing, etc., para diagnóstico.

```json
{
  "input_id": "PMC999",
  "error_type": "ParseError" | "TimeoutError" | "ModelError" | "ValidationError" | "IOError",
  "message": "JSON malformed: expected 'Include' or 'Exclude', got 'Includer'"
}
```

**error_type:**
- `ParseError`: JSON inválido, formato incorrecto del output del LLM
- `TimeoutError`: inferencia excedió tiempo máximo (ej: > 300s)
- `ModelError`: CUDA OOM, model loading failed, etc.
- `ValidationError`: campos faltantes, tipos incorrectos
- `IOError`: fallo al leer entrada o guardar output

---

## 7. Ruta de Salida Canónica

Cada evaluador guardará su JSON en:

```
outputs/eval/micro/{hero_id}_results_{YYYYMMDD_HHMMSS}.json
```

**Ejemplo:**
```
outputs/eval/micro/hero1_results_20260515_142301.json
outputs/eval/micro/hero2_results_20260515_142305.json
outputs/eval/micro/hero3_results_20260515_142310.json
outputs/eval/micro/hero4_results_20260515_142315.json
```

**Orquestador** (`run_heroes_micro.py`) lee todos los hero_*_results_*.json y genera:
- `outputs/eval/micro/summary_{ts}.json` — agregación
- `outputs/eval/micro/summary_{ts}.md` — reporte en español

---

## 8. Cómo Agregar un Nuevo Hero

Si en el futuro añades un 5º evaluador (ej: A7 Synthesis, A8 Manuscript):

### Paso 1: Define schema de métricas
En la **Sección 3** de este documento, añade una subsección **3.5 Hero X — Agent Y**, con:
- Entrada/Salida descripción
- Bloque JSON de `metrics` (hero-específico)
- Definiciones de cada métrica
- Tipo de matriz confusión

### Paso 2: Crea script evaluador
Archivo `experiments/eval_heroX_{name}.py`:
```python
import json
from datetime import datetime

def run_hero_evaluation(sample_path: str, model_path: str) -> dict:
    """Evalúa Agent X contra gold standard."""
    results = {
        "hero_id": "heroX",
        "agent": "AX",
        "sample_path": sample_path,
        "sample_size": ...,
        "model_used": "Qwen2.5-7B-Instruct (4-bit BitsAndBytes)",
        "llm_config": { ... },  # desde config/agents.json
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": elapsed,
        "environment": { ... },
        "metrics": { ... },  # hero X-específico
        "confusion_matrix": { ... },
        "raw_predictions": [ ... ],
        "errors": [ ... ],
        "notes": "..."
    }
    return results

if __name__ == "__main__":
    results = run_hero_evaluation(...)
    with open(f"outputs/eval/micro/heroX_results_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)
```

### Paso 3: Registra en orquestador
En `experiments/run_heroes_micro.py`, añade:
```python
from experiments.eval_heroX import run_hero_evaluation as eval_x

results_x = eval_x(sample_path=..., model_path=...)
all_results.append(results_x)
```

---

## 9. Validación y Compatibilidad

**Pre-flight checks (antes de guardar):**

```python
import jsonschema

HEROES_SCHEMA = {
    "type": "object",
    "required": [
        "hero_id", "agent", "sample_path", "sample_size",
        "model_used", "llm_config", "timestamp_utc", "elapsed_seconds",
        "environment", "metrics", "confusion_matrix", "raw_predictions", "errors"
    ],
    "properties": {
        "hero_id": {"enum": ["hero1", "hero2", "hero3", "hero4"]},
        "agent": {"enum": ["A3", "A4", "A5", "A6"]},
        "sample_path": {"type": "string"},
        "sample_size": {"type": "integer", "minimum": 1},
        "model_used": {"type": "string"},
        "llm_config": {"type": "object"},
        "timestamp_utc": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"},
        "elapsed_seconds": {"type": "number", "minimum": 0},
        "environment": {"type": "object"},
        "metrics": {"type": "object"},
        "confusion_matrix": {"type": ["object", "null"]},
        "raw_predictions": {"type": "array"},
        "errors": {"type": "array"}
    }
}

try:
    jsonschema.validate(results, HEROES_SCHEMA)
    print("✓ JSON válido conforme a schema")
except jsonschema.ValidationError as e:
    raise ValueError(f"Schema validation failed: {e.message}")
```

---

## 10. Ejemplo Completo — Hero 1 Output

```json
{
  "hero_id": "hero1",
  "agent": "A3",
  "sample_path": "data/gold_standard/deporte/samples/micro/sample_50.csv",
  "sample_size": 50,
  "model_used": "Qwen2.5-7B-Instruct (4-bit BitsAndBytes)",
  "llm_config": {
    "temperature": 0,
    "top_p": 1.0,
    "max_tokens": 150,
    "repetition_penalty": 1.3,
    "seed": 42
  },
  "timestamp_utc": "2026-05-15T14:23:01Z",
  "elapsed_seconds": 187.4,
  "environment": {
    "python_version": "3.12.5",
    "ollama_version": null,
    "gpu_info": "RTX 4060 8GB"
  },
  "metrics": {
    "sensitivity_overall": 0.92,
    "sensitivity_abstract_yes": 0.88,
    "sensitivity_abstract_no": 0.95,
    "tp": 46,
    "fn": 4,
    "tn": null,
    "fp": null,
    "false_negatives": ["PMC123456", "PMC234567", "PMC345678", "PMC456789"]
  },
  "confusion_matrix": {
    "include": {"include": 46, "exclude": 0},
    "exclude": {"include": 4, "exclude": 0}
  },
  "raw_predictions": [
    {
      "input_id": "PMC123456",
      "gold": "Include",
      "predicted": "Exclude",
      "match": false,
      "details": {
        "raw_llm_response": "{\"decision\": \"Exclude\", \"reason\": \"population mismatch\"}",
        "field_scores": {}
      }
    },
    {
      "input_id": "PMC234567",
      "gold": "Include",
      "predicted": "Include",
      "match": true,
      "details": {
        "raw_llm_response": "{\"decision\": \"Include\"}",
        "field_scores": {}
      }
    }
  ],
  "errors": [
    {
      "input_id": "PMC999999",
      "error_type": "TimeoutError",
      "message": "Inference exceeded 300s timeout"
    }
  ],
  "notes": "Run on RTX 4060, 8GB VRAM. Batch size 1. No CUDA OOM. 49/50 estudios procesados."
}
```

---

## 11. Notas para el Coordinador

**Ambigüedades y decisiones de la spec:**

1. **Hero 3 (A5) y "match":** El campo `match` es binario pero A5 tiene tolerancias (N exacto, X ±5%, SD ±10%). Solución adoptada: `match=true` si **todos** los campos de un estudio pasan tolerancia; `match=false` si al menos uno falla. Detalle en `outcomes_match` (exact/partial/no_match) y `failed_fields`.

2. **Gold standard "Escala Ordinal" en Hero 4:** RoB 2.0 usa Low/Some Concerns/High, ROBINS-I idem, JBI usa Include/Caution/Exclude. Internamente se mapean a (0, 1, 2) para cálculos de kappa. El JSON guarda los strings originales en `gold`/`predicted` para claridad, pero las métricas usan ordinales.

3. **Matriz confusión nula en Hero 3:** A5 es problema de regresión (estimar valores numéricos), no clasificación. No hay matriz 2×2 estándar. La tabla `by_field_type` y `by_outcome_category` substituye el rol de la confusión.

4. **LLM Config source:** Se asume que `config/agents.json` **siempre existe** y está actualizado per project rule #9. Si no existe, caer a defaults hardcodeados (temperatura=0.2, max_tokens=150) con warning en logs.

5. **Timestamp y timezone:** Todos los timestamps deben ser UTC con sufijo Z. Internamente en Python: `datetime.utcnow().isoformat() + "Z"`. No usar timezone local.

6. **GPU info:** Reportar la GPU específica (RTX 4060 8GB, A100 40GB) o null si CPU-only. Usar `torch.cuda.get_device_name(0)` + `torch.cuda.mem_get_info()` si GPU disponible.

7. **raw_llm_response encoding:** Si el LLM emite caracteres no-ASCII o muy largos, documentar. El JSON serializa con `json.dump(..., ensure_ascii=False)` para mantener Unicode. Si hay control chars inusuales, escaparlos manualmente o loguear warning.

8. **Validación de dates:** Los evaluadores deben producir JSONs **inmediatamente después** de correr (timestamp ≈ wall clock); no editar después. El orquestador puede hacer merge/aggregate pero no modificar `timestamp_utc` individual.

---

## Véase también

- las reglas del proyecto — Reglas de código (especialmente #9 sobre config.yaml/agents.json)
- `experiments/eval_hero1_*.py`, `experiments/eval_hero2_*.py`, etc. — Implementaciones
- `experiments/run_heroes_micro.py` — Orquestador
- `data/gold_standard/deporte/samples/micro/` — Muestras de test
- `config/agents.json` — Parámetros LLM por agente

