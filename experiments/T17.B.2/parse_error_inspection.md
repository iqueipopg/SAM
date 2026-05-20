# T17.B.2 — Inspección de parse-errors del smoke (Phase 3.5)

**Fecha**: 2026-05-21
**Smoke log**: experiments/T17.B.2/runs/smoke_V1_dev_20260521_013514.log
**Cache directory**: experiments/T17.B.2/cache/reporter/
**Contexto**: 4/15 reporter parse-errors (26.7%) vs 1/15 (6.7%) en T17.B.1 con los mismos papers. Protocol V2 (más largo, cita literal completa de Paula 2025 p.3) degrada el parseo del reporter.

## Configuración del reporter (agent4_validation_split en config/agents.json)

- `reporter_max_tokens`: **500**
- `reporter_temperature`: **0.2**
- `reporter_repetition_penalty`: **1.3**
- `decider_max_tokens`: 200, `decider_temperature`: 0.2

Schema activo en el smoke: `REPORTER_REQUIRED_KEYS_V1` (template V1 — los cache files dicen `"variant": "V1"`). 8 claves requeridas: population, intervention, comparator, outcomes, study_design, publication_type, language, **notes**.

## Diagnóstico por paper

Todos los raw outputs producen JSON sintácticamente válido (`json.loads` ok). El fallo se produce en la validación de schema en `_parse_reporter` (agent4_fulltext_split.py L380-382): falta una clave requerida → return None → fallback parse-error.

| covidence_id | gold | causa | breve descripción | acción T17.B.3 |
|---|---|---|---|---|
| #6119 | EXCLUDE | (e) schema-strict | JSON válido; emite 7/8 claves V1; **omite `notes`**. No es truncamiento — termina limpio con `}` y cierra fence ```` ``` ```` | Hacer `notes` opcional o forzarlo con few-shot |
| #2627 | EXCLUDE | (e) schema-strict | JSON válido; 7/8 claves; **omite `notes`**. Output termina limpio | Idem |
| #12682 | INCLUDE | (e) schema-strict | JSON válido; 7/8 claves; **omite `notes`**. Output termina limpio | Idem |
| #6100 | INCLUDE | (e) schema-strict + idioma | JSON válido; 7/8 claves; **omite `notes`**. Además el modelo cambia a chino mid-output (population/intervention/comparator/outcomes/study_design/publication_type/language todos en chino o mixtos) — síntoma de degradación con prompt V2 largo, pero no es causa de parse-error per se | Idem + reforzar “responde en inglés/español” |

## Síntesis

- **Causa dominante: (e) schema-strict** — los 4 fallos son JSON sintácticamente válidos rechazados por `_parse_reporter` porque omiten la clave `notes`. No es truncamiento (a), no es JSON mal formado (b), no es prosa extra (c), no es ausencia de JSON (d).
- **No es problema de max_tokens**: los 4 outputs cierran limpiamente con `}` y fence ```` ``` ```` antes del límite de 500 tokens. Subir max_tokens no soluciona nada aquí.
- **Por qué T17.B.1 (protocol V1) tenía 1/15 y T17.B.2 (protocol V2) tiene 4/15**: el protocol V2 alarga el bloque `{pico_block}` ~3x, empujando el reporter a sintetizar más en menos espacio y a saltarse el campo opcional-en-apariencia `notes` (que él interpreta como “si nada relevante, omitir” en lugar de literal `"notes": "ninguna"`). El template V1 dice explícitamente “'ninguna' si nada”, pero el reporter lo omite por completo bajo carga.
- **Hipótesis de intervención T17.B.3** (2-3 líneas): (1) en `_parse_reporter`, tratar `notes` como opcional → inyectar `"notes": "ninguna"` si falta, antes de validar; alternativa más conservadora (2) añadir 1-shot al template V1/V2 con un ejemplo que contenga `"notes": "ninguna"` literal para anclar el comportamiento. (3) Para #6100 (output en chino), añadir línea explícita al template: “Responde en inglés o español, independientemente del idioma del paper”. Opciones (1) y (3) son ortogonales y combinables.

## Apéndice — raw outputs

### #6119 (gold=EXCLUDE)
```
```json
{
  "population": "Employees aged over 50 in the Netherlands",
  "intervention": "Draft occupational health practice guideline focusing on assessing and intervening on the obesogenic work environment, including five face-to-face behavioral counseling sessions.",
  "comparator": "Usual care",
  "outcomes": "Sedentary behavior at work, fruit intake, physical activity, snacking habits, and body weight-related outcomes measured via questionnaire and physical assessments",
  "study_design": "Randomized Controlled Trial",
  "publication_type": "article_original",
  "language": "English"
}
```
```
Claves emitidas: 7/8. Falta: `notes`. JSON cerrado limpio.

### #2627 (gold=EXCLUDE)
```
```json
{
  "population": "daily smokers aged 18-60",
  "intervention": "Computer tailored personalised counseling letters based on responses to a 62-item questionnaire and stage-matched brochures.",
  "comparator": "Control group with no additional intervention beyond receiving information about being allocated to the non-treatment arm",
  "outcomes": "One-month and one-week abstinence measured using intent-to-treat analysis",
  "study_design": "Randomized Controlled Trial",
  "publication_type": "article_original",
  "language": "English"
}
```
```
Claves emitidas: 7/8. Falta: `notes`. JSON cerrado limpio.

### #12682 (gold=INCLUDE)
```
```json
{
  "population": "Cancer survivors over 18 years old, reading and understanding English",
  "intervention": "A 4-week mindfulness intervention based on Kabat-Zinn's MBSR program",
  "comparator": "Breathing control",
  "outcomes": "Psychosocial factors (stress, anxiety, depression); behavioral changes (sedentary time, perceived stress, sleep quality, daily steps)",
  "study_design": "Randomized Controlled Trial",
  "publication_type": "article_original",
  "language": "English"
}
```
```
Claves emitidas: 7/8. Falta: `notes`. JSON cerrado limpio.

### #6100 (gold=INCLUDE)
```
```json
{
  "population": "non-depressed (<PHQ2 cutoff>) veterans aged >50 years, normal Mini-Cog score, overweight or obese (BMI>25kg/m^2), sedentary生活方式（过去两周内每周运动少于90分钟）",
  "intervention": "虚拟现实干预，包括一个关于基本原理的简短介绍和如何进行有氧、阻力、平衡及灵活性锻炼的一系列指示。具体为：展示包含头像相似度高的自我形象在做身体活动。",
  "comparator": "其他条件组：显示他人或静坐的形象；没有接受任何形式的身体活动指导。",
  "outcomes": "物理活动水平的变化以及自信心变化",
  "study_design": "试点研究",
  "publication_type": "文章原稿",
  "language": "英语"
}
```
```
Claves emitidas: 7/8. Falta: `notes`. Además, el reporter cambia a chino mid-output (degradación bajo protocol V2 largo). JSON cerrado limpio.
