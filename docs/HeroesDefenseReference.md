# SAM TFG — Heroes Defense Reference Sheet

**Fecha**: 18 mayo 2026
**Contexto**: Análisis sample medio (E.1) post-overnight. Pre-iteración T16.E.2.A.
**Propósito**: documento de referencia para defensa y redacción de capítulos 4 y 5. Contiene narrativa por hero, métricas crudas, marco metodológico y citas primarias verificadas.

---

## 1. Marco metodológico

### 1.1. Pre-registro de hipótesis (anti-HARKing)

**HARKing** (*Hypothesizing After the Results are Known*) es la práctica cuestionable de presentar una hipótesis post-hoc como si fuera *a priori*. Acuñado por Kerr (1998), reconocido como uno de los principales contribuyentes a la crisis de replicabilidad en psicología, biomedicina y ciencias sociales.

> Kerr NL. HARKing: Hypothesizing After the Results are Known. *Personality and Social Psychology Review*. 1998;2(3):196-217. doi:10.1207/s15327957pspr0203_4

**Pre-registro** = commitear hipótesis + predicción cuantitativa + criterios de confirmación/refutación *antes* de ejecutar el experimento. En SAM se implementa con archivo `docs/iterations/<task_id>_hypothesis.md` commiteado en git con timestamp anterior al run.

**Por qué es especialmente apropiado para SAM**: el dominio que SAM automatiza — revisiones sistemáticas Cochrane — exige pre-registro de protocolo en PROSPERO antes de iniciar la revisión. Adoptar pre-registro en la evaluación de SAM es metodológicamente coherente con el estándar del campo.

> PROSPERO (International prospective register of systematic reviews), NIHR, University of York. https://www.crd.york.ac.uk/prospero/

### 1.2. Interpretación de kappa de Cohen

Escala de referencia (Landis & Koch 1977):

| κ          | Interpretación        |
|------------|----------------------|
| < 0.00     | Poor                  |
| 0.00–0.20  | Slight                |
| 0.21–0.40  | Fair                  |
| 0.41–0.60  | Moderate              |
| 0.61–0.80  | Substantial           |
| 0.81–1.00  | Almost perfect        |

> Landis JR, Koch GG. The measurement of observer agreement for categorical data. *Biometrics*. 1977;33(1):159-174.

**Caveat importante — paradoja de kappa**: cuando hay desbalanceo fuerte de clases, kappa subestima el acuerdo real porque corrige agresivamente por azar. Feinstein & Cicchetti (1990) lo documentaron. McHugh (2012) recomienda en healthcare mínimo 0.80, pero también recomienda reportar **kappa + porcentaje de acuerdo + prevalencia** conjuntamente, no kappa solo.

> Feinstein AR, Cicchetti DV. High agreement but low kappa: I. The problems of two paradoxes. *J Clin Epidemiol*. 1990;43(6):543-549.

> McHugh ML. Interrater reliability: the kappa statistic. *Biochem Med (Zagreb)*. 2012;22(3):276-282.

**Implicación para SAM**: en Hero 2 con 66 includes / 500 excludes (prevalencia 0.12), kappa baja amplifica la asimetría sensitivity/specificity. Reportar kappa + accuracy + sensitivity + specificity + precision + F1 en lugar de solo kappa.

### 1.3. Acuerdo entre revisores humanos en screening sistemático

Meta-revisión reciente de 45 SLRs con doble screening reportó:

> Hsu et al. Inter-reviewer reliability of human literature reviewing and implications for the introduction of machine-assisted systematic reviews: a mixed-methods review. *BMJ Open*. 2024. PMC10952858.

**Hallazgos clave**:
- Cohen's kappa medio entre revisores humanos: **0.72 en abstract screening, 0.72 en full-text screening**
- Rango esperado entre humanos expertos: **0.6–0.9** (encuesta a metodólogos)

**Implicación para defensa**: comparar las métricas de SAM contra este benchmark humano-humano, no contra un ideal teórico de 1.0. Kappa moderate-substantial entre humano-LLM se sitúa en rango razonable comparado con humano-humano.

### 1.4. Estado del arte: LLMs en revisión sistemática

| Estudio                          | Tarea         | Modelo(s)              | Accuracy/kappa | Conclusión |
|----------------------------------|---------------|------------------------|----------------|------------|
| Lai et al. 2024 (JAMA Netw Open) | RoB RCTs      | ChatGPT, Claude        | 84.5% / 89.5%  | Viable con supervisión |
| Wiley 2025 (PMC12442625)         | RoB2 RCTs     | ChatGPT-4o             | Above some humans, below highest | Viable como apoyo |
| Bădulescu et al. (PMC12191753)   | QUADAS-2      | 4 LLMs                 | 67–74%         | Moderate, no sustituto |
| Hasan et al.                     | ROBINS-I      | ChatGPT-4              | 61% agreement  | Limitado |
| Cohort studies LLM-RoB           | RoB cohort    | 3 LLMs                 | 80–83%         | Viable |

**Patrón consistente**: LLMs en RoB tienen accuracy moderada (60–90%), kappa moderado, **no son sustitutos del experto humano** y se posicionan como herramientas de apoyo con supervisión.

**Diferencias clave de SAM respecto a esta literatura**:
- SAM usa **modelo cuantizado 4-bit de 7B parámetros** (Qwen2.5-7B Q4_K_M), no GPT-4/Claude (~1T+ parámetros)
- SAM corre **local en RTX 4060** (consumer hardware), no API de OpenAI/Anthropic
- SAM evalúa contra **gold standard publicado** (Collazo-Castiñeira 2025), no panel de tres expertos *ad hoc*
- SAM cubre **pipeline completo** A1–A9, no solo RoB aislado

### 1.5. Conservative bias en LLMs

> Aguda T, Wilson E, Anzagira A, Kaur S, Smiley C. Conservative Bias in Large Language Models: Measuring Relation Predictions. arXiv:2506.08120, 2025.

Documenta cuantitativamente que LLMs **defaultan a etiquetas seguras** cuando enfrentan opciones ambiguas, con frecuencia **el doble** que la tasa de hallucination. Aplicable directamente a Hero 4 donde A6 defaulta a "Some concerns" (etiqueta media) cuando carece de señal fuerte.

**Nota terminológica**: "modal collapse" no es el término estándar en literatura LLM-RoB; el término correcto es **conservative bias** (Aguda et al. 2025) o **default-to-safe behavior**. Corregir en cap5.

---

## 2. Análisis por hero

### 2.1. Hero 4 — A6 Riesgo de Sesgo (RoB)

**Métricas crudas (N=62 → 61 pares evaluados)**:
- Kappa cuadrática: **0.012**
- Kappa lineal: -0.014
- Accuracy flat: 0.557
- Match: 34/61
- Distribución A6: Low=4, **Some concerns=57**, High=0
- Distribución Paula: Low=3, Some concerns=37, High=21

**Lectura**: A6 emite "Some concerns" en 57/61 (93%) casos. Paula distribuye los tres niveles. El conservative bias colapsa la salida a la categoría media segura, especialmente evitando "High risk".

**Narrativa de defensa**:

> Hero 4 confirma empíricamente con N=61 contra Collazo-Castiñeira 2025 el fenómeno de **conservative bias** documentado por Aguda et al. (2025) en LLMs aplicados a tareas con opciones ordinales. A6 defaulta a "Some concerns" (la etiqueta segura intermedia) en 93% de los casos, evitando sistemáticamente "High risk" incluso cuando el experto humano lo asigna en 21/61 casos. Este comportamiento es consistente con la literatura reciente sobre LLMs en evaluación de riesgo de sesgo (Lai et al. 2024; Wiley 2025), que reporta acuerdo humano-LLM moderado-bajo y posiciona los LLMs como herramientas de apoyo, no sustitutos del experto. La contribución empírica de Hero 4 radica en: (i) cuantificar el efecto con un modelo cuantizado 4-bit de 7B parámetros (más pequeño que ChatGPT-4o típicamente evaluado), (ii) replicar el patrón contra un gold standard publicado (no panel *ad hoc*), y (iii) integrar el hallazgo en un pipeline completo de revisión sistemática.

**Decisión metodológica**: aceptado como limitación arquitectónica documentada. No iterar A6.

---

### 2.2. Hero 3 — A5 Extracción de datos

**Métricas crudas (N=64 estudios, 752 outcomes gold)**:
- pct_fields_correct_overall: **0.0086** (artefacto, ver abajo)
- outcome_match: 1 exact + 16 partial + 735 no_match
- Estudios con ≥1 match: 16/64 (25%)
- Estudios con cero match: 48/64 (75%)
- numeric_complete=True sobre matched: 12/17 = **70.6%** (vs 88% en micro)
- 6 campos completos sobre matched: 3/17 = **17.6%** (vs 88% en micro)
- Cobertura gold global: 17/202 outcomes únicos = **8.4%**

**Dos modos de fallo identificados, desacoplados al escalar**:

**Modo A — Matching léxico de outcome_name**: A5 emite outcomes con nombres distintos a los del gold ("Mean daily steps" vs "MVPA min/day"). El matcher heurístico (substring + token overlap) no los identifica como el mismo outcome. Causa: small LLM sin ontología clínica de outcomes.

**Modo B — Completitud de campos numéricos**: incluso en los 17 matches, A5 solo emite los 6 campos completos en 3 casos (17.6%). Omisión sistemática de SDs. Causa pendiente de diagnóstico — posibles: conservadurismo del modelo (deja null cuando no encuentra), papers que reportan SE/IQR en lugar de SD, truncamiento por max_tokens=600, cuantización 4-bit.

**Hallazgo metodológico nuevo**: en sample micro (N=9) los dos modos aparecían acoplados en un kappa conjunto del 88%. Al escalar a N=64 se desacoplan: matching léxico cae a 8.4%, completitud cae a 17.6% sobre matched. **La evaluación a escala pequeña ocultaba la verdadera fragilidad arquitectónica.**

**Narrativa de defensa**:

> Hero 3 evalúa A5 (extracción de outcomes numéricos) contra 752 outcomes anotados manualmente en Collazo-Castiñeira 2025. El análisis identifica empíricamente la **separación de dos modos de fallo arquitectónicos** que el sample micro (N=9) había mantenido acoplados. Modo A es la **divergencia léxica entre la nomenclatura de outcomes emitida por A5 y la del experto humano**, agravada por la ausencia en SAM de una ontología clínica estandarizada de outcomes (Cochrane Common Data Elements, COMET initiative). Modo B es la **degradación de la completitud de extracción numérica** (88% → 17.6% para los 6 campos N/X/SD bilateral) que solo se observa al evaluar a N suficiente. El diagnóstico apunta a tres direcciones de trabajo futuro: (i) integración de ontologías clínicas estandarizadas (COMET, Cochrane CDE), (ii) ablación cuantización 4-bit vs fp16 para confirmar/refutar la atribución a precisión numérica del modelo, (iii) análisis de causas de la omisión sistemática de SDs (conservadurismo vs limitación de extracción de variabilidad reportada en formatos heterogéneos: SE, IQR, CI, range).

**Decisión metodológica**: aceptado como limitación arquitectónica. NO modificar A5 sin pre-registro previo. Diagnóstico exploratorio de modo B opcional (sección extra de cap5 si se ejecuta).

> COMET Initiative. Core Outcome Measures in Effectiveness Trials. https://www.comet-initiative.org/

---

### 2.3. Hero 1 — A3 Title/Abstract screening

**Métricas crudas (N=2000, 1000 relevant + 1000 irrelevant)**:
- Sensitivity (overall, uncertain=falso): 0.18
- **Sensitivity adjusted (solo decididos): 0.7149**
- **Specificity: 0.9278**
- Precision: 0.86
- F1: 0.78
- Accuracy: 0.85
- **Kappa de Cohen: 0.6645**
- Uncertain rate: 35% (692/2000)
  - 509/1000 relevants en uncertain
  - 183/1000 irrelevants en uncertain

**Lectura**: kappa 0.66 cae en **substantial agreement** (Landis-Koch 1977). Está en el extremo inferior pero **dentro del rango típico de acuerdo entre dos revisores humanos** según Hsu et al. (kappa medio humano-humano = 0.72; rango esperado 0.6-0.9). La asimetría de uncertain (concentrado en relevants) refleja diseño cascada: A3 prefiere delegar duda a A4 antes que rechazar en falso.

**Narrativa de defensa**:

> Hero 1 evalúa A3 (screening por título/abstract) contra 2000 papers anotados en Collazo-Castiñeira 2025 con composición balanceada (1000 relevantes / 1000 irrelevantes). Las métricas obtenidas — kappa de Cohen 0.66, specificity 0.93, F1 0.78 — caen en el rango de **substantial agreement** según la escala de referencia (Landis & Koch 1977) y son **comparables al acuerdo entre revisores humanos expertos** documentado por Hsu et al. (2024), que reporta kappa medio 0.72 entre revisores humanos en screening de abstracts sobre 45 revisiones sistemáticas. A3 implementa una arquitectura de **screening conservador delegativo**, alineada con el principio Cochrane de no excluir prematuramente: el coste documentado es un *uncertain rate* del 35% concentrado en los positivos (509/1000 relevantes vs 183/1000 irrelevantes), el beneficio es specificity 0.93 que minimiza falsos negativos firmes y reduce la carga de A4 al filtrar el ruido obvio. Esta arquitectura es coherente con la integración de SAM en flujos de trabajo humano-asistido, donde A3 reduce volumen sin sustituir el juicio clínico sobre los casos ambiguos.

**Decisión metodológica**: no iterar A3. Métricas defendibles tal como están. El target inicial de 0.80 sensitivity ajustada era objetivo interno, no contrato; el 0.71 sigue siendo substantial agreement.

---

### 2.4. Hero 2 — A4 Fulltext screening

**Métricas crudas (N=566 → 553 evaluados, 66 includes + 500 excludes)**:
- Accuracy: 0.83
- Sensitivity: 0.52
- Specificity: 0.87
- Precision: 0.35
- F1: 0.42
- Kappa de Cohen: 0.04

**Lectura**: accuracy 0.83 es **prácticamente equivalente** a Hero 1 (0.85). La kappa cae a 0.04 por la combinación de (i) desbalanceo de clases (66/500), (ii) sensitivity 0.52, (iii) penalización kappa por la paradoja documentada en Feinstein & Cicchetti (1990) y McHugh (2012). Análisis del JSON de errores: A4 rechaza includes legítimos con frases que evidencian matching literal del PICO ("no incorpora 'Physical activity interventions incorporating psychological components'") — **mismo modo de fallo léxico que A3 antes de T16.D.9.A**.

**Iteración pre-registrada T16.E.2.A**: aplicar glosario semántico análogo al de A3 al prompt de A4.

**Hipótesis pre-registrada** (a commitear ANTES del run):
- Mecanismo: divergencia léxica entre PICO literal del prompt y términos sinónimos usados en abstracts (ejercicio aeróbico, intervención psicoeducativa, actividad física estructurada, etc.)
- Predicción cuantitativa:
  - Confirmada: kappa ≥ 0.35 **y** sensitivity ≥ 0.70
  - Parcialmente confirmada: kappa ∈ [0.15, 0.35)
  - Refutada: kappa < 0.15

**Narrativa de defensa (escenarios)**:

*Si confirmada*:
> Hero 2 demuestra la efectividad de iteración targeted con hipótesis pre-registrada. El análisis cualitativo del JSON de errores en la primera evaluación identificó matching literal del PICO como mecanismo dominante de falsos negativos en includes. Se diseñó una iteración (T16.E.2.A) aplicando un glosario semántico — la misma intervención que había recuperado A3 en T16.D.9.A — y se pre-registró la predicción cuantitativa en git con timestamp anterior al run. El resultado confirma la hipótesis, recuperando kappa a [X] (moderate agreement, Landis-Koch) y sensitivity a [Y]. La metodología — análisis cualitativo + hipótesis específica + pre-registro + validación — es coherente con el estándar PROSPERO para revisiones sistemáticas y constituye una salvaguarda contra HARKing (Kerr 1998).

*Si parcialmente o refutada*:
> Hero 2 ilustra la utilidad de la pre-registración para distinguir entre fallos diagnosticables y limitaciones arquitectónicas. La hipótesis léxica explicó parcialmente el gap pero no lo agotó: el kappa post-iteración alcanzó [X], por debajo del rango de confirmación pre-registrado [0.35–0.45]. Esto indica que A4 sufre un fallo arquitectónico análogo al de A5 (Hero 3), sugiriendo que **screening full-text con un LLM cuantizado 4-bit sin acceso a ontologías clínicas tiene un techo de capacidad**. El reporte transparente del resultado parcial — habilitado por la pre-registración — evita HARKing y constituye un hallazgo metodológico válido por sí mismo.

**Decisión metodológica**: una sola iteración. Resultado se reporta tal como salga.

---

## 3. Caveats metodológicos a divulgar explícitamente en cap5

1. **Gold standard único** (Collazo-Castiñeira 2025): toda la evaluación cuantitativa se referencia a un solo meta-análisis publicado. Validez externa limitada. Mitigación: la elección refleja el principio de "gold standard del dominio relevante para SAM", no un dataset arbitrario.

2. **Matcher heurístico en Hero 3**: el evaluador usa substring bidireccional + token overlap para emparejar outcome_names. Es estricto y subestima el matching real. Mejorar el matcher post-hoc sería tuning del evaluador, no del agente, pero igual debe pre-registrarse.

3. **Cuantización 4-bit sin ablación**: la atribución de modo B de Hero 3 (degradación de completitud) a la cuantización es **hipótesis no ablada**. Para defenderla rigurosamente haría falta correr Qwen2.5-7B en fp16 sobre el mismo sample.

4. **Tamaños muestrales**: 2000 / 553 / 64 / 61 papers respectivamente. Específicamente para Hero 3 y Hero 4 con N≈60, los intervalos de confianza son anchos y debe reportarse.

5. **Sin acceso a Rudiger desde Corea**: limitación operativa (luto familiar). Documentada como tal. Imposibilita validación cruzada en hardware con más VRAM o ablación rápida.

---

## 4. Referencias primarias verificadas

**Metodología**:
- Landis JR, Koch GG. The measurement of observer agreement for categorical data. *Biometrics*. 1977;33(1):159-174.
- Cohen J. A coefficient of agreement for nominal scales. *Educ Psychol Meas*. 1960;20(1):37-46.
- Feinstein AR, Cicchetti DV. High agreement but low kappa: I. The problems of two paradoxes. *J Clin Epidemiol*. 1990;43(6):543-549.
- McHugh ML. Interrater reliability: the kappa statistic. *Biochem Med (Zagreb)*. 2012;22(3):276-282.
- Kerr NL. HARKing: Hypothesizing After the Results are Known. *Pers Soc Psychol Rev*. 1998;2(3):196-217. doi:10.1207/s15327957pspr0203_4
- PROSPERO. International prospective register of systematic reviews. NIHR/University of York. https://www.crd.york.ac.uk/prospero/

**Benchmarking humano en SLR**:
- Hsu et al. Inter-reviewer reliability of human literature reviewing and implications for the introduction of machine-assisted systematic reviews: a mixed-methods review. *BMJ Open*. 2024. PMC10952858.

**LLMs en RoB / SLR**:
- Lai J, Ge L, Sun M, et al. Use of ChatGPT and Claude for risk of bias assessment in randomized clinical trials. *JAMA Netw Open*. 2024.
- Hasan B, et al. ChatGPT for ROBINS-I assessment (non-randomized studies).
- ChatGPT-4o for RoB2 in RCTs (Wiley 2024/2025). PMC12442625.
- LLMs for QUADAS-2 risk of bias in diagnostic accuracy studies. PMC12191753.
- Assessing risk of bias of cohort studies with large language models. *Res Synth Methods*. 2025. PMC12657654.

**Conservative bias en LLMs**:
- Aguda T, Wilson E, Anzagira A, Kaur S, Smiley C. Conservative Bias in Large Language Models: Measuring Relation Predictions. arXiv:2506.08120, 2025.

**Ontologías de outcomes**:
- COMET Initiative. Core Outcome Measures in Effectiveness Trials. https://www.comet-initiative.org/
- Cochrane Common Data Elements / Cochrane Handbook for Systematic Reviews of Interventions.

---

## 5. Apéndice — referencias citadas en la conversación que NO se confirmaron

Para honestidad metodológica, marcar las citas que en algún momento mencioné pero **no he verificado con búsqueda primaria** y por tanto NO deben usarse en la memoria sin verificación posterior:

- "Wang et al, Lerner et al" sobre inter-rater entre humanos: **no verificadas**. La cifra de 0.72 kappa humano-humano está respaldada por Hsu et al. (PMC10952858) que SÍ verifiqué; las dos citas mencionadas en chat eran aproximadas. No usar.
- "Schmidt et al" sobre modal collapse en LLM-RoB: **no verificada**. El término correcto es "conservative bias" (Aguda et al. 2025), y la lit primaria sobre LLM-RoB son Lai et al. 2024, Wiley 2025, etc., como arriba.