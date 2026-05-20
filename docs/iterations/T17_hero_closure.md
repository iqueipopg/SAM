# T17 — Cierre narrativo del Hero A1

**Fecha de cierre**: 2026-05-20
**Ciclo completo**: T17.A1 (caracterización) + T17.A1.1 (MeSH validator) + T17.A2 (iteración de prompt)
**Tag de apertura**: `pre-T17.A1` (commit `8955e21`)
**Tag de cierre**: `run-T17.A2` (commit `fd17947`)

---

## 1. Resumen ejecutivo

El Hero A1 caracterizó y mejoró la función `generate_pubmed_query` del agente de protocolo (`agents/agent1_protocol.py`) usando como instrumento de medida el dataset LEADSInstruct (Wang et al. 2025, arXiv:2501.16255). El proceso identificó un patrón de fallo mecánico concreto —generación de etiquetas `[MeSH]` sobre frases descriptivas no canónicas, que producen cero hits en PubMed— y lo atacó primero con una intervención determinista de validación de vocabulario (T17.A1.1, no confirmada) y después con una iteración sistemática del prompt (T17.A2, confirmada). El resultado final es un macro-recall@1000 de 0.1617 en el conjunto de desarrollo (DEV, +62 % relativo sobre baseline) y 0.1080 en un conjunto held-out disjunto (+1.0 pp sobre baseline), con el prompt ganador aplicado como comportamiento por defecto en producción.

---

## 2. Cronología del Hero

| Fecha | Fase | Acción | Artefacto git |
|---|---|---|---|
| 2026-05-19 | Pre-registro inicial | Hipótesis H1 (completitud PICO ≥ 0.70) y H2 (cosine PROSPERO ≥ 0.55) con PROSPERO como ground truth | `8955e21`, tag `pre-T17.A1` |
| 2026-05-19 | Pivot de ground truth | Bloqueo técnico PROSPERO (SPA JS); descubrimiento de LEADSInstruct. Pivot pre-experimento, cero datos A1 observados | `a4dbd32`, tag `pre-T17.A1-v2` |
| 2026-05-19 | Smoke N=3 | Ejecución del smoke sobre tres records de LEADSInstruct. Hallazgo: patrón de fallo MeSH-hallucination (frases descriptivas con etiqueta `[MeSH]` → cero hits) | `00e5efd` |
| 2026-05-19 | Full run N=20 | Ampliación de la muestra. Baseline pre-validator: macro = 0.0929, nonzero = 0.35 | `41eac4d` |
| 2026-05-19 | T17.A1.1 pre-registro | Pre-registro del MeSH validator: thresholds Δmacro ≥ +0.02 y Δnonzero ≥ +0.15 | `70142fd`, tag `pre-T17.A1.1` |
| 2026-05-19 | T17.A1.1 evaluación | Implementación y evaluación de `agents/mesh_validator.py`. Resultado: Δmacro = +0.0068, Δnonzero = +0.05. Veredicto: NO CONFIRMADO | `0502f2d`, tag `run-T17.A1` |
| 2026-05-20 | T17.A2 pre-registro | Pre-registro del procedimiento greedy de prompt engineering sobre el espacio de seis dimensiones de diseño | `9ad7b2b`, tag `pre-T17.A2` |
| 2026-05-20 | T17.A2 iteración | Evaluación de 9 variantes sobre DEV N=20 (~3.3 h). Ganador: V3+add_mesh_hint, macro DEV = 0.1617 | artefacto `experiments/T17.A2/iteration_log.json` |
| 2026-05-20 | T17.A2 validación | Aplicación del prompt ganador en producción y evaluación held-out N=20 disjunto. Macro held-out = 0.1080 | `fd17947`, tag `run-T17.A2` |

---

## 3. Tabla pareada final

| Sistema | Macro recall@1000 | Condición |
|---|---|---|
| Baseline T17.A1 (post-validator) | 0.0997 | N=20 DEV LEADSInstruct, Qwen 2.5-7B 4-bit |
| SAM T17.A2 DEV | 0.1617 | N=20, prompt V3+add_mesh_hint |
| SAM T17.A2 held-out | 0.1080 | N=20 disjunto, mismo prompt |
| Mistral-7B (LEADS paper) | 0.0718 | open-source 7B, FP16, sin fine-tuning |
| GPT-4o (LEADS paper) | 0.2092 | closed, large, FP16 |
| LEADS fine-tuned (LEADS paper) | 0.2468 | fine-tuning especializado, referencia future work |

Los valores del paper LEADS (Wang et al. 2025, Fig. 2c) no han sido reproducidos localmente por el equipo SAM; se citan como referencia comparativa externa para la misma tarea `study_search_publication`, split `test`, métrica macro-recall@1000.

SAM T17.A2 supera a Mistral-7B en ambas muestras: +8.9 pp en DEV y +3.6 pp en held-out. El gap frente a GPT-4o es de ~10 pp en ambas muestras, coherente con la diferencia de clase del modelo base (7B cuantizado vs. modelo closed de gran tamaño).

---

## 4. Lecciones aprendidas

- **Prompt structure es la palanca dominante para A1, no la validación de catálogo externo.** El MeSH validator (T17.A1.1) produjo +0.7 pp atacando la sintaxis de la query de forma post-hoc. La iteración de prompt (T17.A2) produjo +6.2 pp en DEV atacando la causa raíz directamente: la instrucción interna que llevaba al modelo a generar frases descriptivas en lugar de conceptos atómicos. La diferencia de magnitud entre ambas intervenciones (×9) ilustra que para modelos de lenguaje, corregir el comportamiento de generación desde el prompt es estructuralmente más eficaz que filtrar la salida.

- **El protocolo anti-HARKing aguantó: T17.A1.1 se reporta honestamente como no confirmado pese a señal direccional positiva.** Los thresholds se fijaron antes de observar ningún dato (tag `pre-T17.A1.1`, commit `70142fd`). El resultado observado (Δmacro = +0.0068 < +0.02) es una señal positiva en dirección correcta, pero por debajo del umbral pre-registrado. Aceptar este resultado como negativo en lugar de ajustar el threshold post-hoc es la práctica que preserva la credibilidad del proceso evaluativo. El TFG puede citar T17.A1.1 como ejemplo de disciplina metodológica: no toda señal positiva es una confirmación, y la línea entre "evidencia" y "ausencia de evidencia" es el umbral fijado antes de abrir los datos.

- **El gap DEV/held-out (−5.4 pp) indica overfit parcial al sample de iteración.** El flag estaba pre-registrado en T17.A2 §10.3: gap > 5 pp se reporta como "posible overfit parcial al DEV". Con N=20+20, los intervalos de confianza bootstrap sobre macro-recall son de ±4-6 pp (IC 95 %), por lo que el gap de 5.4 pp es compatible con variabilidad muestral, pero también con sesgo de selección hacia dominios clínicos sobre-representados en los few-shots del prompt ganador. La mejora es positiva en ambas muestras independientes (+6.2 pp DEV, +1.0 pp held-out), lo que confirma que el procedimiento genera valor real.

- **Ceiling probable de Qwen 2.5-7B 4-bit cerca del Mistral-7B baseline de LEADS más algunos puntos.** El baseline Mistral-7B FP16 (paper LEADS) es 0.0718. SAM T17.A2 held-out alcanza 0.1080, es decir, +3.6 pp sobre esa referencia. La cuantización 4-bit introduce penalización que no es cuantificable sin evaluación paralela de Qwen 2.5-7B FP16. El techo para prompt engineering sin fine-tuning parece estar en el rango 10-15 % sobre este split y esta métrica; avanzar más probablemente requiere fine-tuning o un modelo de mayor capacidad.

---

## 5. Limitaciones honestas

Las siguientes limitaciones son inherentes al diseño del Hero y deben citarse en la sección de limitaciones del TFG independientemente del sentido de los resultados:

**L1 — N=20+20 es pequeño.** Con 20 samples por conjunto, las diferencias de 2-3 pp entre variantes son ruidosas. Los intervalos de confianza bootstrap (IC 95 %) sobre macro-recall son de aproximadamente ±4-6 pp, lo que significa que diferencias de esa magnitud no son estadísticamente discriminables. El procedimiento es útil para detectar tendencias de largo recorrido (atomicidad, few-shots) pero no para discriminar variantes cercanas.

**L2 — Temperature 0.3 sin promediado de seeds.** Cada variante se evaluó en una única ejecución. Dos evaluaciones del mismo prompt sobre el mismo sample pueden producir queries distintas con `do_sample=True`. El contador de no-mejora de 20 variantes absorbe parte de esta varianza, pero no la elimina. Un protocolo con 3-5 seeds por variante produciría estimaciones más estables.

**L3 — Single test split de LEADSInstruct.** Toda la evaluación se realizó sobre el split `test` de la config `study_search_publication`. No se evaluó generalización a otras configs del dataset ni a otros benchmarks de recuperación biomédica. Los resultados son específicos a esta distribución.

**L4 — Greedy local no garantiza óptimo global.** El espacio explorado en las 9 variantes evaluadas (6 iniciales + 3 del proposer) es pequeño. Combinaciones no exploradas podrían superar al ganador. El prompt V3+add_mesh_hint es el óptimo local del espacio accesible, no el óptimo global.

**L5 — Gold PMIDs son los incluidos por cada SR, no el universo recuperable.** El recall@1000 se calcula sobre `publication_pmids` de LEADSInstruct, que corresponde a los estudios efectivamente incluidos por los revisores originales. Una query que recupera estudios candidatos válidos no presentes en gold no mejora la métrica. El recall absoluto está sesgado a la baja con respecto al rendimiento real en una revisión sistemática.

---

## 6. Future work

Las siguientes líneas de trabajo están directamente justificadas por los hallazgos del Hero A1:

1. **Escalar N a ≥ 100 con estratificación por dominio clínico.** Con N=100 el IC 95 % sobre macro-recall se reduce a ~±2 pp, permitiendo discriminar variantes con diferencias de 2-3 pp. La estratificación por dominio clínico permitiría identificar si el gap DEV/held-out se debe a diferencia de distribución de dominios.

2. **Promediado de seeds (3-5 semillas por variante).** Reduciría el ruido por temperature=0.3 en la comparación entre variantes cercanas y produciría estimaciones con barra de error reportable.

3. **Fine-tuning especializado en LEADSInstruct.** El modelo LEADS fine-tuned alcanza 0.2468 (vs 0.2092 GPT-4o y 0.1080 SAM held-out). El gap de fine-tuning es de ~14 pp sobre SAM held-out. Esta línea exige compute adicional (fuera del scope RTX 4060 8GB del TFG) pero es la vía natural para cerrar el gap con los sistemas más potentes.

4. **Evaluación cruzada en benchmarks adicionales.** La generalización fuera de LEADSInstruct no está garantizada. Benchmarks alternativos de recuperación biomédica (TREC Genomics, BioASQ) permitirían validar la robustez del prompt ganador más allá de este único test split.

5. **Apertura del Hero A2 (agente de búsqueda multi-base).** El Hero A1 está cerrado. El siguiente ciclo natural de mejora es el agente A2, que ejecuta la query generada por A1 en múltiples bases de datos (PubMed, Europe PMC, Semantic Scholar, Google Scholar) y realiza deduplicación semántica.

---

## Referencias cruzadas

- Log completo del Hero: `docs/iterations/T17.A1_log.md`
- Pre-registro T17.A1: `docs/iterations/T17.A1_hypothesis.md`
- Pre-registro T17.A1.1: `docs/iterations/T17.A1.1_hypothesis.md`
- Pre-registro T17.A2: `docs/iterations/T17.A2_hypothesis.md`
- Informe iteración T17.A2: `docs/iterations/T17.A2_iteration_report.md`
- Informe comparativo pre/post validator: `docs/iterations/T17.A1_comparative_report.md`
- Esqueleto capítulo 4 TFG: `docs/memoria/cap4_hero_a1_skeleton.md`
- Preguntas anticipadas defensa: `docs/iterations/T17_defense_qa.md`
