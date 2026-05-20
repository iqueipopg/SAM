# T17 — Preguntas anticipadas para la defensa del TFG (Hero A1)

**Fecha de redacción**: 2026-05-20
**Contexto**: Hero A1 completo (T17.A1 + T17.A1.1 + T17.A2). Ciclo cerrado bajo tag `run-T17.A2` (commit `fd17947`).
**Propósito**: preparar respuestas honestas y técnicamente fundadas para el tribunal de defensa del TFG.

Las respuestas no están diseñadas para "ganar" el debate sino para anticipar preguntas legítimas y responderlas con precisión. Una respuesta honesta sobre una limitación es más sólida que una respuesta evasiva.

---

## P1. El gap DEV/held-out es de 5.4 pp. ¿No es eso sobreajuste al conjunto de desarrollo?

**Respuesta corta**: es overfit parcial, pre-registrado como bandera exacta antes de abrir los datos del held-out.

**Respuesta completa**: el flag se pre-registró en `docs/iterations/T17.A2_hypothesis.md` §10.3 antes de ejecutar ninguna variante de la iteración: "gap > 5 pp se reporta como posible overfit parcial al DEV". El gap observado es exactamente 5.4 pp, que activa la bandera.

Hay tres matices que contextualizan el gap sin eliminarlo:

1. **Mejora positiva en ambas muestras independientes.** El held-out macro (0.1080) supera al baseline T17.A1 (0.0997) en +1.0 pp. No hay regresión en el conjunto de validación externo: el prompt T17.A2 generaliza positivamente respecto al punto de partida en datos que el procedimiento de búsqueda nunca vio.

2. **Variabilidad muestral con N=20.** Los intervalos de confianza bootstrap (IC 95 %) sobre macro-recall son aproximadamente ±4-6 pp con 20 samples. Un gap de 5.4 pp es compatible con ausencia de diferencia real en la distribución poblacional. La distribución de recalls individuales es *long-tailed*: uno o dos samples con gold sets pequeños donde el recall pasa de 0 a positivo (o viceversa) pueden mover la media varios puntos porcentuales.

3. **Diferencias de distribución de dominios.** Los 20 samples del DEV y los 20 del held-out son disjuntos pero no estratificados por dominio clínico. Si el held-out contiene una proporción mayor de preguntas en dominios poco representados en los few-shots del prompt ganador, el recall será estructuralmente inferior en esa muestra. El nonzero rate inferior (0.30 vs 0.40 en DEV) es coherente con esta hipótesis, aunque no la confirma sin estratificación explícita.

**Conclusión para el tribunal**: el gap no invalida el resultado pero sí indica que la estimación del beneficio del prompt sobre el held-out (+1.0 pp) es más conservadora que la obtenida en DEV (+6.2 pp). El TFG reporta ambas cifras y las atribuye correctamente a sus respectivos conjuntos. El procedimiento pre-registró la bandera antes de observar los datos; eso es metodológicamente honesto.

---

## P2. El nonzero rate en held-out es 0.30 frente a 0.40 en DEV. ¿Qué samples siguen fallando y por qué?

**Respuesta corta**: la hipótesis dominante es dominios clínicos sub-representados en los few-shots del prompt ganador. No se hizo estratificación cuantitativa.

**Respuesta completa**: en held-out, 6/20 samples tienen recall > 0 (frente a 8/20 en DEV). Los 14 samples con recall = 0 en held-out no se analizaron individualmente en el informe T17.A2 porque el protocolo de validación held-out solo calculó métricas agregadas, no análisis por muestra de causa raíz.

La hipótesis más plausible, consistente con el análisis cualitativo del comparative report (§7 de `T17.A1_comparative_report.md`), es que los samples con recall = 0 caen en uno de dos patrones:

- **Patrón A — dominio clínico no cubierto por ningún few-shot**: el prompt ganador incluye seis ejemplos (tabaquismo/IAM, TCC/depresión, mindfulness/ansiedad, colecistectomía laparoscópica, mamografía, vacuna VPH). Preguntas sobre, por ejemplo, enfermedades raras, pediatría, psiquiatría infantil o nutrición específica pueden generar frases descriptivas en lugar de MeSH atómicos porque ningún ejemplo del prompt ancla ese dominio.

- **Patrón B — pregunta con comparador ultra-específico en lenguaje libre**: el smoke mostró que términos técnicos muy específicos ("open cot", "different types of dental implants") permanecen en cero incluso con el validator activo porque la especificidad del comparador restringe el espacio de búsqueda a un conjunto infimo de documentos. El prompt V3+add_mesh_hint instruye a usar descriptores MeSH más amplios, pero si el modelo no identifica un descriptor padre relevante para ese comparador, colapsa igual.

**Lo que no se hizo**: estratificación cuantitativa por dominio de los 40 samples (20 DEV + 20 held-out) para verificar si los dominios sub-representados explican el gap de nonzero rate. Esta sería una extensión directa del análisis que justificaría o descartaría la hipótesis A.

---

## P3. N=20+20: ¿es ese tamaño muestral suficiente para las conclusiones del TFG?

**Respuesta corta**: suficiente para detectar tendencias de largo recorrido (atomicidad, few-shots), insuficiente para discriminar variantes que difieren en menos de 3 pp.

**Respuesta completa**: con 20 samples por conjunto, el IC 95 % bootstrap sobre macro-recall es aproximadamente ±4-6 pp. Esto significa:

- Una diferencia de 6 pp entre V0 (0.0952) y V3 (0.1581) es estadísticamente robusta: supera el IC incluso en el escenario pesimista.
- Una diferencia de 0.4 pp entre V3 (0.1581) y V3+add_mesh_hint (0.1617) no es discriminable a este tamaño muestral. La selección del ganador sobre DEV con esa diferencia introduce ruido.
- El gap DEV/held-out de 5.4 pp está en el límite del IC, lo que lo hace compatible tanto con overfit real como con variabilidad muestral.

El tamaño N=20 fue elegido por restricciones de cómputo: cada variante requiere entre 67 y 72 segundos por sample (con GPU RTX 4060, carga de Qwen 2.5-7B 4-bit + llamadas PubMed E-utilities). Con N=100, cada iteración costaría ~7.000 s (~2h), haciéndolo inviable dentro del presupuesto de 4 horas establecido para el procedimiento completo.

**Conclusión**: el TFG es transparente sobre esta limitación. La magnitud de la mejora observada (tendencia de largo recorrido: atomicidad + few-shots = +6.3 pp) supera ampliamente el IC estimado, lo que la hace creíble. Las conclusiones más granulares (diferencia exacta entre V3 y V3+add_mesh_hint) deben tratarse con cautela.

---

## P4. La evaluación usa temperature=0.3 con una sola semilla. ¿La variabilidad no capturada puede cambiar el resultado?

**Respuesta corta**: sí, introduce ruido no cuantificado en las comparaciones entre variantes cercanas.

**Respuesta completa**: con `do_sample=True` y `temperature=0.3`, dos evaluaciones del mismo prompt sobre el mismo sample pueden producir queries distintas y, por tanto, distintos valores de recall@1000. El protocolo de T17.A2 no promedió entre semillas (limitación C4 pre-registrada). Esto introduce ruido en la comparación entre variantes, especialmente entre V3 (0.1581) y V3+add_mesh_hint (0.1617), cuya diferencia de 0.4 pp es probablemente indistinguible del ruido por temperatura.

Mitigación parcial: el contador de no-mejora de 20 variantes continúas sin batir el mejor actual hace que el algoritmo greedy requiera una ventaja sostenida para actualizar el best_so_far, no una ventaja puntual. Esto reduce la probabilidad de que una variante suba por azar y sea seleccionada como ganadora.

**Implicación para la memoria**: la variante ganadora no es necesariamente V3+add_mesh_hint sino "la familia V3 con o sin mesh_hint". La conclusión robusta es que atomicidad + seis few-shots multidominios produce la mayor parte de la mejora; el mesh_hint añade un beneficio marginal que puede ser ruido con una sola semilla.

---

## P5. ¿Por qué prompt engineering en lugar de fine-tuning?

**Respuesta corta**: por coste de cómputo, alcance del TFG, y porque el prompt engineering es la palanca más barata y reproducible para un agente integrado en un pipeline.

**Respuesta completa**: el fine-tuning supervisado de Qwen 2.5-7B sobre LEADSInstruct requiere:

1. Un conjunto de entrenamiento etiquetado (pares pregunta clínica → query PubMed óptima con PMIDs de recall alto), que no está disponible directamente en LEADSInstruct.
2. VRAM para el proceso de entrenamiento: fine-tuning de un modelo 7B con LoRA requiere aproximadamente 10-16 GB de VRAM, por encima de los 8 GB disponibles en la RTX 4060 del proyecto.
3. Tiempo de entrenamiento y evaluación de múltiples épocas.

El modelo LEADS fine-tuned (Wang et al. 2025) alcanza 0.2468 de macro-recall, superando a GPT-4o (0.2092). Esta referencia muestra que el fine-tuning especializado es la vía de mayor impacto a largo plazo. Sin embargo, el scope del TFG es demostrar que un pipeline multi-agente basado en LLMs open-source y prompt engineering puede producir resultados comparables a sistemas de propósito general. En ese marco, el prompt engineering es la palanca más accesible, transparente y reproducible: el prompt ganador puede replicarse en cualquier despliegue de SAM sin infraestructura adicional.

El fine-tuning queda documentado como future work con evidencia cuantitativa de su impacto potencial (+14 pp sobre el held-out actual).

---

## P6. T17.A2 es una búsqueda greedy que reporta solo al ganador. ¿No es eso HARKing?

**Respuesta corta**: es HARKing-adjacent (Kerr 1998) en el sentido de que el proceso de búsqueda informa el resultado, pero el pre-registro y la validación held-out son los mecanismos que hacen el procedimiento metodológicamente defendible.

**Respuesta completa**: en los términos de Kerr (1998), HARKing ocurre cuando se presenta como hipótesis a priori lo que en realidad fue generado por los datos observados. T17.A2 tiene las siguientes garantías que lo distinguen de HARKing puro:

1. **Pre-registro del procedimiento antes de ejecutar ninguna variante.** El espacio de dimensiones, los operadores, los criterios de parada y el protocolo held-out estaban fijados y tageados como `pre-T17.A2` antes de evaluar V0. No se añadieron operadores después de observar resultados intermedios.

2. **El veredicto es factual, no confirmatorio.** T17.A2 no afirma "se confirmó la hipótesis H". Afirma: "el procedimiento greedy produjo un macro-recall de 0.1617 en DEV y 0.1080 en held-out con el prompt V3+add_mesh_hint". Esta formulación es reproducible directamente desde los artefactos de disco.

3. **Validación held-out externa.** El held-out es un conjunto disjunto que el procedimiento de búsqueda nunca vio durante la iteración. Proporciona una estimación del rendimiento del ganador en datos no contaminados por la búsqueda.

4. **Reporte del rango completo de variantes, no solo del ganador.** La trajectory table (9 variantes) está incluida en el informe y en el capítulo de resultados del TFG. El lector puede observar que V3+add_mesh_hint no destaca por un margen dramático sobre V3 (+0.4 pp), y que V1 produjo regresión.

La limitación C6 del informe T17.A2 reconoce explícitamente el carácter HARKing-adjacent del proceso: "el proceso de búsqueda explora el espacio y el informe final presenta el extremo. La validación held-out mitiga este efecto pero no lo elimina." El TFG cita esta limitación.

---

## P7. ¿Qué garantías de generalización existen fuera de LEADSInstruct?

**Respuesta corta**: ninguna directa. SAM supera a Mistral-7B en held-out, lo que sugiere robustez parcial, pero con un único test split. La generalización no está verificada.

**Respuesta completa**: toda la evaluación del Hero A1 se realizó sobre el split `test` de la config `study_search_publication` de `zifeng-ai/LEADSInstruct`. No se evaluó:

- Generalización a la config `study_search_trial` del mismo dataset.
- Generalización a otros benchmarks de recuperación biomédica (TREC Genomics, BioASQ, PubMedQA).
- Generalización a preguntas clínicas fuera del dominio de systematic reviews (p.ej. preguntas de práctica clínica directa).

El hecho de que SAM held-out (0.1080) supere a Mistral-7B FP16 (0.0718) del paper LEADS en +3.6 pp es evidencia indirecta de que el prompt engineering sistemático produce mejoras que no son puro sobreajuste a las distribución del DEV. Pero es evidencia sobre la misma distribución (LEADSInstruct), no sobre distribuciones externas.

**Lo que el TFG afirma y lo que no**: el TFG afirma que el prompt T17.A2 mejora el rendimiento de `generate_pubmed_query` sobre el benchmark LEADSInstruct de forma reproducible. No afirma que el prompt sea óptimo para cualquier tarea de recuperación bibliográfica biomédica. La evaluación de generalización queda como future work explícito.

---

## P8. ¿Por qué no se completó la anotación manual H1 (completitud PICO)?

**Respuesta corta**: H1 requiere anotación manual de 10 sub-elementos PICO × 20 records = 200 anotaciones binarias. No se completó en el alcance del Hero A1; queda como tarea pendiente documentada.

**Respuesta completa**: H1 pre-registrada en `T17.A1_hypothesis.md` mide la proporción de sub-elementos PICO (población, intervención, comparador, outcome: 10 sub-componentes) que A1 recupera correctamente respecto a los campos `pico.P/I/C/O` de LEADSInstruct. La anotación es manual: requiere comparar el protocolo generado por A1 (en `outputs/eval/T17.A1/a1_outputs.json`) con el ground truth del dataset, elemento a elemento.

La plantilla está disponible en `docs/iterations/T17.A1_completeness_annotation.md`. La anotación no se completó porque:

1. El bloqueo técnico de PROSPERO en la Fase 1 consumió tiempo de la sesión.
2. El foco de la iteración se desplazó a la métrica funcional (recall@1000) que es más directamente relevante para el pipeline SAM que la métrica de completitud PICO.

**Consecuencia para la memoria**: H1 queda sin veredicto. El TFG puede presentar H2 (recall) como la métrica principal y H1 como métrica secundaria pendiente, con la plantilla de anotación disponible como evidencia de que el proceso estaba preparado para evaluarla.

---

## P9. ¿Por qué se modificó el pre-registro original (pivot de PROSPERO a LEADSInstruct)? ¿No invalida eso el experimento?

**Respuesta corta**: no. El pivot se realizó con cero datos de A1 observados, por causas externas a los resultados, y preservando los elementos anti-HARKing del pre-registro original.

**Respuesta completa**: los criterios de admisibilidad metodológica para un pivot pre-experimento son tres (documentados en `T17.A1_log.md` §1.5.4):

1. **Cero datos del agente observados en el momento del cambio.** No se había ejecutado ningún paso de `generate_protocol` ni `generate_pubmed_query`. No existía ninguna métrica de A1 que pudiera haber orientado la elección del nuevo instrumento.

2. **La causa del cambio es externa a los resultados.** El pivot responde al bloqueo técnico de acceso a PROSPERO (SPA JavaScript, sin API REST útil, 11 intentos fallidos documentados) y al descubrimiento de LEADSInstruct como instrumento superior. Ninguna de estas causas está relacionada con el rendimiento de A1.

3. **Los elementos anti-HARKing del pre-registro se preservaron.** H1 (threshold ≥ 0.70) se mantuvo intacto. El nuevo threshold de H2 (≥ 0.05) se calibró contra Mistral-7B (0.0718) de forma deliberadamente conservadora (por debajo del baseline de referencia), respetando el principio original de "threshold en el lado seguro". La cláusula anti-modificación se conservó.

**Diferencia sustancial con HARKing**: el pivot habría sido metodológicamente inadmisible si se hubiera realizado con datos de A1 ya generados o si el nuevo threshold se hubiera elegido para garantizar la confirmación de H2 con métricas ya conocidas. Ninguna de esas condiciones se cumple, como acredita la secuencia de tags en el historial git: `pre-T17.A1` → `pre-T17.A1-v2` → ejecución del smoke.

---

## P10. El baseline pre-registrado de H2 era ≥ 0.05. Con 0.0526 en el smoke, H2 quedó "marginalmente confirmada". ¿No es ese margen demasiado pequeño para ser significativo?

**Respuesta corta**: sí. La confirmación marginal con N=3 tiene valor instrumental (la mecánica del instrumento funciona) pero no valor inferencial sobre el rendimiento de A1.

**Respuesta completa**: el smoke N=3 no está diseñado para producir estimaciones robustas del recall de A1. Su propósito es validar la mecánica del instrumento de evaluación (sanity checks 7/7) y obtener una primera señal cualitativa. En ese marco, "H2 marginalmente confirmada" significa "el instrumento es operativo y produce algún resultado no degenerado", no "A1 cumple el threshold de forma fiable".

El margen de +0.0026 sobre el umbral (0.0526 vs 0.05) es el resultado de un único sample (MLSEP017395 con recall = 0.1579) que empuja la media por encima del umbral. La mediana del smoke es 0.0000. Con N=3 y esa distribución, la confirmación marginal no puede interpretarse como evidencia de que A1 supera de forma generalizada el threshold pre-registrado.

El full run N=20 (baseline T17.A1 post-validator: macro = 0.0997) es la estimación más robusta disponible del rendimiento real de A1 sobre LEADSInstruct. El smoke cumplió su función de diagnóstico; el full run es la estimación de referencia.

---

*Documento de preparación para defensa. Refleja el estado del Hero A1 en el cierre `run-T17.A2` (2026-05-20). Fuentes primarias: `docs/iterations/T17.A1_log.md`, `docs/iterations/T17.A2_iteration_report.md`, `docs/iterations/T17_hero_closure.md`.*
