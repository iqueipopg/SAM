# A3 — Glosario semántico (T16.D.9.A)

- **Qué es**: bloque de equivalencias semánticas + política de título-only embebido en el prompt inline de A3 para que el LLM reconozca "psychological components" del PICO de Paula sin literal-string matching.
- **Cuándo se añadió**: T16.D.9.A, 2026-05-16.
- **Dónde vive**: `agents/agent3_screening.py`, función `_single_vote` (no `_screen_paper_once`), líneas ~275-297 al momento de la extracción.
- **Por qué se añadió**: A3 fallaba al excluir papers que sí incluían intervenciones psicológicas relevantes (MI, mindfulness, CBT, ACT, MBSR, behaviour change theory, goal-setting, self-monitoring…) porque el matching era literal contra la cadena "psychological components".

## Texto literal del bloque

```
SEMANTIC EQUIVALENCE (avoid literal-string matching):
Treat the intervention "{intv}" as a concept, not a literal string. If "{intv}"
mentions broad psychological/behavioural constructs (e.g. "psychological
components", "behaviour change", "self-management"), the following are
operationally equivalent and should be treated as MEETING the criterion, even
if the exact phrase is not used:
- behaviour change theory (COM-B, TTM, HBM, SCT, Self-Determination Theory)
- goal-setting, self-monitoring, action planning
- motivational interviewing, coaching, counseling, psychoeducation
- mindfulness, MBSR, MBCT, CBT, ACT
- self-efficacy, self-regulation, self-management interventions
- theory-informed digital / SMS / mobile-health behaviour interventions
- text-messaging programmes anchored in behaviour-change techniques (BCTs)
Apply this guidance only when "{intv}" includes a psychological or behavioural
concept. Do NOT invent equivalences for unrelated criteria.

TITLE-ONLY POLICY (when abstract is absent or empty):
- INCLUDE   if the title contains BOTH a topic term AND an inclusion-relevant
            concept (or its semantic equivalent above).
- EXCLUDE   if the title clearly addresses an unrelated topic (e.g. drug
            therapy only, surgery, paediatrics, animal study, basic science).
- UNCERTAIN ONLY if the title is uninformative or corrupted (e.g. a file name
            like "DX096272_1_0001.tif", "en_AO05.indd", "301 hawthorne.qxd").
```

---

Este documento se mantiene como referencia para decidir si trasladar el mismo glosario semántico a A4 (Phase B: full-text screening) o ampliarlo con términos PICO-específicos del MA de Paula (p. ej. ejercicio aeróbico, resistencia, intervenciones psicoeducativas estructuradas). El bucket "Wrong age → Wrong intervention" (316/385, 82%) detectado en T16.E.1.fix sugiere que A4 se beneficiaría de un bloque equivalente.
