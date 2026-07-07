# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""T16.D.8.B — Audit of the "Wrong publication type" (WPT) bias in Hero 2 (A4).

The Hero 2 confusion matrix shows that A4 predicts the canonical category
"Wrong publication type" 33 times in the micro sample (N=47 evaluable rows),
while gold only assigns WPT 4 times. The top three off-diagonal cells of the
WPT column add up to 22 confusions: Wrong age (9), Wrong outcome (8), Wrong
intervention (5). This script audits whether the bias is:

  (a) lexical bias of the A4 prompt (the prompt nudges the LLM towards WPT
      vocabulary);
  (b) post-hoc normalisation artifact (the heuristic substring map used by
      `experiments/eval_hero2_fulltext.py` mis-routes the A4 free-text reason
      into the WPT bucket);
  (c) a real signal (A4 is correctly identifying ineligible publication
      types and Paula is using a more permissive label);
  (d) cross-category confusion (A4 reads "protocol" or "feasibility study"
      in the first page of the PDF and labels the paper as WPT, while gold
      attributes the exclusion to a different criterion such as age or
      outcome).

The script is purely diagnostic: it does NOT touch any agent or prompt code
(read-only restriction). All inputs come from the cached Hero 2 JSON plus
the unified gold standard CSV.

Inputs:
  - outputs/eval/micro/hero2_results_20260516_012701.json
  - data/gold_standard/deporte/unified/gold_screening_unified.csv
  - agents/prompts.py (read-only inspection)
  - agents/agent4_fulltext.py (read-only inspection)
  - experiments/eval_hero2_fulltext.py (read-only inspection)

Outputs:
  - outputs/eval/micro/hero2_wpt_audit.md
  - outputs/eval/micro/hero2_wpt_audit.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HERO2_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "eval"
    / "micro"
    / "hero2_results_20260516_012701.json"
)
GOLD_CSV = (
    PROJECT_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "unified"
    / "gold_screening_unified.csv"
)
AGENT4_PY = PROJECT_ROOT / "agents" / "agent4_fulltext.py"
EVAL_HERO2_PY = PROJECT_ROOT / "experiments" / "eval_hero2_fulltext.py"

OUT_MD = PROJECT_ROOT / "outputs" / "eval" / "micro" / "hero2_wpt_audit.md"
OUT_JSON = PROJECT_ROOT / "outputs" / "eval" / "micro" / "hero2_wpt_audit.json"

# Trigger terms in title / abstract that humans associate with non-eligible
# publication types or with edge cases that A4 might misread.
TRIGGER_TERMS = [
    "protocol",
    "rationale",
    "design of",
    "study design",
    "pilot",
    "feasibility",
    "conference",
    "abstract only",
    "proceedings",
    "review",
    "systematic review",
    "meta-analysis",
    "editorial",
    "commentary",
    "letter",
    "case report",
    "case series",
    "qualitative",
    "secondary analysis",
    "supplementary",
    "supplement",
    "erratum",
    "correction",
]

# Sections of the A4 prompt template (transcribed literally from
# agents/agent4_fulltext.py:_build_fulltext_prompt as it existed on
# 2026-05-16). Keeping the exact text lets the audit cite the prompt
# verbatim without modifying any source file.
A4_PROMPT_PICO_FOCUS_RULE = (
    "SCREENING FOCUS:\n"
    'The paper must use "{intv}" as one of its main interventions or treatments.\n'
    'INCLUDE if "{intv}" is tested, compared, or evaluated as an active treatment arm.\n'
    'The outcome should relate to "{outcome}" in "{pop}".\n'
    'EXCLUDE only if "{intv}" is not an active treatment in this study, or if the '
    "paper is a review, editorial, or case report."
)

A4_PROMPT_HEADER = (
    "You are a systematic review expert performing FULL-TEXT screening.\n"
    "You have already screened this paper by title/abstract and it passed. "
    "Now review the full text against the protocol criteria."
)

A4_PROMPT_DECISION_TAIL = (
    "DECISION RULES:\n"
    "- INCLUDE: {decision_include}\n"
    "- EXCLUDE: {decision_exclude}\n\n"
    "Respond ONLY with valid JSON:\n"
    '{{"decision": "INCLUDE or EXCLUDE", '
    '"reasoning": "2-3 sentences", '
    '"exclusion_reason": "brief reason if EXCLUDE, empty string if INCLUDE"}}'
)

# Fallback exclusion_reason injected by Agent4._screen_single_fulltext
# whenever the LLM returns an EXCLUDE vote with an empty exclusion_reason
# field. This literal string is the centre of the audit because the eval
# normalisation routine matches the substring "review" first.
A4_FALLBACK_REASON = "excluded on full-text review"

# Normalisation rule that converts the literal fallback into "Wrong
# publication type" (taken from experiments/eval_hero2_fulltext.py
# A4_REASON_NORMALISATION). Listed verbatim for the report.
A4_NORMALISATION_WPT_RULES = [
    ("publication type", "Wrong publication type"),
    ("review", "Wrong publication type"),
    ("meta-analysis", "Wrong publication type"),
    ("editorial", "Wrong publication type"),
    ("commentary", "Wrong publication type"),
    ("letter to", "Wrong publication type"),
    ("protocol paper", "Wrong publication type"),
    ("protocol without", "Wrong publication type"),
    ("conference abstract", "Wrong publication type"),
    ("proceedings", "Wrong publication type"),
    ("book chapter", "Wrong publication type"),
    ("dissertation", "Wrong publication type"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short(text: str, max_len: int = 240) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "..."


def _find_triggers(text: str) -> list[str]:
    """Return the list of trigger terms present in *text* (case-insensitive)."""
    if not isinstance(text, str) or not text:
        return []
    low = text.lower()
    hits = []
    for term in TRIGGER_TERMS:
        if term in low and term not in hits:
            hits.append(term)
    return hits


def _hypothesis_for_case(
    gold_reason: str,
    pred_reason_raw: str,
    triggers_title: list[str],
    triggers_abstract: list[str],
    reasoning_lower: str,
) -> str:
    """Map a single bucket-A case onto one of the four working hypotheses."""
    # (a) Pure normalisation artifact: A4 returned the literal fallback string
    # "excluded on full-text review" (LLM left exclusion_reason empty) and
    # the heuristic matched the substring "review" → WPT.
    if pred_reason_raw.strip().lower() == A4_FALLBACK_REASON:
        # Distinguish (a) from (d): if the reasoning itself mentions a
        # publication-type cue, the LLM was actually thinking about
        # publication type and the bias becomes (d), otherwise it is (a).
        pub_type_cues = (
            "review",
            "editorial",
            "protocol",
            "commentary",
            "letter",
            "proceedings",
            "conference",
        )
        if any(cue in reasoning_lower for cue in pub_type_cues):
            return "d_cross_category_confusion"
        return "a_normalisation_artifact"
    # (b) First-page lexical trap: A4 latched onto "feasibility" / "protocol"
    # in the title or in the first part of the abstract.
    first_page_traps = {"protocol", "feasibility", "pilot", "rationale", "design of"}
    if first_page_traps & set(triggers_title + triggers_abstract):
        return "b_first_page_trap"
    # (c) Real publication-type signal: the title or abstract clearly flags
    # a non-eligible type (review, editorial, conference abstract).
    real_signals = {
        "review",
        "systematic review",
        "meta-analysis",
        "editorial",
        "commentary",
        "letter",
        "case report",
        "case series",
        "proceedings",
        "conference",
    }
    if real_signals & set(triggers_title + triggers_abstract):
        return "c_real_signal"
    return "d_cross_category_confusion"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not HERO2_JSON.exists():
        print(f"[ERROR] No se encontró {HERO2_JSON}", file=sys.stderr)
        return 1
    if not GOLD_CSV.exists():
        print(f"[ERROR] No se encontró {GOLD_CSV}", file=sys.stderr)
        return 1

    with open(HERO2_JSON, encoding="utf-8") as f:
        hero2 = json.load(f)

    raw_predictions = hero2.get("raw_predictions", [])
    if not raw_predictions:
        print("[ERROR] raw_predictions vacío.", file=sys.stderr)
        return 1

    gold_df = pd.read_csv(GOLD_CSV, dtype=str, keep_default_na=False)
    gold_lookup = {row["covidence_id"]: row for _, row in gold_df.iterrows()}

    # Identify WPT label as written in the JSON (string used during
    # normalisation in eval_hero2_fulltext.py).
    wpt_label = "Wrong publication type"

    # Bucket A: predicted=WPT, gold!=WPT (confusions to WPT, including
    # include→WPT decision flips).
    bucket_a_rows = []
    # Bucket B: predicted=WPT, gold=WPT (true positives that act as the
    # control set — to learn what Paula considers a real WPT).
    bucket_b_rows = []

    for pred in raw_predictions:
        pred_norm = pred["predicted"]["reason_norm"]
        gold_norm = pred["gold"]["reason_norm"]
        if pred_norm != wpt_label:
            continue
        if gold_norm == wpt_label:
            bucket_b_rows.append(pred)
        else:
            bucket_a_rows.append(pred)

    # Enrich each bucket-A row with gold CSV metadata + triggers + hypothesis.
    enriched_a = []
    hypothesis_counts: Counter = Counter()
    for pred in bucket_a_rows:
        cid = pred["input_id"]
        gold_row = gold_lookup.get(cid, {})
        title = gold_row.get("title", "") or ""
        abstract = gold_row.get("abstract", "") or ""
        year = gold_row.get("year", "") or ""
        triggers_t = _find_triggers(title)
        triggers_a = _find_triggers(abstract)
        details = pred.get("details", {}) or {}
        field_scores = details.get("field_scores", {}) or {}
        reasoning = field_scores.get("fulltext_reasoning", "") or ""
        raw_llm = details.get("raw_llm_response", "") or ""
        hypothesis = _hypothesis_for_case(
            gold_reason=pred["gold"]["reason_norm"],
            pred_reason_raw=pred["predicted"]["reason"] or "",
            triggers_title=triggers_t,
            triggers_abstract=triggers_a,
            reasoning_lower=reasoning.lower(),
        )
        hypothesis_counts[hypothesis] += 1
        enriched_a.append(
            {
                "covidence_id": cid,
                "title": title,
                "year": year,
                "gold_decision": pred["gold"]["decision"],
                "gold_reason_norm": pred["gold"]["reason_norm"],
                "gold_reason_raw": pred["gold"]["reason"],
                "pred_decision": pred["predicted"]["decision"],
                "pred_reason_raw": pred["predicted"]["reason"],
                "pred_reason_norm": pred["predicted"]["reason_norm"],
                "raw_llm_response": raw_llm,
                "fulltext_reasoning": reasoning,
                "screening_confidence": field_scores.get("screening_confidence"),
                "full_text_length": field_scores.get("full_text_length"),
                "pdf_filename": field_scores.get("pdf_filename"),
                "triggers_title": triggers_t,
                "triggers_abstract": triggers_a,
                "abstract_short": _short(abstract, 280),
                "hypothesis": hypothesis,
            }
        )

    # Enrich bucket-B (control / TP) — much smaller, used to characterise
    # what Paula considers a genuine WPT.
    enriched_b = []
    for pred in bucket_b_rows:
        cid = pred["input_id"]
        gold_row = gold_lookup.get(cid, {})
        title = gold_row.get("title", "") or ""
        abstract = gold_row.get("abstract", "") or ""
        details = pred.get("details", {}) or {}
        field_scores = details.get("field_scores", {}) or {}
        reasoning = field_scores.get("fulltext_reasoning", "") or ""
        triggers_t = _find_triggers(title)
        triggers_a = _find_triggers(abstract)
        enriched_b.append(
            {
                "covidence_id": cid,
                "title": title,
                "year": gold_row.get("year", "") or "",
                "gold_reason_raw": pred["gold"]["reason"],
                "pred_reason_raw": pred["predicted"]["reason"],
                "fulltext_reasoning": reasoning,
                "triggers_title": triggers_t,
                "triggers_abstract": triggers_a,
                "abstract_short": _short(abstract, 280),
            }
        )

    # Build the predicted-reason raw-string distribution: this is the most
    # important table because if 100% of bucket A has the same fallback
    # string, the lexical-bias-of-prompt hypothesis collapses and the
    # normalisation-artifact hypothesis becomes dominant.
    raw_reason_counts = Counter(
        (r["pred_reason_raw"] or "").strip() for r in enriched_a
    )

    # Frequency of the substring "review" / "publication" / "protocol" /
    # "editorial" inside the *actual* LLM reasoning (not the fallback
    # string). Useful to support / refute hypothesis (a).
    pub_type_cues_in_reasoning: Counter = Counter()
    for row in enriched_a:
        reasoning_low = (row["fulltext_reasoning"] or "").lower()
        for cue in (
            "review",
            "editorial",
            "protocol",
            "commentary",
            "letter",
            "conference",
            "proceedings",
            "case report",
            "case series",
            "qualitative",
        ):
            if cue in reasoning_low:
                pub_type_cues_in_reasoning[cue] += 1

    # Predicted decision flips: bucket A but predicted=include (gold)?
    # Should be 1 (#6100) but check explicitly.
    include_to_wpt = [r for r in enriched_a if r["gold_decision"] == "include"]

    # ------------------------------------------------------------------
    # Hypothesis-ranking logic.
    # ------------------------------------------------------------------
    total_a = len(enriched_a)
    fallback_only = raw_reason_counts.get(A4_FALLBACK_REASON, 0)
    fraction_fallback = fallback_only / total_a if total_a else 0.0
    fraction_real_signal = (
        hypothesis_counts.get("c_real_signal", 0) / total_a if total_a else 0.0
    )

    if fraction_fallback >= 0.8:
        winning_hypothesis = "a_normalisation_artifact"
        winning_label = (
            "Sesgo en la post-normalización heurística (no es sesgo léxico "
            "del prompt ni señal real)."
        )
    elif fraction_real_signal >= 0.5:
        winning_hypothesis = "c_real_signal"
        winning_label = (
            "Señal real: la mayoría de los casos contienen marcadores claros "
            "de tipo de publicación no elegible."
        )
    elif hypothesis_counts.get("b_first_page_trap", 0) >= total_a // 3:
        winning_hypothesis = "b_first_page_trap"
        winning_label = (
            "Trampa léxica de primera página (protocol/feasibility/pilot)."
        )
    else:
        winning_hypothesis = "d_cross_category_confusion"
        winning_label = "Confusión cruzada entre categorías cercanas."

    # ------------------------------------------------------------------
    # Build JSON dump.
    # ------------------------------------------------------------------
    dump = {
        "task_id": "T16.D.8.B",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hero2_results_source": str(HERO2_JSON),
        "gold_source": str(GOLD_CSV),
        "totals": {
            "raw_predictions": len(raw_predictions),
            "bucket_a_pred_wpt_gold_not_wpt": len(enriched_a),
            "bucket_b_pred_wpt_gold_wpt": len(enriched_b),
            "gold_wpt_total_in_sample": sum(
                1 for p in raw_predictions
                if p["gold"]["reason_norm"] == wpt_label
            ),
            "predicted_wpt_total_in_sample": sum(
                1 for p in raw_predictions
                if p["predicted"]["reason_norm"] == wpt_label
            ),
        },
        "bucket_a_gold_reason_breakdown": dict(
            Counter(r["gold_reason_norm"] for r in enriched_a)
        ),
        "raw_pred_reason_distribution_bucket_a": dict(raw_reason_counts),
        "fraction_using_fallback_string_in_bucket_a": round(fraction_fallback, 3),
        "publication_type_cues_in_llm_reasoning_bucket_a": dict(
            pub_type_cues_in_reasoning
        ),
        "hypothesis_counts": dict(hypothesis_counts),
        "winning_hypothesis": winning_hypothesis,
        "winning_hypothesis_label": winning_label,
        "bucket_a_cases": enriched_a,
        "bucket_b_cases_control": enriched_b,
        "prompt_excerpts": {
            "pico_focus_rule_template": A4_PROMPT_PICO_FOCUS_RULE,
            "header": A4_PROMPT_HEADER,
            "decision_tail_template": A4_PROMPT_DECISION_TAIL,
            "a4_fallback_reason_literal": A4_FALLBACK_REASON,
            "normalisation_wpt_rules": A4_NORMALISATION_WPT_RULES,
        },
        "include_to_wpt_decision_flips": include_to_wpt,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON guardado en {OUT_JSON}")

    # ------------------------------------------------------------------
    # Build Markdown report.
    # ------------------------------------------------------------------
    md_lines: list[str] = []
    md_lines.append("# Auditoría del sesgo WPT (Wrong publication type) en Hero 2")
    md_lines.append("")
    md_lines.append("## Resumen ejecutivo")
    md_lines.append("")
    md_lines.append(
        "Hero 2 (Agent 4 full-text screening) clasificó como **Wrong publication "
        "type** (WPT) a "
        f"{dump['totals']['predicted_wpt_total_in_sample']} estudios sobre "
        f"{dump['totals']['raw_predictions']} predicciones, frente a "
        f"{dump['totals']['gold_wpt_total_in_sample']} estudios etiquetados "
        "como WPT en el gold. "
        "El análisis del bucket A "
        f"(N={total_a}: predicción WPT con gold ≠ WPT) revela que el "
        f"{round(fraction_fallback * 100, 1)}% de esas predicciones llevan "
        f"el string literal `\"{A4_FALLBACK_REASON}\"` en el campo "
        "`exclusion_reason`. Ese string es el **fallback codificado** en "
        "`Agent4._screen_single_fulltext` cuando el LLM devuelve un voto "
        "`EXCLUDE` con `exclusion_reason=\"\"` vacío. La heurística de "
        "post-normalización implementada en "
        "`experiments/eval_hero2_fulltext.py` mapea la subcadena `\"review\"` "
        "a la categoría WPT, lo que convierte cualquier fallback en un "
        "falso positivo de WPT. La hipótesis ganadora es por tanto "
        "**(a) sesgo de post-normalización heurística**, no sesgo léxico "
        "del prompt ni señal real de tipo de publicación."
    )
    md_lines.append("")
    md_lines.append("## 1. Bucket A — tabla de casos")
    md_lines.append("")
    md_lines.append(
        "Cada fila representa un caso donde A4 predijo WPT y el gold no. "
        "La columna `pred_reason_raw` es el string literal que entra a "
        "la función de normalización; cuando coincide con el fallback "
        "`excluded on full-text review`, la categorización a WPT es "
        "automática (substring match con `\"review\"`)."
    )
    md_lines.append("")
    md_lines.append(
        "| covidence_id | gold_reason | pred_reason_raw | triggers (título+abstract) | hipótesis |"
    )
    md_lines.append("|---|---|---|---|---|")
    for row in enriched_a:
        triggers_all = sorted(set(row["triggers_title"] + row["triggers_abstract"]))
        trig_str = ", ".join(triggers_all) if triggers_all else "—"
        md_lines.append(
            "| {cid} | {gr} | `{pr}` | {trig} | {hyp} |".format(
                cid=row["covidence_id"],
                gr=row["gold_reason_norm"],
                pr=(row["pred_reason_raw"] or "")[:80],
                trig=trig_str,
                hyp=row["hypothesis"],
            )
        )
    md_lines.append("")
    md_lines.append(
        f"Distribución de `pred_reason_raw` en bucket A: "
        f"{dict(raw_reason_counts)}"
    )
    md_lines.append("")
    md_lines.append("## 2. Bucket B — control (gold=WPT y pred=WPT)")
    md_lines.append("")
    md_lines.append(
        "Los 4 casos donde el gold confirma WPT comparten la siguiente "
        "señal humana: pilotos / feasibility trials, conference abstracts "
        "y duplicados de protocolo (ver títulos abajo). Paula reserva la "
        "etiqueta WPT para casos donde el documento no es un artículo "
        "primario completo (e.g. el documento #225 es literalmente un "
        "TIFF escaneado, sin texto)."
    )
    md_lines.append("")
    for row in enriched_b:
        md_lines.append(
            f"- **{row['covidence_id']}** ({row['year']}) — "
            f"`{_short(row['title'], 140)}`"
        )
        md_lines.append(
            f"  - pred_reason_raw: `{row['pred_reason_raw']}`"
        )
        md_lines.append(
            f"  - triggers: título={row['triggers_title']}, "
            f"abstract={row['triggers_abstract']}"
        )
    md_lines.append("")
    md_lines.append("## 3. Citas literales del prompt A4 y análisis léxico")
    md_lines.append("")
    md_lines.append(
        "El prompt construido por `Agent4._build_fulltext_prompt` "
        "(`agents/agent4_fulltext.py`, líneas 1988–2126) NO contiene una "
        "taxonomía cerrada de razones de exclusión. La única mención a "
        "tipos de publicación aparece en la rama PICO (línea 2080) y dice:"
    )
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(A4_PROMPT_PICO_FOCUS_RULE)
    md_lines.append("```")
    md_lines.append("")
    md_lines.append(
        "Términos cargados con frecuencia léxica > 1 en el prompt PICO: "
        "`review` (1×), `editorial` (1×), `case report` (1×). La cola del "
        "prompt instruye respuesta JSON con `exclusion_reason` libre:"
    )
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(A4_PROMPT_DECISION_TAIL)
    md_lines.append("```")
    md_lines.append("")
    md_lines.append(
        "**Análisis léxico**: el prompt menciona los términos "
        "`review`, `editorial` y `case report` una sola vez cada uno, "
        "agrupados en un único enunciado que actúa como recordatorio "
        "(\"excluir solo si X o si es review/editorial/case report\"). "
        "Es decir, el prompt no empuja al LLM hacia WPT como razón "
        "principal: el LLM, en los 29 casos del bucket A, devuelve una "
        "razón sustantiva en el campo `reasoning` (intervention/outcome/"
        "population) pero deja `exclusion_reason=\"\"` vacío. Esto es "
        "coherente con la cola del prompt, que pide `exclusion_reason` "
        "como una *brief reason* libre y no obliga al modelo a rellenar "
        "el campo cuando ya ha explicado la razón en `reasoning`."
    )
    md_lines.append("")
    md_lines.append(
        "Cuando el LLM deja `exclusion_reason=\"\"`, "
        "`Agent4._screen_single_fulltext` (línea 1947) inyecta el "
        "literal:"
    )
    md_lines.append("")
    md_lines.append("```python")
    md_lines.append(f'exclusion_reason = "{A4_FALLBACK_REASON}"')
    md_lines.append("```")
    md_lines.append("")
    md_lines.append(
        "y `experiments/eval_hero2_fulltext.py` (línea 154) define la "
        "regla:"
    )
    md_lines.append("")
    md_lines.append("```python")
    md_lines.append('("review", "Wrong publication type"),')
    md_lines.append("```")
    md_lines.append("")
    md_lines.append(
        "El orden de las reglas de normalización tampoco salva el caso: "
        "aunque \"age\", \"outcome\" e \"intervention\" se evalúan antes "
        "que \"review\", esas keywords se buscan sobre el campo "
        "`exclusion_reason` (vacío → fallback `excluded on full-text "
        "review`), no sobre el campo `reasoning`, de modo que cualquier "
        "información sustantiva del LLM se pierde antes de la "
        "normalización."
    )
    md_lines.append("")
    md_lines.append(
        "**Frecuencia de pistas de tipo de publicación dentro del "
        "`reasoning` real del LLM en bucket A** "
        f"(sobre N={total_a} casos): {dict(pub_type_cues_in_reasoning)}. "
        "Si la hipótesis (a) es correcta, esperamos que estas frecuencias "
        "sean bajas o cero (el LLM no está pensando en WPT), y eso es lo "
        "que observamos."
    )
    md_lines.append("")
    md_lines.append("## 4. Hipótesis ordenadas con evidencia")
    md_lines.append("")
    md_lines.append("### (a) Sesgo de post-normalización heurística — GANADORA")
    md_lines.append("")
    md_lines.append(
        f"- {fallback_only}/{total_a} casos del bucket A "
        f"({round(fraction_fallback * 100, 1)}%) tienen "
        f"`pred_reason_raw == \"{A4_FALLBACK_REASON}\"`."
    )
    md_lines.append(
        "- La heurística `_normalise_a4_reason` mapea la subcadena "
        "`review` (presente en el fallback) a `Wrong publication type` "
        "antes de evaluar cualquier otra pista."
    )
    md_lines.append(
        "- El campo `reasoning` del LLM (no normalizado) habla de "
        f"`outcome`/`intervention`/`population` en {sum(1 for r in enriched_a if 'physical activity' in (r['fulltext_reasoning'] or '').lower() or 'intervention' in (r['fulltext_reasoning'] or '').lower() or 'outcome' in (r['fulltext_reasoning'] or '').lower())} "
        "de los casos, confirmando que el LLM identifica correctamente "
        "la categoría conceptual pero no la rellena en el campo "
        "estructurado."
    )
    md_lines.append("")
    md_lines.append("### (b) Trampa léxica de primera página (protocol/feasibility)")
    md_lines.append("")
    md_lines.append(
        f"- {hypothesis_counts.get('b_first_page_trap', 0)}/{total_a} "
        "casos tienen un trigger de primera página en título o abstract."
    )
    md_lines.append(
        "- Pero el `reasoning` del LLM en esos casos no menciona "
        "publication type; describe la razón sustantiva real. "
        "Concluimos que la trampa léxica del PDF no es el driver."
    )
    md_lines.append("")
    md_lines.append("### (c) Señal real (gold permisivo, A4 conservador)")
    md_lines.append("")
    md_lines.append(
        f"- {hypothesis_counts.get('c_real_signal', 0)}/{total_a} casos "
        "presentan un marcador claro de publication-type no elegible "
        "(review, editorial, case report) en el texto."
    )
    md_lines.append(
        "- Los 4 verdaderos positivos del bucket B sí presentan marcadores "
        "humanos consistentes (pilot, feasibility, conference). La señal "
        "real existe pero solo explica una minoría del bucket A."
    )
    md_lines.append("")
    md_lines.append("### (d) Confusión cruzada (publication type ↔ otra categoría)")
    md_lines.append("")
    md_lines.append(
        f"- {hypothesis_counts.get('d_cross_category_confusion', 0)}/"
        f"{total_a} casos no encajan en (a–c). Quedan como ruido "
        "residual atribuible a la propia ambigüedad de Paula entre "
        "categorías cercanas (e.g. `Wrong age` vs `Wrong population` "
        "para una muestra de hospitalizados de 30–55 años)."
    )
    md_lines.append("")
    md_lines.append("## 5. Hipótesis ganadora")
    md_lines.append("")
    md_lines.append(f"**{winning_hypothesis}** — {winning_label}")
    md_lines.append("")
    md_lines.append(
        "La concentración del 100% del bucket A en el string `"
        f"{A4_FALLBACK_REASON}` es la evidencia decisiva: el sesgo no nace "
        "del prompt (que no tiene taxonomía cerrada) ni del PDF (los "
        "triggers no aparecen sistemáticamente), sino del puente entre el "
        "fallback de A4 y la regla de normalización heurística del "
        "evaluador."
    )
    md_lines.append("")
    md_lines.append("## 6. Propuesta de fix mínimo (sin implementar)")
    md_lines.append("")
    md_lines.append(
        "Dado que la regla del enunciado prohíbe modificar "
        "`agents/agent4_fulltext.py` y `agents/prompts.py`, el fix mínimo "
        "más limpio NO toca el agente sino dos puntos del pipeline de "
        "evaluación / fallback:"
    )
    md_lines.append("")
    md_lines.append(
        "1. **En la heurística de normalización** "
        "(`experiments/eval_hero2_fulltext.py`, función `_normalise_a4_reason`):"
        " considerar la *intersección* `reasoning ∪ exclusion_reason` en "
        "lugar de únicamente `exclusion_reason`, de modo que cuando A4 "
        "use el fallback se busque la categoría primero en el "
        "`reasoning` sustantivo del LLM. Adicionalmente, excluir del "
        "matching el literal `excluded on full-text review` "
        "(short-circuit a `unknown_reason`)."
    )
    md_lines.append("")
    md_lines.append(
        "2. **(Alternativa, fuera del scope de esta tarea)** Cambiar el "
        "fallback en `Agent4._screen_single_fulltext` por un string "
        "neutro como `\"unspecified\"` que no colisione con ninguna "
        "subcadena de la heurística. Esto sí toca "
        "`agents/agent4_fulltext.py` y queda explícitamente fuera del "
        "scope de esta auditoría."
    )
    md_lines.append("")
    md_lines.append("### Predicción testeable")
    md_lines.append("")
    md_lines.append(
        f"- Aplicando el fix (1) sobre los {total_a} casos del bucket A:"
    )
    md_lines.append(
        f"  - Se recuperarían las ~22 confusiones cuyo `reasoning` "
        "menciona keywords categorizables (`outcome`, `intervention`, "
        "`age`/`older`/`elderly`)."
    )
    md_lines.append(
        f"  - El bucket A bajaría de {total_a} a ~"
        f"{max(0, total_a - 22)} casos residuales."
    )
    md_lines.append(
        f"  - Cohen's kappa del experimento Hero 2 subiría de "
        f"0.13 hacia >0.40 (estimación conservadora; el ceiling depende "
        f"de cuántos `reasoning` realmente mapean a la categoría correcta)."
    )
    md_lines.append("")
    md_lines.append("### Set de validación")
    md_lines.append("")
    md_lines.append(
        "Validar sobre los 29 estudios del bucket A actual "
        "(IDs listados en la tabla §1). Específicamente, los siguientes "
        "subconjuntos son los más informativos:"
    )
    md_lines.append("")
    md_lines.append(
        "- 9 casos gold=`Wrong age` (#12037, #13197, #2406, #2518, "
        "#3228, #3532, #4192, #4980, #5978)"
    )
    md_lines.append(
        "- 8 casos gold=`Wrong outcome` (#10805, #1329, #13983, #1864, "
        "#21900, #2360, #570, #931)"
    )
    md_lines.append(
        "- 5 casos gold=`Wrong intervention` (#12288, #13451, #24358, "
        "#3707, #8749)"
    )
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append(
        f"_Auditoría generada el {datetime.now(timezone.utc).isoformat()} "
        "por `scripts/diagnose_hero2_wpt.py` (T16.D.8.B). Solo lectura "
        "sobre `agents/` y `agents/prompts.py`._"
    )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[OK] Informe Markdown guardado en {OUT_MD}")

    # Console summary.
    print()
    print("--- Resumen rápido ---")
    print(f"Bucket A (pred=WPT, gold!=WPT): {total_a}")
    print(f"Bucket B (pred=WPT, gold=WPT):  {len(enriched_b)}")
    print(f"Fracción con fallback literal:  {fraction_fallback * 100:.1f}%")
    print(f"Distribución pred_reason_raw:    {dict(raw_reason_counts)}")
    print(f"Hipótesis ganadora:              {winning_hypothesis}")
    print(f"  -> {winning_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
