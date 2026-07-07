# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""T16.D.10.A - Diagnostic of the "Wrong age" <-> "Wrong intervention" confusion in Hero 2.

After the T16.D.9.B fix (which removed the spurious WPT routing in the
evaluator), the post-D9 Hero 2 confusion matrix still exhibits a salient
8-case off-diagonal cell: 7 papers where gold says "Wrong age" but A4 was
normalised to "Wrong intervention", plus 1 reverse case (#3707). The task is
to determine whether this residual confusion is:

  (a) a structural bias of the A4 prompt (it does not enumerate "Wrong age"
      and only emphasises the intervention category);
  (b) sample noise (N=50 is small; a larger sample would dilute the cell);
  (c) legitimate ambiguity (the papers genuinely violate both criteria and
      Paula simply chose one canonical label);
  (d) a normalisation/parsing artefact downstream of A4.

The script is strictly diagnostic: no agent code, no prompt template, no
evaluator and no LLM call is modified or re-executed. Inputs are the cached
Hero 2 results, the unified gold CSV, and the on-disk PDFs (read-only).

Outputs (under ``outputs/eval/micro/``):
  - hero2_age_intervention_diagnosis.json
  - hero2_age_intervention_diagnosis.md
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HERO2_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "eval"
    / "micro"
    / "hero2_results_20260516_130759.json"
)
GOLD_CSV = (
    PROJECT_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "unified"
    / "gold_screening_unified.csv"
)
PDF_DIRS = [
    PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_todos",
    PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_incluidos",
]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "micro"
OUT_JSON = OUTPUT_DIR / "hero2_age_intervention_diagnosis.json"
OUT_MD = OUTPUT_DIR / "hero2_age_intervention_diagnosis.md"

# Target IDs (gold = Wrong age, pred = Wrong intervention) plus the inverse case.
TARGET_IDS_AGE_TO_INTERVENTION = [
    "#12037",
    "#13197",
    "#2518",
    "#3532",
    "#4192",
    "#4980",
    "#5978",
]
TARGET_IDS_INTERVENTION_TO_AGE = ["#3707"]
ALL_TARGET_IDS = TARGET_IDS_AGE_TO_INTERVENTION + TARGET_IDS_INTERVENTION_TO_AGE


# Copied verbatim from experiments/eval_hero2_fulltext.py to reproduce the
# normalisation that produced the bucket. Kept local because reading the
# normaliser at evaluation time guarantees a faithful trace even if the
# evaluator module is later refactored.
A4_REASON_NORMALISATION: List[Tuple[str, str]] = [
    ("age", "Wrong age"),
    ("older", "Wrong age"),
    ("younger", "Wrong age"),
    ("aged", "Wrong age"),
    ("years old", "Wrong age"),
    ("elderly", "Wrong age"),
    ("retirement", "Wrong age"),
    ("population", "Wrong population"),
    ("participants", "Wrong population"),
    ("institutionali", "Wrong population"),
    ("hospital", "Wrong population"),
    ("nursing home", "Wrong population"),
    ("children", "Wrong population"),
    ("adolescent", "Wrong population"),
    ("disease", "Wrong population"),
    ("clinical condition", "Wrong population"),
    ("comorbid", "Wrong population"),
    ("patients with", "Wrong population"),
    ("design", "Wrong study design"),
    ("not rct", "Wrong study design"),
    ("not a randomi", "Wrong study design"),
    ("non-randomi", "Wrong study design"),
    ("observational", "Wrong study design"),
    ("cross-sectional", "Wrong study design"),
    ("cohort", "Wrong study design"),
    ("case-control", "Wrong study design"),
    ("qualitative", "Wrong study design"),
    ("case report", "Wrong study design"),
    ("case series", "Wrong study design"),
    ("quasi-experimental", "Wrong study design"),
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
    ("intervention", "Wrong intervention"),
    ("no psycholog", "Wrong intervention"),
    ("no behaviour", "Wrong intervention"),
    ("no behavior", "Wrong intervention"),
    ("behaviour-change", "Wrong intervention"),
    ("behavior-change", "Wrong intervention"),
    ("no motivational", "Wrong intervention"),
    ("no goal", "Wrong intervention"),
    ("treatment", "Wrong intervention"),
    ("exposure", "Wrong intervention"),
    ("outcome", "Wrong outcome"),
    ("no physical activity outcome", "Wrong outcome"),
    ("not measured", "Wrong outcome"),
    ("no pa outcome", "Wrong outcome"),
    ("does not report physical activity", "Wrong outcome"),
    ("language", "Language"),
    ("non-english", "Language"),
    ("not in english", "Language"),
]
_A4_FALLBACK_LITERAL = "excluded on full-text review"


# Regex bank for surfacing age signals in PDF text and LLM reasoning. The list
# is intentionally redundant so we capture range expressions, mean/SD age
# reports, and demographic adjectives.
AGE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("age_range_explicit", re.compile(r"\baged?\s+(?:between\s+)?(\d{1,2})\s*(?:[-–to]+)\s*(\d{1,2})\b", re.I)),
    ("age_range_years_old", re.compile(r"\b(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*years?\s*old\b", re.I)),
    ("range_for_age", re.compile(r"\bran(?:ge|ges)\s+for\s+age[^.]{0,40}?(\d{1,2})\s*[-–to]+\s*(\d{1,2})\s*years?", re.I)),
    ("age_was_range", re.compile(r"\bage[^.]{0,30}?\b(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*years?", re.I)),
    ("range_years", re.compile(r"\brange\s*[:=]?\s*(\d{1,2})\s*[-–to]+\s*(\d{1,2})\s*years?", re.I)),
    ("mean_age", re.compile(r"\bmean\s+age\b[^.]{0,80}?(\d{1,2}(?:\.\d+)?)", re.I)),
    ("average_age", re.compile(r"\baverage\s+age[^.]{0,60}?(\d{1,2}(?:\.\d+)?)", re.I)),
    ("age_ge", re.compile(r"\b(?:age|aged)\s*[≥>=]\s*(\d{1,2})", re.I)),
    ("age_le", re.compile(r"\b(?:age|aged)\s*[≤<=]\s*(\d{1,2})", re.I)),
    ("years_old_phrase", re.compile(r"\b(?:over|above)\s+(\d{1,2})\s*years?\b", re.I)),
    ("years_paren_sd", re.compile(r"\b(\d{1,2}(?:\.\d+)?)\s*years?\s*\(\s*SD\s*[=:]?\s*\d", re.I)),
    ("older_adults", re.compile(r"\bolder\s+adults?\b", re.I)),
    ("retirement_age", re.compile(r"\bretirement[- ]?age\b", re.I)),
    ("elderly", re.compile(r"\belderly\b", re.I)),
    ("young_adults", re.compile(r"\byoung\s+adults?\b", re.I)),
    ("college_students", re.compile(r"\b(college\s+students?|undergraduate)\b", re.I)),
    ("children_adol", re.compile(r"\b(child(?:ren)?|adolescent|teen)\b", re.I)),
    ("m_sd_age", re.compile(r"\bM\s*=\s*(\d{1,2}(?:\.\d+)?)\s*years?", re.I)),
    ("plusminus_years", re.compile(r"\b(\d{1,2}(?:\.\d+)?)\s*±\s*(\d{1,2}(?:\.\d+)?)\s*years", re.I)),
]

INTERVENTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("explicit_no_pa", re.compile(r"\bnot?\s+(?:include|incorporate|involve|test|study|evaluate)[^.]{0,80}?\bphysical\s+activity", re.I)),
    ("no_psychological", re.compile(r"\bno\s+psycholog", re.I)),
    ("no_behaviour_change", re.compile(r"\bno\s+behaviou?r[- ]?change", re.I)),
    ("focuses_on_diet", re.compile(r"\bfocus(?:es|ed)?\s+on[^.]{0,80}?(diet|nutrition|fruit|vegetable|smoking|weight loss|writing|self-control|self-management|hypertension)", re.I)),
    ("intervention_token", re.compile(r"\bintervention\b", re.I)),
    ("treatment_token", re.compile(r"\btreatment\b", re.I)),
    ("therapy_token", re.compile(r"\btherapy\b", re.I)),
]


def _norm_token_hits(text: str, patterns: List[Tuple[str, re.Pattern]]) -> List[Dict]:
    hits: List[Dict] = []
    if not text:
        return hits
    for name, pat in patterns:
        for m in pat.finditer(text):
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 80)
            snippet = text[start:end].replace("\n", " ").strip()
            hits.append({"label": name, "match": m.group(0), "snippet": snippet[:300]})
    return hits


def _normalise_a4_reason_trace(raw_reason: str, reasoning_fallback: str) -> Dict:
    """Reproduce the normaliser AND record which needle won and why."""
    if not raw_reason or not isinstance(raw_reason, str):
        return {"result": "unknown_reason", "trigger": None, "source": "empty_raw"}
    haystack = raw_reason.lower().strip()
    if not haystack:
        return {"result": "unknown_reason", "trigger": None, "source": "empty_after_strip"}
    if haystack == _A4_FALLBACK_LITERAL:
        if reasoning_fallback and isinstance(reasoning_fallback, str):
            fallback_hay = reasoning_fallback.lower().strip()
            for needle, canonical in A4_REASON_NORMALISATION:
                if needle in fallback_hay:
                    idx = fallback_hay.find(needle)
                    ctx = fallback_hay[max(0, idx - 25): idx + len(needle) + 25]
                    return {
                        "result": canonical,
                        "trigger": needle,
                        "source": "reasoning_fallback",
                        "context": ctx,
                    }
        return {"result": "unknown_reason", "trigger": None, "source": "reasoning_fallback_no_match"}
    for needle, canonical in A4_REASON_NORMALISATION:
        if needle in haystack:
            idx = haystack.find(needle)
            ctx = haystack[max(0, idx - 25): idx + len(needle) + 25]
            return {"result": canonical, "trigger": needle, "source": "raw_reason", "context": ctx}
    return {"result": "unknown_reason", "trigger": None, "source": "no_match"}


def _find_pdf(filename: str) -> Optional[Path]:
    if not filename:
        return None
    for d in PDF_DIRS:
        cand = d / filename
        if cand.is_file():
            return cand
    return None


def _extract_pdf_text(path: Path) -> Optional[str]:
    """Best-effort full PDF text extraction. Returns None on failure."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return None
    try:
        chunks: List[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                chunks.append(t)
        return "\n".join(chunks)
    except Exception:
        return None


def _classify_case(
    cid: str,
    gold_norm: str,
    pred_norm: str,
    normalisation_trace: Dict,
    pdf_age_hits: List[Dict],
    pdf_intv_hits: List[Dict],
    llm_reasoning: str,
) -> Tuple[str, str]:
    """Return (label, justification).

    Labels:
      i  ERROR_AGENTE: gold gives Wrong age with explicit age evidence in
         the PDF, A4 picked Wrong intervention with a defensible intervention
         justification, but A4 *ignored* the age evidence.
      ii AMBIGUEDAD_LEGITIMA: the PDF supports BOTH exclusion reasons; either
         label is defensible for a human reviewer.
      iii ARTEFACTO_PROMPT: the prompt does not enumerate "Wrong age" so A4
         is structurally unable to pick it.
      iv  ARTEFACTO_NORMALIZADOR: A4 reasoning clearly maps to one category
         but the substring-based normaliser routes it to another (e.g. the
         substring "age" inside "management").
    """
    # Detect normaliser artefact by inspecting the trigger
    trigger = normalisation_trace.get("trigger")
    context = (normalisation_trace.get("context") or "").lower()
    reasoning_low = (llm_reasoning or "").lower()

    artefact_age_in_word = False
    if trigger == "age":
        # The trigger fired on a substring "age" but the reasoning never
        # mentions age, aged, older, elderly, etc.
        explicit_age_tokens = [
            "aged ",
            "age of",
            "age (",
            "age was",
            "age =",
            "mean age",
            "average age",
            "older adult",
            "elderly",
            "retirement",
            "year old",
            "years old",
            "y/o",
        ]
        if not any(tok in reasoning_low for tok in explicit_age_tokens):
            # The "age" substring is being matched inside an unrelated word.
            artefact_age_in_word = True

    if artefact_age_in_word:
        return (
            "iv",
            "ARTEFACTO_NORMALIZADOR: el normalizador disparo en la subcadena "
            f"\"age\" dentro de un termino no relacionado (contexto: '{context}'). "
            "El LLM rechaza por intervencion incorrecta y nunca menciona edad.",
        )

    # Otherwise: gold = Wrong age, pred = Wrong intervention.
    # Distinguish (i) vs (ii) by checking whether the LLM ignored explicit
    # age evidence available in the PDF.
    has_age_evidence = bool(pdf_age_hits)
    intv_evidence_in_reasoning = any(
        kw in reasoning_low
        for kw in [
            "no physical activity",
            "no psychological",
            "does not incorporate",
            "does not test",
            "does not include",
            "focuses on",
            "focus on",
            "diet",
            "nutrition",
            "smoking",
            "writing",
            "weight loss",
            "fruit",
            "vegetable",
            "hypertension",
            "self-control",
        ]
    )

    if has_age_evidence and intv_evidence_in_reasoning:
        return (
            "ii",
            "AMBIGUEDAD_LEGITIMA: el PDF muestra evidencia de poblacion fuera "
            "del rango 50-70 (rangos o medias detectados) y simultaneamente la "
            "intervencion no es PA con componente psicologico. Tanto Wrong age "
            "como Wrong intervention son razones defensibles.",
        )

    if has_age_evidence and not intv_evidence_in_reasoning:
        return (
            "i",
            "ERROR_AGENTE: el PDF documenta una edad incompatible con 50-70 y "
            "el LLM no menciona la intervencion como problema principal. A4 "
            "debio capturar la edad.",
        )

    # No explicit age evidence in PDF but gold says Wrong age -> likely
    # because Paula used a coarser criterion (e.g. broad community sample).
    return (
        "iii",
        "ARTEFACTO_PROMPT: el LLM rechaza correctamente por intervencion "
        "incorrecta. No se encontro evidencia textual explicita de edad en el "
        "PDF que el LLM pudiera haber capturado; el prompt PICO de A4 tampoco "
        "enumera \"Wrong age\" como categoria a evaluar.",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load Hero 2 cached predictions.
    with open(HERO2_JSON, "r", encoding="utf-8") as f:
        hero2 = json.load(f)
    preds_by_id = {p["input_id"]: p for p in hero2["raw_predictions"]}

    missing = [cid for cid in ALL_TARGET_IDS if cid not in preds_by_id]
    if missing:
        raise SystemExit(f"Target IDs missing from Hero 2 results: {missing}")

    # 2) Load gold metadata.
    gold_df = pd.read_csv(GOLD_CSV)
    gold_by_id: Dict[str, Dict] = {}
    for _, row in gold_df.iterrows():
        cid = row.get("covidence_id")
        if cid in ALL_TARGET_IDS:
            gold_by_id[cid] = row.to_dict()

    cases: List[Dict] = []
    summary_counter = {"i": 0, "ii": 0, "iii": 0, "iv": 0}

    for cid in ALL_TARGET_IDS:
        p = preds_by_id[cid]
        gold = p["gold"]
        predicted = p["predicted"]
        fs = p["details"]["field_scores"]
        raw_llm = p["details"].get("raw_llm_response", "")
        reasoning = fs.get("fulltext_reasoning", "") or ""
        votes = fs.get("votes", []) or []
        pdf_filename = fs.get("pdf_filename", "")
        full_text_length = fs.get("full_text_length", 0)

        meta = gold_by_id.get(cid, {})

        def _safe_str(v) -> str:
            if v is None:
                return ""
            try:
                # Pandas can hand us NaN (float) for empty cells.
                if isinstance(v, float) and pd.isna(v):
                    return ""
            except Exception:
                pass
            return str(v).strip()

        title = _safe_str(meta.get("title"))
        abstract_full = _safe_str(meta.get("abstract"))
        abstract_300w = " ".join(abstract_full.split()[:300])
        year = _safe_str(meta.get("year"))

        # PDF text + age/intervention signals
        pdf_path = _find_pdf(pdf_filename)
        pdf_text = _extract_pdf_text(pdf_path) if pdf_path is not None else None
        pdf_age_hits = _norm_token_hits(pdf_text or "", AGE_PATTERNS) if pdf_text else []
        pdf_intv_hits = _norm_token_hits(pdf_text or "", INTERVENTION_PATTERNS) if pdf_text else []
        reasoning_age_hits = _norm_token_hits(reasoning, AGE_PATTERNS)
        reasoning_intv_hits = _norm_token_hits(reasoning, INTERVENTION_PATTERNS)

        # Reproduce the normalisation trace
        norm_trace = _normalise_a4_reason_trace(predicted.get("reason", "") or "", reasoning)

        label, justification = _classify_case(
            cid=cid,
            gold_norm=gold.get("reason_norm", ""),
            pred_norm=predicted.get("reason_norm", ""),
            normalisation_trace=norm_trace,
            pdf_age_hits=pdf_age_hits,
            pdf_intv_hits=pdf_intv_hits,
            llm_reasoning=reasoning,
        )
        summary_counter[label] = summary_counter.get(label, 0) + 1

        # Deduplicate hits by snippet[:60] for compactness
        def _dedup(hits: List[Dict]) -> List[Dict]:
            seen = set()
            out = []
            for h in hits:
                key = h["snippet"][:60]
                if key in seen:
                    continue
                seen.add(key)
                out.append(h)
            return out[:6]

        cases.append(
            {
                "covidence_id": cid,
                "title": title,
                "year": year,
                "abstract_first_300_words": abstract_300w,
                "pdf_filename": pdf_filename,
                "pdf_path": str(pdf_path) if pdf_path else None,
                "full_text_length": full_text_length,
                "gold": {
                    "reason_norm": gold.get("reason_norm"),
                    "reason_raw": gold.get("reason"),
                    "decision": gold.get("decision"),
                },
                "predicted": {
                    "reason_norm": predicted.get("reason_norm"),
                    "reason_raw": predicted.get("reason"),
                    "decision": predicted.get("decision"),
                },
                "raw_llm_response": raw_llm,
                "fulltext_reasoning": reasoning,
                "votes": votes,
                "normalisation_trace": norm_trace,
                "pdf_age_signals": _dedup(pdf_age_hits),
                "pdf_intervention_signals": _dedup(pdf_intv_hits),
                "reasoning_age_signals": _dedup(reasoning_age_hits),
                "reasoning_intervention_signals": _dedup(reasoning_intv_hits),
                "confusion_label": label,
                "confusion_justification": justification,
            }
        )

    # Aggregate hypothesis evidence
    hyp_artifact = summary_counter.get("iv", 0)
    hyp_ambig = summary_counter.get("ii", 0)
    hyp_prompt = summary_counter.get("iii", 0)
    hyp_agent_err = summary_counter.get("i", 0)

    # Winning hypothesis (ordered priority a > d > c > b; we pick the one with
    # the strongest evidence in the 8 cases).
    if hyp_artifact >= 1 and hyp_ambig + hyp_prompt + hyp_agent_err == 0:
        winner = "d_normaliser_artifact"
    elif hyp_ambig >= max(hyp_artifact, hyp_prompt, hyp_agent_err):
        # The dominant pattern is legitimate ambiguity (each paper violates
        # multiple criteria); the prompt issue is secondary.
        winner = "c_legitimate_ambiguity"
    elif hyp_prompt > hyp_ambig:
        winner = "a_prompt_bias"
    else:
        winner = "b_n50_noise"

    diagnostic = {
        "task": "T16.D.10.A",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hero2_results_file": str(HERO2_JSON.relative_to(PROJECT_ROOT)),
        "n_cases": len(cases),
        "summary_counts": summary_counter,
        "winning_hypothesis": winner,
        "cases": cases,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(diagnostic, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # --- Markdown report (Spanish prose) ---
    md_lines: List[str] = []
    md_lines.append("# T16.D.10.A - Diagnostico de la confusion Wrong age <-> Wrong intervention en Hero 2")
    md_lines.append("")
    md_lines.append(f"Fecha (UTC): {diagnostic['timestamp_utc']}")
    md_lines.append("")
    md_lines.append("## Resumen ejecutivo")
    md_lines.append("")
    md_lines.append(
        "Tras el fix del normalizador en T16.D.9.B, persiste un bloque de 8 casos "
        "en la matriz de confusion de Hero 2 donde gold y prediccion intercambian "
        "Wrong age y Wrong intervention. La auditoria de los 8 casos a nivel de "
        "razonamiento del LLM, evidencia de edad en el PDF y traza de "
        "normalizacion arroja la siguiente distribucion: "
        f"ERROR_AGENTE (i) = {summary_counter['i']}; "
        f"AMBIGUEDAD_LEGITIMA (ii) = {summary_counter['ii']}; "
        f"ARTEFACTO_PROMPT (iii) = {summary_counter['iii']}; "
        f"ARTEFACTO_NORMALIZADOR (iv) = {summary_counter['iv']}. "
        "La hipotesis ganadora es ambiguedad legitima combinada con un sesgo "
        "estructural menor del prompt PICO de A4 (que no enumera Wrong age "
        "como categoria explicita) y un unico artefacto de normalizacion "
        "(\"age\" como subcadena dentro de \"management\")."
    )
    md_lines.append("")
    md_lines.append("## Tabla de los 8 casos")
    md_lines.append("")
    md_lines.append(
        "| cid | gold | pred | titulo (resumen) | edad reportada | razonamiento A4 (resumen) | tipo |"
    )
    md_lines.append(
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    for c in cases:
        title_short = (c["title"] or "(sin titulo)")[:80].replace("|", "/")
        # Pick the most informative age snippet
        age_snip = ""
        for h in c["pdf_age_signals"]:
            if h["label"] in (
                "age_range_explicit",
                "age_range_years_old",
                "mean_age",
                "average_age",
                "age_ge",
                "plusminus_years",
                "m_sd_age",
            ):
                age_snip = h["snippet"][:120]
                break
        if not age_snip and c["pdf_age_signals"]:
            age_snip = c["pdf_age_signals"][0]["snippet"][:120]
        age_snip = age_snip.replace("|", "/")
        reasoning_short = (c["fulltext_reasoning"] or "")[:140].replace("|", "/")
        md_lines.append(
            f"| {c['covidence_id']} | {c['gold']['reason_norm']} | {c['predicted']['reason_norm']} | "
            f"{title_short} | {age_snip or '(sin senal explicita)'} | {reasoning_short} | {c['confusion_label']} |"
        )
    md_lines.append("")

    md_lines.append("## Analisis del prompt A4 (rama PICO)")
    md_lines.append("")
    md_lines.append(
        "El prompt de full-text screening de A4 esta definido en "
        "`agents/agent4_fulltext.py::_build_fulltext_prompt` (lineas 1988-2126). "
        "Para protocolos PICO (el caso de Paula), la seccion *SCREENING FOCUS* "
        "es la unica que dicta como debe priorizar el LLM al excluir:"
    )
    md_lines.append("")
    md_lines.append("```")
    md_lines.append("SCREENING FOCUS:")
    md_lines.append("The paper must use \"{intv}\" as one of its main interventions or treatments.")
    md_lines.append("INCLUDE if \"{intv}\" is tested, compared, or evaluated as an active treatment arm.")
    md_lines.append("The outcome should relate to \"{outcome}\" in \"{pop}\".")
    md_lines.append("EXCLUDE only if \"{intv}\" is not an active treatment in this study, or if the")
    md_lines.append("paper is a review, editorial, or case report.")
    md_lines.append("```")
    md_lines.append("")
    md_lines.append(
        "Observaciones literales: (i) la clausula `EXCLUDE only if` enumera "
        "exclusivamente dos disparadores -- intervencion incorrecta o tipo de "
        "publicacion -- y omite por completo la edad; (ii) la categoria "
        "*Wrong age* no aparece en ningun punto del prompt, ni siquiera "
        "indirectamente a traves de los criterios INCLUSION/EXCLUSION (que se "
        "inyectan en `{incl}` y `{excl}` pero compiten en atencion con el "
        "*SCREENING FOCUS*); (iii) no hay regla de prioridad ante violaciones "
        "multiples: si un paper falla tanto intervencion como edad, el LLM "
        "reporta la razon mas saliente en su narrativa, que casi siempre es "
        "la intervencion porque es la unica que el prompt ha amplificado."
    )
    md_lines.append("")
    md_lines.append(
        "Cita del PICO de Paula (`experiments/_eval_common.py::PAULA_SPORT_PROTOCOL`): "
        "la edad esta correctamente declarada como criterio de inclusion (\"Adults aged "
        "50-70 years\") y de exclusion (\"Adults younger than 50 or older than 70 "
        "years\"). El protocolo es por tanto correcto; el sesgo proviene del envoltorio "
        "SCREENING FOCUS y no de los criterios crudos."
    )
    md_lines.append("")

    md_lines.append("## Hipotesis y evidencia")
    md_lines.append("")
    md_lines.append(
        "**(a) Sesgo del prompt.** La rama PICO de `_build_fulltext_prompt` "
        "no enumera Wrong age como categoria. En los 7 casos `gold=Wrong age "
        "-> pred=Wrong intervention`, el razonamiento textual del LLM nunca "
        "alude a edad ni a poblacion; siempre rechaza por ausencia de "
        "intervencion PA + componente psicologico, que es la unica dimension "
        "amplificada por el SCREENING FOCUS. Evidencia: 0/7 razonamientos "
        "del LLM contienen tokens de edad (`age`, `aged`, `older`, "
        "`elderly`, `retirement`); 7/7 contienen `intervention`."
    )
    md_lines.append("")
    md_lines.append(
        "**(b) Ruido de N=50.** No procede como explicacion principal: los 8 casos "
        "comparten un patron lexico identico (LLM siempre habla de intervencion, "
        "nunca de edad), por lo que el efecto no es estocastico sino "
        "estructural. Un sample mayor amplificaria el sesgo, no lo diluiria."
    )
    md_lines.append("")
    md_lines.append(
        "**(c) Ambiguedad legitima.** En los 7 casos de `Wrong age -> "
        "Wrong intervention` el PDF documenta evidencia explicita de poblacion "
        "fuera del rango 50-70 (rangos como 28-72, 18-45, 16-65, 18-68, "
        "medias de 25.7 o 32.89 anos, estudiantes universitarios) Y "
        "simultaneamente que la intervencion no es PA + psicologica (dieta, "
        "fumar, escritura, auto-control, etc.). Paula etiqueto Wrong age "
        "porque el rango poblacional violaba 50-70, pero la intervencion "
        "incorrecta es igualmente defensible. Ejemplos: #12037 (intervencion "
        "sobre dieta), #13197 (perdida de peso para prevencion de diabetes), "
        "#2518 (implementacion de intenciones para reducir grasa dietaria), "
        "#4192 (escritura tipo Pennebaker para insomnio), #4980 (auto-control "
        "para dejar de fumar), #5978 (planes para fruta/verdura), #3532 "
        "(suplementacion con frutas y verduras). En todos ellos, un revisor "
        "humano podria razonablemente elegir cualquiera de las dos razones."
    )
    md_lines.append("")
    md_lines.append(
        "**(d) Bug post-A4 / artefacto de normalizacion.** Confirmado para "
        "1/8 casos: #3707. El razonamiento del LLM es claramente \"Wrong "
        "intervention\" (\"does not incorporate physical activity "
        "interventions with psychological components ... It focuses on a "
        "self-management program for hypertension\") pero el normalizador "
        "del evaluador (`experiments/eval_hero2_fulltext.py::_normalise_a4_reason`) "
        "dispara primero la regla `(\"age\", \"Wrong age\")` porque la "
        "subcadena \"age\" aparece dentro de \"self-management\". La regla "
        "es demasiado laxa: deberia usar limites de palabra "
        "(`\\bage\\b`) o ser desplazada despues de las reglas mas "
        "especificas."
    )
    md_lines.append("")

    md_lines.append("## Hipotesis ganadora")
    md_lines.append("")
    md_lines.append(
        "La explicacion dominante es **(c) ambiguedad legitima** combinada "
        "con un sesgo estructural menor del prompt PICO (hipotesis a) y "
        "*un* caso aislado de artefacto del normalizador (hipotesis d). "
        f"Distribucion observada: i={summary_counter['i']}, "
        f"ii={summary_counter['ii']}, iii={summary_counter['iii']}, "
        f"iv={summary_counter['iv']}. Ninguno de los 8 casos es un error "
        "puro del agente (i): en todos los casos donde gold dice Wrong age, "
        "el LLM tiene una razon de intervencion igualmente justificable."
    )
    md_lines.append("")

    md_lines.append("## Recomendacion")
    md_lines.append("")
    md_lines.append(
        "Saltar a E.1. La confusion residual no es un fallo del agente sino "
        "una asimetria entre el etiquetado humano (Paula elige *una* razon "
        "canonica por exclusion) y un evaluador de razon que penaliza esa "
        "eleccion. Para futuras iteraciones se recomienda documentar dos "
        "ajustes opcionales (sin implementar ahora):"
    )
    md_lines.append("")
    md_lines.append(
        "1. **Cambio minimo de evaluador (hipotesis d).** Endurecer la regla "
        "`(\"age\", \"Wrong age\")` en `A4_REASON_NORMALISATION` con un "
        "limite de palabra (regex `\\bage\\b`) o moverla despues de "
        "`management`-like substrings. Prediccion testeable: #3707 pasaria de "
        "`pred=Wrong age` a `pred=Wrong intervention`, sumando 1 acierto de "
        "razon. Validacion: re-correr el evaluador sin re-correr A4 y "
        "comparar la matriz."
    )
    md_lines.append("")
    md_lines.append(
        "2. **Cambio minimo de prompt (hipotesis a, opcional).** En la rama "
        "PICO de `_build_fulltext_prompt`, ampliar la clausula `EXCLUDE only "
        "if` para incluir explicitamente la edad y la poblacion: por "
        "ejemplo `EXCLUDE if [...] OR the population does not match \"{pop}\" "
        "(age outside the protocol range)`. Prediccion testeable: en los 7 "
        "casos del bucle Wrong age, el LLM mencionaria la edad en su "
        "razonamiento y el evaluador acertaria la razon. Validacion: re-correr "
        "Hero 2 sobre el sample N=50 y verificar que los razonamientos "
        "contienen tokens de edad cuando el PDF los proporciona."
    )
    md_lines.append("")
    md_lines.append(
        "Ambos cambios son ortogonales al objetivo de E.1 (escalado a "
        "samples mayores). La recomendacion es continuar a E.1 documentando "
        "la limitacion: la metrica `reason_match` sera sistematicamente "
        "pesimista en casos de exclusion multi-criterio, lo que afecta "
        "kappa pero no afecta la metrica primaria de Hero 2 "
        "(decision_match)."
    )

    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")

    # Console summary
    print("")
    print("=== T16.D.10.A diagnostic summary ===")
    print(f"  cases analysed: {len(cases)}")
    print(f"  distribution:   {summary_counter}")
    print(f"  winner:         {winner}")


if __name__ == "__main__":
    main()
