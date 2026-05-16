"""
Hero 2 evaluator -- Agent4 Full-Text screening vs Paula's gold standard.

Sprint:  T16.D.5.C
Schema:  docs/heroes_output_schema.md (section 3.2)
Inputs:
  - config/eval/micro.yaml          : paths to sample, gold, PDFs
  - data/gold_standard/deporte/samples/micro/sample_50.csv
  - data/gold_standard/deporte/unified/gold_screening_unified.csv
  - data/gold_standard/deporte/pdfs_todos/*.pdf
  - config/agents.json              : per-agent LLM params (agent4_validation)

Output:
  outputs/eval/micro/hero2_results_{YYYYMMDD_HHMMSS}.json
    + short stdout report (Spanish)

Pipeline overview:
  1. Load sample_50.csv, drop rows with has_pdf=False (registered in errors[]).
  2. Join with gold_screening_unified.csv on covidence_id to get title+abstract.
  3. For each row, extract local PDF text (PyMuPDF -> pdfplumber fallback, same
     backends Agent4 uses internally). PDFs that fail extraction are skipped
     with an entry in errors[].
  4. Build paper dicts with full_text + text_source='local_pdf' so we bypass
     Phase A (HTTP cascade) entirely and feed Agent4.screen_fulltext directly.
  5. Call agent4.screen_fulltext(papers, protocol) once.
  6. Compute include/exclude binary metrics (accuracy, sensitivity, specificity,
     F1, Cohen's kappa) and the 8x8 reason-vs-reason confusion matrix.

Agent4 exclusion-reason taxonomy
--------------------------------
Agent4 does NOT use a closed enumeration. Its prompt
(_build_fulltext_prompt, line ~2126 of agents/agent4_fulltext.py) instructs the
LLM to emit a free-text "brief reason if EXCLUDE". In practice the LLM tends to
produce short phrases such as "wrong population", "not RCT", "review article",
"no behaviour-change component", etc. We map these to the seven canonical
categories used by Paula's gold standard (column exclusion_reason_ft):

  - Wrong age              (population aged <50 or >70)
  - Wrong outcome          (no PA outcome reported)
  - Wrong study design     (not RCT: observational, qualitative, case report)
  - Wrong intervention     (no psychological / behaviour-change component)
  - Wrong publication type (review, editorial, protocol, conference abstract)
  - Wrong population       (institutionalised, major chronic disease, children)
  - Language               (non-English where excluded)
  - Sin motivo registrado  (gold reason absent; treated as a residual bucket)

Unknown / unmappable A4 reasons are tracked under "unknown_reason" so the
stratified_matrix_8x8 keeps a complete tally. Includes are tracked under
"include" so the matrix is square (include + 7 gold reasons + unknown).

Notes:
  - The script does NOT modify agents/agent4_fulltext.py, base.py, config.yaml,
    or prompts.py (per task scope).
  - max_input_chars (Agent4) defaults to 6000 chars; the full local PDF is
    passed but Agent4 truncates internally in _prepare_text_for_prompt.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so `agents` and `experiments` packages
# resolve when launching as `python experiments/eval_hero2_fulltext.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._eval_common import (  # noqa: E402
    PAULA_SPORT_PROTOCOL,
    extract_pdf_text as _extract_pdf_text_shared,
    gpu_info as _gpu_info,
    load_agent_params,
    load_eval_config,
    seed_torch as _seed_torch,
    truncate as _truncate,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "eval" / "micro.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "micro"
HERO_ID = "hero2"
AGENT_TAG = "A4"

# Canonical gold exclusion categories (column exclusion_reason_ft in
# sample_50.csv). Order matters for the 8x8 stratified matrix output.
GOLD_REASON_CATEGORIES = [
    "Wrong age",
    "Wrong outcome",
    "Wrong study design",
    "Wrong intervention",
    "Wrong publication type",
    "Wrong population",
    "Language",
    "Sin motivo registrado",
]

# Labels for the stratified matrix rows/columns. Always include the "include"
# decision plus an "unknown_reason" bucket for A4 outputs we cannot map.
MATRIX_LABELS = ["include"] + GOLD_REASON_CATEGORIES + ["unknown_reason"]


# Normalisation map: substring (case-insensitive) on A4 reason string ->
# canonical gold category. Ordering matters: the first matching substring wins.
# This is a *heuristic* map: A4's reason field is free-form English so we tag
# the most discriminative keywords. Anything not matched falls into
# "unknown_reason".
A4_REASON_NORMALISATION = [
    # Wrong age (age criterion)
    ("age", "Wrong age"),
    ("older", "Wrong age"),
    ("younger", "Wrong age"),
    ("aged", "Wrong age"),
    ("years old", "Wrong age"),
    ("elderly", "Wrong age"),
    ("retirement", "Wrong age"),
    # Wrong population (clinical/setting issue)
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
    # Wrong study design
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
    # Wrong publication type (must come AFTER design check)
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
    # Wrong intervention (psychological / behaviour-change component absent)
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
    # Wrong outcome
    ("outcome", "Wrong outcome"),
    ("no physical activity outcome", "Wrong outcome"),
    ("not measured", "Wrong outcome"),
    ("no pa outcome", "Wrong outcome"),
    ("does not report physical activity", "Wrong outcome"),
    # Language
    ("language", "Language"),
    ("non-english", "Language"),
    ("not in english", "Language"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_agent4_params() -> dict:
    """Load agent4_validation params from config/agents.json (defaults merged)."""
    return load_agent_params("agent4_validation")


def _extract_pdf_text_from_path(pdf_path: Path) -> tuple[str | None, str | None]:
    """Extract text from a local PDF (thin wrapper around the shared helper).

    Preserved as a private name for backwards compatibility within this module;
    the implementation now lives in ``experiments._eval_common.extract_pdf_text``
    so hero2 / hero3 / hero4 share a single extraction routine.
    """
    return _extract_pdf_text_shared(pdf_path)


# Fallback literal injected by Agent4._screen_single_fulltext when the LLM
# returns EXCLUDE with exclusion_reason="". Matching this literal as substring
# would route every fallback to "Wrong publication type" via the "review"
# needle, which was the root cause of the WPT bias documented in T16.D.8.B.
_A4_FALLBACK_LITERAL = "excluded on full-text review"


def _normalise_a4_reason(raw_reason: str, reasoning_fallback: str = "") -> str:
    """Map a free-text A4 exclusion reason to the canonical gold category.

    Returns one of GOLD_REASON_CATEGORIES, or "unknown_reason" if no keyword
    matched. Matching is case-insensitive substring search; the *first*
    matching entry of A4_REASON_NORMALISATION wins.

    If raw_reason equals the A4 fallback literal "excluded on full-text review"
    (which contains the substring "review" and would otherwise collide with
    Wrong publication type), normalisation falls back to reasoning_fallback —
    the substantive justification text the LLM emits in the `reasoning` field.
    See T16.D.8.B diagnosis and T16.D.9.B fix.
    """
    if not raw_reason or not isinstance(raw_reason, str):
        return "unknown_reason"
    haystack = raw_reason.lower().strip()
    if not haystack:
        return "unknown_reason"
    if haystack == _A4_FALLBACK_LITERAL:
        if reasoning_fallback and isinstance(reasoning_fallback, str):
            fallback_hay = reasoning_fallback.lower().strip()
            for needle, canonical in A4_REASON_NORMALISATION:
                if needle in fallback_hay:
                    return canonical
        return "unknown_reason"
    for needle, canonical in A4_REASON_NORMALISATION:
        if needle in haystack:
            return canonical
    return "unknown_reason"


def _normalise_gold_reason(raw_reason) -> str:
    """Map a gold exclusion_reason_ft cell value to a canonical category label."""
    if raw_reason is None:
        return "Sin motivo registrado"
    s = str(raw_reason).strip()
    if not s or s.lower() in ("nan", "none"):
        return "Sin motivo registrado"
    # Exact match against canonical names
    for cat in GOLD_REASON_CATEGORIES:
        if s.lower() == cat.lower():
            return cat
    # Fallback: case-insensitive substring matching against canonical names
    low = s.lower()
    for cat in GOLD_REASON_CATEGORIES:
        if cat.lower() in low or low in cat.lower():
            return cat
    return "Sin motivo registrado"


def _build_papers(
    sample_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    pdfs_dir: Path,
    errors: list,
) -> tuple[list[dict], list[dict]]:
    """Build paper dicts ready for Agent4.screen_fulltext.

    Returns:
        (papers, skipped) — `papers` is the list of dicts to feed to A4,
        each populated with `full_text`, `full_text_length`, `text_source`,
        `title`, `abstract`, `pmid` (= covidence_id), plus internal metadata
        used by the evaluator to compute metrics (`_gold_decision`,
        `_gold_reason_raw`, `_gold_reason_norm`, `_pdf_filename`).
        `skipped` is the list of {covidence_id, reason} entries for rows that
        could not be turned into a usable paper (no PDF, extraction failure).
    """
    gold_idx = gold_df.set_index("covidence_id")
    papers: list[dict] = []
    skipped: list[dict] = []

    for _, row in sample_df.iterrows():
        cid = str(row["covidence_id"])
        has_pdf = bool(row.get("has_pdf", False))
        pdf_filename = row.get("pdf_filename")
        gold_decision = str(row.get("human_label_ft", "")).strip().lower()
        gold_reason_raw = row.get("exclusion_reason_ft")
        gold_reason_norm = (
            "include"
            if gold_decision == "include"
            else _normalise_gold_reason(gold_reason_raw)
        )

        # Row 1: no PDF available -> skip + error log
        if not has_pdf or pdf_filename is None or (
            isinstance(pdf_filename, float) and pd.isna(pdf_filename)
        ):
            errors.append(
                {
                    "input_id": cid,
                    "error_type": "no_pdf_available",
                    "message": (
                        f"Sample row marked has_pdf=False (pdf_filename={pdf_filename!r}); "
                        "skipped from full-text evaluation."
                    ),
                }
            )
            skipped.append({"covidence_id": cid, "reason": "no_pdf_available"})
            continue

        pdf_path = pdfs_dir / str(pdf_filename)
        text, err = _extract_pdf_text_from_path(pdf_path)
        if text is None:
            errors.append(
                {
                    "input_id": cid,
                    "error_type": "pdf_extraction_failed",
                    "message": f"PDF '{pdf_filename}' extraction failed: {err}",
                }
            )
            skipped.append({"covidence_id": cid, "reason": "pdf_extraction_failed"})
            continue

        # Pull title / abstract from gold join
        title = ""
        abstract = ""
        if cid in gold_idx.index:
            gold_row = gold_idx.loc[cid]
            if isinstance(gold_row, pd.DataFrame):
                gold_row = gold_row.iloc[0]
            title = str(gold_row.get("title", "") or "").strip()
            abstract = str(gold_row.get("abstract", "") or "").strip()
            if abstract.lower() in ("nan", "none"):
                abstract = ""
            if title.lower() in ("nan", "none"):
                title = ""

        papers.append(
            {
                "pmid": cid,                       # Use covidence_id as identifier
                "title": title,
                "abstract": abstract,
                "full_text": text,
                "full_text_length": len(text),
                "text_source": "local_pdf",        # synthetic source tag
                "year": "",
                "journal": "",
                # Internal evaluator metadata (not consumed by Agent4):
                "_gold_decision": gold_decision,
                "_gold_reason_raw": (
                    str(gold_reason_raw)
                    if gold_reason_raw is not None and not (
                        isinstance(gold_reason_raw, float) and pd.isna(gold_reason_raw)
                    )
                    else ""
                ),
                "_gold_reason_norm": gold_reason_norm,
                "_pdf_filename": str(pdf_filename),
            }
        )

    return papers, skipped


def _compute_binary_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> tuple[dict, dict]:
    """Compute accuracy, sensitivity, specificity, F1 and the 2x2 matrix.

    Both lists encode: 1 = include, 0 = exclude. Returns
    (metrics_dict, confusion_matrix_dict).
    """
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        return (
            {
                "accuracy": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "n": 0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            },
            {
                "include": {"include": 0, "exclude": 0},
                "exclude": {"include": 0, "exclude": 0},
            },
        )

    tp = fn = fp = tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    if precision + sensitivity > 0:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    else:
        f1 = 0.0

    metrics = {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    confusion_matrix = {
        "include": {"include": tp, "exclude": fn},
        "exclude": {"include": fp, "exclude": tn},
    }
    return metrics, confusion_matrix


def _build_stratified_matrix(
    pairs: list[tuple[str, str]],
    extra_labels: list[str] | None = None,
) -> tuple[dict, list[str]]:
    """Build the 8x8 stratified matrix of gold_reason vs predicted_reason.

    Args:
        pairs: list of (gold_label, predicted_label) tuples. Labels are
            "include" or one of GOLD_REASON_CATEGORIES, or "unknown_reason".
        extra_labels: optional extra labels to include as columns/rows even
            if absent from `pairs` (useful for unmapped A4 reasons).

    Returns:
        (matrix_dict, ordered_labels). Each `matrix_dict[gold][pred]` is a
        count. Labels are ordered as MATRIX_LABELS plus any extras.
    """
    labels = list(MATRIX_LABELS)
    if extra_labels:
        for lab in extra_labels:
            if lab not in labels:
                labels.append(lab)

    # Ensure every label observed in data is included
    for gold, pred in pairs:
        for lab in (gold, pred):
            if lab not in labels:
                labels.append(lab)

    matrix = {g: {p: 0 for p in labels} for g in labels}
    for gold, pred in pairs:
        matrix[gold][pred] += 1
    return matrix, labels


def _cohens_kappa(y_true: list, y_pred: list) -> float | None:
    """Compute Cohen's kappa via sklearn. Returns None if sklearn unavailable
    or inputs are empty / degenerate."""
    if not y_true or len(y_true) != len(y_pred):
        return None
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return None
    try:
        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def run_hero2_evaluation(
    config_path: Path,
    output_dir: Path,
    agent_instance=None,
) -> dict:
    """Run the Hero 2 evaluation and return the result dict (also written to disk).

    Args:
        config_path: Path to the eval YAML config.
        output_dir: Directory where ``hero2_results_*.json`` will be written.
        agent_instance: Optional pre-instantiated ``Agent4FullText`` whose model
            is already loaded (used by ``run_heroes_micro.py`` to share VRAM
            across heroes). When ``None``, a new instance is created and its
            VRAM is freed before returning (standalone mode).
    """
    _seed_torch(42)

    eval_cfg = load_eval_config(config_path)
    sample_path = (PROJECT_ROOT / eval_cfg["sample"]).resolve()
    gold_path = (PROJECT_ROOT / eval_cfg["gold_screening"]).resolve()
    pdfs_dir = (PROJECT_ROOT / eval_cfg["pdfs_dir"]).resolve()

    # ── 1. Load CSVs ───────────────────────────────────────────────────────
    sample_df = pd.read_csv(sample_path, dtype={"covidence_id": str})
    gold_df = pd.read_csv(gold_path, dtype={"covidence_id": str})

    errors: list = []
    papers, skipped = _build_papers(sample_df, gold_df, pdfs_dir, errors)
    sample_size = len(sample_df)
    eval_n = len(papers)

    if eval_n == 0:
        # Edge case: no usable papers. Emit a valid envelope with empty
        # metrics so the orchestrator can still ingest the file.
        return _emit_empty_envelope(
            eval_cfg=eval_cfg,
            sample_size=sample_size,
            output_dir=output_dir,
            errors=errors,
            notes=(
                "No se pudo procesar ningún PDF: todas las filas con has_pdf=True "
                "fallaron en extracción o no había PDFs disponibles."
            ),
        )

    # ── 2. Agent4 setup ────────────────────────────────────────────────────
    from agents.agent4_fulltext import Agent4FullText

    a4_params = _load_agent4_params()
    owns_agent = agent_instance is None
    if agent_instance is None:
        # ensemble_size=1 is the project default for A4 (line 71 of
        # agents/agent4_fulltext.py); we honour it for reproducibility.
        agent = Agent4FullText(ensemble_size=1)
    else:
        agent = agent_instance

    # Agent4 hardcodes temperature=0.2 inside _simple_generate (line ~2318).
    # We record the effective values rather than overrides we cannot inject.
    llm_config = {
        "temperature": 0.2,
        "top_p": getattr(agent, "top_p", None),
        "max_tokens": 200,                                  # hardcoded in _single_fulltext_vote
        "repetition_penalty": a4_params.get("repetition_penalty", 1.3),
        "seed": 42,
        "ensemble_size": agent.ensemble_size,
        "voting_threshold": agent.voting_threshold,
        "max_input_chars": a4_params.get("max_input_chars", 6000),
    }

    model_used = agent.model_path or "unknown"
    if isinstance(model_used, str) and "qwen" in model_used.lower():
        model_display = "Qwen2.5-7B-Instruct (4-bit BitsAndBytes)"
    else:
        model_display = str(model_used)

    # ── 3. Run Phase B (full-text screening) ───────────────────────────────
    start_ts = time.time()
    try:
        screen_out = agent.screen_fulltext(papers, PAULA_SPORT_PROTOCOL)
    except Exception as e:
        errors.append(
            {
                "input_id": "<batch>",
                "error_type": "ModelError",
                "message": f"screen_fulltext raised: {type(e).__name__}: {e}",
            }
        )
        screen_out = {"papers": papers, "include": [], "exclude": [], "stats": {}}
    elapsed = round(time.time() - start_ts, 2)

    # screen_fulltext mutates papers in-place, so we can iterate over the same
    # list we built. Build a lookup by pmid (== covidence_id) for safety.
    a4_by_id: dict = {p["pmid"]: p for p in screen_out.get("papers", papers)}

    # ── 4. Per-paper rows + label vectors for metrics ──────────────────────
    raw_predictions: list = []
    y_true: list[int] = []                 # 1 = include, 0 = exclude (gold)
    y_pred: list[int] = []                 # 1 = include, 0 = exclude (A4)
    cat_true: list[str] = []               # gold reason or "include"
    cat_pred: list[str] = []               # A4 normalised reason or "include"
    pairs: list[tuple[str, str]] = []      # for the 8x8 matrix
    unknown_a4_reasons: list[str] = []
    mismatches: list = []

    for paper in papers:
        cid = paper["pmid"]
        gold_decision = paper["_gold_decision"]  # 'include' or 'exclude'
        gold_reason_norm = paper["_gold_reason_norm"]
        gold_reason_raw = paper["_gold_reason_raw"]

        a4_paper = a4_by_id.get(cid, paper)
        a4_decision_raw = (a4_paper.get("fulltext_decision") or "EXCLUDE").upper()
        a4_decision = "include" if a4_decision_raw == "INCLUDE" else "exclude"
        a4_reason_raw = a4_paper.get("exclusion_reason") or ""
        if a4_decision == "include":
            a4_reason_norm = "include"
        else:
            a4_reasoning_text = a4_paper.get("fulltext_reasoning") or ""
            a4_reason_norm = _normalise_a4_reason(a4_reason_raw, a4_reasoning_text)
            if a4_reason_norm == "unknown_reason" and a4_reason_raw:
                unknown_a4_reasons.append(a4_reason_raw)

        y_true.append(1 if gold_decision == "include" else 0)
        y_pred.append(1 if a4_decision == "include" else 0)
        cat_true.append(gold_reason_norm)
        cat_pred.append(a4_reason_norm)
        pairs.append((gold_reason_norm, a4_reason_norm))

        match_decision = (gold_decision == a4_decision)
        match_reason = (gold_reason_norm == a4_reason_norm)
        match = match_decision and match_reason

        if not match:
            mismatches.append(
                {
                    "covidence_id": cid,
                    "gold_decision": gold_decision,
                    "gold_reason": gold_reason_raw,
                    "gold_reason_norm": gold_reason_norm,
                    "pred_decision": a4_decision,
                    "pred_reason": a4_reason_raw,
                    "pred_reason_norm": a4_reason_norm,
                    "fulltext_reasoning": a4_paper.get("fulltext_reasoning", ""),
                    "screening_confidence": a4_paper.get("screening_confidence", 0.0),
                    "title": _truncate(paper.get("title", ""), 120),
                }
            )

        votes_dump = a4_paper.get("fulltext_votes", []) or []
        if votes_dump:
            raw_pieces = []
            for i, v in enumerate(votes_dump):
                if isinstance(v, dict):
                    d = v.get("decision", "")
                    r = v.get("reasoning", "")
                    er = v.get("exclusion_reason", "")
                    raw_pieces.append(
                        f"vote{i+1}={d}: reasoning={r} | exclusion_reason={er}"
                    )
                else:
                    raw_pieces.append(f"vote{i+1}={v}")
            raw_llm_response = "\n".join(raw_pieces)
        else:
            raw_llm_response = ""

        raw_predictions.append(
            {
                "input_id": cid,
                "gold": {
                    "decision": gold_decision,
                    "reason": gold_reason_raw,
                    "reason_norm": gold_reason_norm,
                },
                "predicted": {
                    "decision": a4_decision,
                    "reason": a4_reason_raw,
                    "reason_norm": a4_reason_norm,
                },
                "match": match,
                "details": {
                    "raw_llm_response": raw_llm_response,
                    "field_scores": {
                        "decision_match": match_decision,
                        "reason_match": match_reason,
                        "fulltext_reasoning": a4_paper.get("fulltext_reasoning", ""),
                        "votes": votes_dump,
                        "screening_confidence": a4_paper.get(
                            "screening_confidence", 0.0
                        ),
                        "fulltext_confidence": a4_paper.get(
                            "fulltext_confidence", 0.0
                        ),
                        "borderline": a4_paper.get("borderline", False),
                        "text_source": a4_paper.get("text_source", "local_pdf"),
                        "full_text_length": a4_paper.get("full_text_length", 0),
                        "pdf_filename": paper.get("_pdf_filename", ""),
                    },
                },
            }
        )

    # ── 5. Metrics ─────────────────────────────────────────────────────────
    binary_metrics, confusion_matrix = _compute_binary_metrics(y_true, y_pred)
    kappa = _cohens_kappa(cat_true, cat_pred)
    extra_labels = sorted({r for r in cat_pred if r not in MATRIX_LABELS})
    stratified_matrix, matrix_labels = _build_stratified_matrix(
        pairs, extra_labels=extra_labels
    )

    metrics = dict(binary_metrics)
    metrics["cohens_kappa"] = round(kappa, 4) if kappa is not None else None
    metrics["stratified_matrix_8x8"] = stratified_matrix
    metrics["stratified_matrix_labels"] = matrix_labels
    metrics["unknown_a4_reasons_sample"] = unknown_a4_reasons[:20]
    metrics["mismatches"] = mismatches

    notes = (
        "Hero 2 evalúa Agent4.screen_fulltext (Phase B) en muestras con PDF "
        "local. La cascada HTTP de Phase A se bypasa: text_source='local_pdf'. "
        f"PDFs procesados: {eval_n}/{sample_size} ({sample_size - eval_n} "
        "saltados por has_pdf=False o fallo de extracción)."
    )

    # ── 6. Final JSON envelope ─────────────────────────────────────────────
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "hero_id": HERO_ID,
        "agent": AGENT_TAG,
        "sample_path": str(Path(eval_cfg["sample"])),
        "sample_size": sample_size,
        "eval_size": eval_n,
        "model_used": model_display,
        "llm_config": llm_config,
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": elapsed,
        "environment": {
            "python_version": platform.python_version(),
            "ollama_version": None,
            "gpu_info": _gpu_info(),
        },
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        "raw_predictions": raw_predictions,
        "errors": errors,
        "notes": notes,
    }

    # ── 7. Persist JSON ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"hero2_results_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    _print_short_report(result, out_path, skipped=skipped)

    # Free VRAM before exit (Agent4 holds the model after screen_fulltext).
    # Only when we own the instance — injected agents are managed by caller.
    if owns_agent:
        try:
            agent.free_vram()
        except Exception:
            pass

    return result


def _emit_empty_envelope(
    eval_cfg: dict,
    sample_size: int,
    output_dir: Path,
    errors: list,
    notes: str,
) -> dict:
    """Emit a minimal but schema-conformant JSON when no papers are evaluable."""
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "hero_id": HERO_ID,
        "agent": AGENT_TAG,
        "sample_path": str(Path(eval_cfg.get("sample", ""))),
        "sample_size": sample_size,
        "eval_size": 0,
        "model_used": "n/a (no papers processed)",
        "llm_config": {
            "temperature": 0.2,
            "top_p": None,
            "max_tokens": 200,
            "repetition_penalty": 1.3,
            "seed": 42,
        },
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": 0.0,
        "environment": {
            "python_version": platform.python_version(),
            "ollama_version": None,
            "gpu_info": _gpu_info(),
        },
        "metrics": {
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "cohens_kappa": None,
            "stratified_matrix_8x8": {},
            "stratified_matrix_labels": list(MATRIX_LABELS),
            "mismatches": [],
        },
        "confusion_matrix": None,
        "raw_predictions": [],
        "errors": errors,
        "notes": notes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"hero2_results_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[WARN] Hero 2 emitted empty envelope -> {out_path}")
    return result


def _print_short_report(result: dict, out_path: Path, skipped: list | None = None) -> None:
    """Print a brief Spanish summary to stdout."""
    m = result["metrics"]
    cm = result["confusion_matrix"] or {}
    skipped = skipped or []
    print()
    print("=" * 64)
    print("HERO 2 (A4 — Screening Full-Text): RESUMEN")
    print("=" * 64)
    print(f"Sample size            : {result['sample_size']}")
    print(f"Papers evaluados       : {result.get('eval_size', 0)}")
    print(f"Saltados (sin PDF/etc) : {len(skipped)}")
    print(f"Modelo                 : {result['model_used']}")
    print(f"Tiempo total           : {result['elapsed_seconds']}s")
    print()
    print(f"Accuracy               : {m.get('accuracy')}")
    print(f"Sensitivity            : {m.get('sensitivity')}")
    print(f"Specificity            : {m.get('specificity')}")
    print(f"F1                     : {m.get('f1')}")
    print(f"Cohen's kappa          : {m.get('cohens_kappa')}")
    print()
    print("Matriz de confusión (binaria):")
    inc = cm.get("include", {})
    exc = cm.get("exclude", {})
    print(f"   gold=include  -> pred include={inc.get('include', 0)}, exclude={inc.get('exclude', 0)}")
    print(f"   gold=exclude  -> pred include={exc.get('include', 0)}, exclude={exc.get('exclude', 0)}")
    print()
    unknowns = m.get("unknown_a4_reasons_sample", [])
    if unknowns:
        print(f"Razones A4 sin mapear (muestra de hasta 20): {len(unknowns)} casos")
        for r in unknowns[:5]:
            print(f"   - {_truncate(r, 120)}")
    print(f"\nOutput JSON guardado en: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hero 2 evaluator — Agent4 full-text screening vs Paula gold standard."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to eval YAML config (default: config/eval/micro.yaml).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for hero2_results_*.json (default: outputs/eval/micro/).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()

    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2

    run_hero2_evaluation(config_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
