"""
Hero 4 evaluator -- Agent6 Quality / Risk-of-Bias vs Paula's gold standard.

Sprint:  T16.D.5.E
Schema:  docs/heroes_output_schema.md (section 3.4) -- this evaluator EXTENDS
         that schema by emitting ``metrics.primary`` (worst-case aggregation,
         the metric that goes into the memoria) AND ``metrics.secondary``
         (one-to-many diagnostic). The top-level ``confusion_matrix`` mirrors
         ``metrics.primary.confusion_matrix`` so the canonical 3x3 contract is
         still satisfied.

Inputs:
  - config/eval/micro.yaml          : paths to sample / gold / mapping / PDFs
  - data/gold_standard/deporte/samples/micro/sample_50.csv
    (filtered to the 10 rows tagged with `hero4` in the `purpose` column)
  - data/gold_standard/deporte/unified/gold_screening_unified.csv
    (joined to get title / abstract / authors / year / journal)
  - data/gold_standard_normalized/gold_rob.csv
    (147-row reference table with the per-outcome RoB judgments)
  - data/gold_standard/deporte/unified/ref_to_covidence_mapping.csv
    (ref_number -> covidence_id, confidence='high' only)
  - data/gold_standard/deporte/pdfs_todos/*.pdf
  - config/agents.json              : per-agent LLM params (agent6_quality)

Output:
  outputs/eval/micro/hero4_results_{YYYYMMDD_HHMMSS}.json


Granularity (decided by the user) -- TWO METRICS, both reported
---------------------------------------------------------------
Agent6 emits ONE RoB ``overall`` per paper but Paula's gold standard records
1-3 RoB judgments per paper (one per outcome). We bridge the granularity
mismatch with two complementary views:

PRIMARY -- "worst-case aggregation"
    Collapse the 1-3 gold RoB judgments per paper to a SINGLE value using
    ``max`` on the ordinal scale L(0) < SC(1) < H(2). Compares the A6
    prediction against the worst-case gold in 10 independent pairs. Reports
    Cohen's kappa (quadratic + linear), accuracy, and a 3x3 confusion matrix.
    Caveat: N=10 gives low statistical power for κ with 3 classes.

SECONDARY -- "one-to-many replication" (diagnostic only)
    A6's single overall is replicated for every outcome of the same paper in
    the gold, producing ~18-20 (covidence_id, outcome) pairs. Reports the
    fraction of outcomes whose gold matches the replicated prediction and a
    per-study coverage breakdown. NO kappa is computed (replication would
    violate the independence assumption).


A6 overall scale -> {L, SC, H} normalisation
--------------------------------------------
A6 emits ``overall`` in different scales depending on the design-detection
branch (see ``agents/agent6_quality.py``):

    RoB 2.0   : "Low" | "Some concerns" | "High"
    ROBINS-I  : "Low" | "Moderate" | "Serious" | "Critical" | "No information"
    JBI       : "Include" | "Include with caution" | "Exclude"

Mapping to the gold ordinal {L, SC, H}:

    L  <- Low / Include
    SC <- Some concerns / Moderate / Include with caution / No information
    H  <- High / Serious / Critical / Exclude

The "No information" ROBINS-I bucket is mapped to SC (middle) because the
quality_weight assigned by A6 (0.3) places it between Low (1.0) and the
Serious/Critical tail. This choice is documented in ``metrics.notes``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is on sys.path so `agents` and `experiments` packages
# resolve when launching as `python experiments/eval_hero4_rob.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._eval_common import (  # noqa: E402
    PAULA_SPORT_PROTOCOL,
    extract_pdf_text as _extract_pdf_text,
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
HERO_ID = "hero4"
AGENT_TAG = "A6"

# Canonical ordinal scale used for kappa and the confusion matrix.
ROB_SCALE = ["L", "SC", "H"]
ROB_ORDINAL = {"L": 0, "SC": 1, "H": 2}

# Marginal extraction: PDF extractions below this many chars are flagged in
# errors[] as `marginal_extraction`. The paper is NOT skipped — downstream
# quality assessment proceeds — but the marker lets post-hoc analysis identify
# cases where the RoB judgement is likely unreliable (e.g. mostly-empty PDFs
# such as appendices or pages with embedded images that defeat the cascade).
MARGINAL_EXTRACTION_THRESHOLD = 1000

# A6 -> {L, SC, H} normalisation. Keys are lower-cased after strip().
# Anything missing maps to None and is recorded as a parse error.
A6_OVERALL_NORMALISATION: dict[str, str] = {
    # RoB 2.0
    "low": "L",
    "some concerns": "SC",
    "high": "H",
    # ROBINS-I (Low already handled above)
    "moderate": "SC",
    "serious": "H",
    "critical": "H",
    "no information": "SC",
    # JBI
    "include": "L",
    "include with caution": "SC",
    "exclude": "H",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — sample / gold loading
# ──────────────────────────────────────────────────────────────────────────────

def _filter_hero4_includes(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only sample rows tagged with 'hero4' in the `purpose` column."""
    purpose = sample_df["purpose"].fillna("").astype(str)
    mask = purpose.str.contains("hero4", case=False, na=False)
    return sample_df[mask].reset_index(drop=True)


def _parse_ref_number(ref_str: object) -> set[int]:
    """Parse a gold_rob `ref_number` cell into a set of integers.

    Gold rows may carry ranges (``"3–5"``, ``"3-5"``) and comma lists
    (``"12,13"``, ``"49,50"``). Returns an empty set for unparseable input.
    """
    s = str(ref_str or "").strip()
    if not s:
        return set()
    refs: set[int] = set()
    # Normalise en/em dashes to a plain hyphen, then split on commas.
    normalised = s.replace("–", "-").replace("—", "-")
    for chunk in normalised.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            try:
                lo_s, hi_s = chunk.split("-", 1)
                lo, hi = int(lo_s.strip()), int(hi_s.strip())
                if lo <= hi:
                    refs.update(range(lo, hi + 1))
                else:
                    refs.add(lo)
                    refs.add(hi)
            except ValueError:
                continue
        else:
            try:
                refs.add(int(chunk))
            except ValueError:
                continue
    return refs


def _build_ref_to_covidence(
    ref_map_df: pd.DataFrame,
    include_ids: set[str],
) -> tuple[dict[int, str], set[int]]:
    """Build a ``ref_number -> covidence_id`` lookup restricted to high-confidence pairs.

    Returns:
        (ref_to_cid, all_refs_for_includes). ``ref_to_cid`` contains every
        high-confidence (ref_number, covidence_id) mapping (used to attribute
        gold_rob rows). ``all_refs_for_includes`` is the subset of ref_numbers
        whose covidence_id is in `include_ids` (used to filter gold_rob to the
        10 hero4 studies).
    """
    has_confidence = "confidence" in ref_map_df.columns
    if has_confidence:
        high = ref_map_df[
            ref_map_df["confidence"].astype(str).str.lower() == "high"
        ]
    else:
        # Defensive fallback: schema documents a confidence column, but if it
        # is ever missing we treat every row as high-confidence rather than
        # silently drop the mapping.
        high = ref_map_df

    ref_to_cid: dict[int, str] = {}
    refs_for_includes: set[int] = set()
    for _, row in high.iterrows():
        cid_val = row.get("covidence_id")
        ref_val = row.get("ref_number")
        if cid_val is None or (isinstance(cid_val, float) and pd.isna(cid_val)):
            continue
        try:
            ref_int = int(ref_val)
        except (TypeError, ValueError):
            continue
        cid = str(cid_val).strip()
        if not cid:
            continue
        ref_to_cid[ref_int] = cid
        if cid in include_ids:
            refs_for_includes.add(ref_int)
    return ref_to_cid, refs_for_includes


def _build_gold_subset(
    gold_rob_df: pd.DataFrame,
    ref_to_cid: dict[int, str],
    target_refs: set[int],
) -> tuple[pd.DataFrame, dict]:
    """Filter gold_rob to rows whose ref_number resolves to one of the target studies.

    For each gold_rob row:
      1. Parse ``ref_number`` into a set of integers.
      2. Intersect with ``target_refs``.
      3. If any intersection, attribute the row to the corresponding
         covidence_id (multi-ref rows like ``"49,50"`` should all point to the
         same study; we assert this is the case and otherwise prefer the first
         matched ref).
      4. Populate ``_derived_covidence_id`` with the resolved id. Rows whose
         original ``covidence_id`` is NaN inherit the derived value.
    """
    derived_cids: list[Optional[str]] = []
    parsed_refs_col: list[list[int]] = []
    kept_idx: list[int] = []
    multi_cid_warnings: list[dict] = []

    for idx, row in gold_rob_df.iterrows():
        refs_here = _parse_ref_number(row.get("ref_number"))
        parsed_refs_col.append(sorted(refs_here))
        intersection = refs_here & target_refs
        if not intersection:
            derived_cids.append(None)
            continue
        # Resolve to covidence_ids; in normal data all refs in `intersection`
        # share the same parent study (e.g. refs 49 and 50 -> #5298).
        cids_resolved = {ref_to_cid[r] for r in intersection if r in ref_to_cid}
        if not cids_resolved:
            derived_cids.append(None)
            continue
        if len(cids_resolved) > 1:
            multi_cid_warnings.append({
                "row_index": int(idx),
                "ref_number": str(row.get("ref_number")),
                "resolved_cids": sorted(cids_resolved),
            })
        # Deterministic choice when multiple cids match: smallest ref wins.
        sorted_refs = sorted(intersection)
        chosen_cid = next(
            (ref_to_cid[r] for r in sorted_refs if r in ref_to_cid),
            None,
        )
        derived_cids.append(chosen_cid)
        kept_idx.append(int(idx))

    annotated = gold_rob_df.copy()
    annotated["_parsed_refs"] = parsed_refs_col
    annotated["_derived_covidence_id"] = derived_cids
    subset = annotated.loc[kept_idx].reset_index(drop=True)

    debug = {
        "total_gold_rob_rows": int(len(gold_rob_df)),
        "high_conf_refs_for_includes": sorted(target_refs),
        "subset_rows": int(len(subset)),
        "subset_by_covidence": (
            subset["_derived_covidence_id"].value_counts().to_dict()
            if len(subset)
            else {}
        ),
        "multi_cid_rows": multi_cid_warnings,
    }
    return subset, debug


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — paper construction
# ──────────────────────────────────────────────────────────────────────────────

def _build_papers(
    includes_df: pd.DataFrame,
    gold_screening_df: pd.DataFrame,
    pdfs_dir: Path,
    errors: list,
) -> tuple[list[dict], list[dict]]:
    """Build paper dicts ready for ``Agent6Quality.assess_all``.

    Each dict carries: pmid (= covidence_id), title, abstract, authors, year,
    journal, doi, full_text. Rows whose PDF cannot be extracted are skipped
    with an ``errors[]`` entry.
    """
    gold_idx = gold_screening_df.set_index("covidence_id")
    papers: list[dict] = []
    skipped: list[dict] = []

    for _, row in includes_df.iterrows():
        cid = str(row["covidence_id"])
        pdf_filename = row.get("pdf_filename")
        has_pdf = bool(row.get("has_pdf", False))

        if (
            not has_pdf
            or pdf_filename is None
            or (isinstance(pdf_filename, float) and pd.isna(pdf_filename))
        ):
            errors.append({
                "input_id": cid,
                "error_type": "no_pdf_available",
                "message": (
                    f"Sample row marked has_pdf=False (pdf_filename={pdf_filename!r}); "
                    "skipped from RoB evaluation."
                ),
            })
            skipped.append({"covidence_id": cid, "reason": "no_pdf_available"})
            continue

        pdf_path = pdfs_dir / str(pdf_filename)
        text, err = _extract_pdf_text(pdf_path)
        if text is None:
            msg = f"PDF '{pdf_filename}' extraction failed: {err}"
            if err and "empty text" in err:
                msg += (
                    " (likely scanned PDF without OCR; future work: "
                    "add OCR cascade to _eval_common.extract_pdf_text)"
                )
            errors.append({
                "input_id": cid,
                "error_type": "pdf_extraction_failed",
                "message": msg,
            })
            skipped.append({"covidence_id": cid, "reason": "pdf_extraction_failed"})
            continue

        chars_extracted = len(text.strip())
        if chars_extracted < MARGINAL_EXTRACTION_THRESHOLD:
            errors.append({
                "input_id": cid,
                "error_type": "marginal_extraction",
                "message": (
                    f"PDF '{pdf_filename}' extracted only {chars_extracted} "
                    f"chars (< {MARGINAL_EXTRACTION_THRESHOLD}). Paper is NOT "
                    "skipped; downstream RoB judgement may be unreliable."
                ),
            })

        title = ""
        abstract = ""
        authors: list = []
        year: object = ""
        journal: str = ""
        doi: str = ""
        if cid in gold_idx.index:
            gold_row = gold_idx.loc[cid]
            if isinstance(gold_row, pd.DataFrame):
                gold_row = gold_row.iloc[0]
            title = str(gold_row.get("title", "") or "").strip()
            abstract = str(gold_row.get("abstract", "") or "").strip()
            if title.lower() in ("nan", "none"):
                title = ""
            if abstract.lower() in ("nan", "none"):
                abstract = ""
            for src_key in ("authors", "year", "journal", "doi"):
                if src_key in gold_row.index:
                    val = gold_row.get(src_key)
                    if val is None or (
                        isinstance(val, float) and pd.isna(val)
                    ):
                        continue
                    s = str(val).strip()
                    if s.lower() in ("nan", "none") or not s:
                        continue
                    if src_key == "authors":
                        authors = [a.strip() for a in s.split(";") if a.strip()] or [s]
                    elif src_key == "year":
                        year = s
                    elif src_key == "journal":
                        journal = s
                    elif src_key == "doi":
                        doi = s

        papers.append({
            "pmid": cid,                  # covidence_id is the canonical identifier
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "journal": journal,
            "doi": doi,
            "full_text": text,
            "full_text_length": len(text),
            "text_source": "local_pdf",
            "_pdf_filename": str(pdf_filename),
        })

    return papers, skipped


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — A6 normalisation + metrics
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_a6_overall(overall: object) -> Optional[str]:
    """Normalise an A6 ``quality_assessment.overall`` string to {L, SC, H}.

    Returns ``None`` if the value is empty or outside the documented A6 scale.
    Comparison is case-insensitive and tolerant of surrounding whitespace.
    """
    if overall is None:
        return None
    s = str(overall).strip().lower()
    if not s:
        return None
    return A6_OVERALL_NORMALISATION.get(s)


def _worst_case_gold(judgments: list[str]) -> Optional[str]:
    """Collapse 1-N gold RoB judgments to the worst (max ordinal) value.

    Inputs must be members of ROB_SCALE. Returns ``None`` if the input list is
    empty or contains only unknown values.
    """
    valid = [j for j in judgments if j in ROB_ORDINAL]
    if not valid:
        return None
    worst_score = max(ROB_ORDINAL[j] for j in valid)
    inverse = {v: k for k, v in ROB_ORDINAL.items()}
    return inverse[worst_score]


# --- T17.C.3 C.2: additive replicas of agent6_quality.py RoB aggregation doctrine ---
# Shared so the Claude harness (eval_a6_claude.py) recomputes overall/weight with the
# SAME logic the Qwen track applies, instead of a parallel copy.

# Canonical casing per tool — whitelist replicated from agent6_quality.py:211-228 / 288-302 / 360-376
_ROB2_CANON = {"low": "Low", "some concerns": "Some concerns", "high": "High"}
_ROBINS_CANON = {"low": "Low", "moderate": "Moderate", "serious": "Serious",
                 "critical": "Critical", "no information": "No information"}
_JBI_CANON = {"yes": "Yes", "no": "No", "unclear": "Unclear", "not applicable": "Not applicable"}


def canonicalize_domain_verdict(raw, tool):
    """Normalize a raw judgment/answer to canonical casing, mirroring the whitelist
    normalization Qwen applies before aggregation (agent6_quality.py:211-228/288-302/360-376).
    Out-of-whitelist -> conservative default identical to Qwen
    (RoB2 'Some concerns', ROBINS-I 'No information', JBI 'Unclear')."""
    s = str(raw or "").strip().lower()
    if tool == "RoB2":
        return _ROB2_CANON.get(s, "Some concerns")
    if tool == "ROBINS-I":
        return _ROBINS_CANON.get(s, "No information")
    return _JBI_CANON.get(s, "Unclear")  # JBI


def aggregate_rob2_overall(domains):
    """Least-favorable rule. Identical to agent6_quality.py:230-237. Assumes canonical judgments."""
    judgments = [d.get("judgment") for d in domains.values()]
    if "High" in judgments:
        return "High"
    elif "Some concerns" in judgments:
        return "Some concerns"
    return "Low"


def aggregate_robins_i_overall(domains):
    """Worst-case rule. Identical to agent6_quality.py:304-315. Assumes canonical judgments."""
    judgments = [d.get("judgment") for d in domains.values()]
    if "Critical" in judgments:
        return "Critical"
    elif "Serious" in judgments:
        return "Serious"
    elif "Moderate" in judgments:
        return "Moderate"
    elif all(j == "No information" for j in judgments):
        return "No information"
    return "Low"


def aggregate_jbi_overall(items):
    """Count-based rule. Identical to agent6_quality.py:378-385.
    DECLARED DEVIATION: Qwen counts field 'answer'; the Claude schema uses 'judgment'.
    Reads 'answer' then falls back to 'judgment'. Thresholds/literals unchanged."""
    yes_count = sum(1 for it in items.values()
                    if (it.get("answer") or it.get("judgment")) == "Yes")
    if yes_count >= 8:
        overall = "Include"
    elif yes_count >= 5:
        overall = "Include with caution"
    else:
        overall = "Exclude"
    return yes_count, overall


def compute_quality_weight(overall):
    """Identical to the weight_map of agent6_quality.py:422-438."""
    overall = (overall or "").lower()
    weight_map = {
        "low": 1.0, "some concerns": 0.6, "high": 0.3,
        "moderate": 0.7, "serious": 0.4, "critical": 0.2, "no information": 0.3,
        "include": 1.0, "include with caution": 0.6, "exclude": 0.2,
    }
    return weight_map.get(overall, 0.5)


def _empty_confusion_matrix() -> dict[str, dict[str, int]]:
    """Build an empty 3x3 confusion matrix indexed by ROB_SCALE."""
    return {row: {col: 0 for col in ROB_SCALE} for row in ROB_SCALE}


def _compute_primary_metrics(
    pairs: list[tuple[str, str]],
) -> dict:
    """Compute primary (worst-case) metrics from ``[(gold, predicted)]`` pairs.

    Both gold and predicted must already be normalised to {L, SC, H}.
    """
    confusion = _empty_confusion_matrix()
    accuracy_hits = 0
    for gold, pred in pairs:
        confusion[gold][pred] += 1
        if gold == pred:
            accuracy_hits += 1

    n_pairs = len(pairs)
    kappa_quadratic: Optional[float] = None
    kappa_linear: Optional[float] = None

    if n_pairs >= 2:
        try:
            from sklearn.metrics import cohen_kappa_score  # type: ignore

            gold_idx = [ROB_ORDINAL[g] for g, _ in pairs]
            pred_idx = [ROB_ORDINAL[p] for _, p in pairs]
            # If everything collapsed to a single class for one of the two
            # rater sides, sklearn warns about division by zero. Guard:
            if len(set(gold_idx)) == 1 and len(set(pred_idx)) == 1:
                # Perfect agreement -> κ undefined under the standard formula;
                # report 1.0 only when both raters fully agreed.
                if gold_idx == pred_idx:
                    kappa_quadratic = 1.0
                    kappa_linear = 1.0
                else:
                    kappa_quadratic = None
                    kappa_linear = None
            else:
                kappa_quadratic = float(
                    cohen_kappa_score(
                        gold_idx, pred_idx,
                        weights="quadratic",
                        labels=[0, 1, 2],
                    )
                )
                kappa_linear = float(
                    cohen_kappa_score(
                        gold_idx, pred_idx,
                        weights="linear",
                        labels=[0, 1, 2],
                    )
                )
        except ImportError:
            kappa_quadratic = None
            kappa_linear = None

    accuracy_flat = (
        round(accuracy_hits / n_pairs, 4) if n_pairs > 0 else 0.0
    )

    return {
        "description": (
            "Worst-case aggregation: gold per-paper = max(per-outcome) "
            "on ordinal L<SC<H. One pair per paper."
        ),
        "n_pairs": n_pairs,
        "cohens_kappa_quadratic": (
            round(kappa_quadratic, 4) if kappa_quadratic is not None else None
        ),
        "cohens_kappa_linear": (
            round(kappa_linear, 4) if kappa_linear is not None else None
        ),
        "accuracy_flat": accuracy_flat,
        "confusion_matrix": confusion,
        "notes": (
            "N=10 -> low statistical power for κ with 3 classes. "
            "Wide CI expected; treat κ as orientativo, not conclusivo."
        ),
    }


def _compute_secondary_metrics(
    per_outcome_pairs: list[dict],
    per_study_coverage: list[dict],
) -> dict:
    """Compute secondary one-to-many metrics. No kappa is reported here."""
    n_outcomes = len(per_outcome_pairs)
    n_hits = sum(1 for p in per_outcome_pairs if p["gold"] == p["predicted"])
    pct = round(n_hits / n_outcomes, 4) if n_outcomes else 0.0
    return {
        "description": (
            "One-to-many replication: A6 overall replicated per outcome of the "
            "same study. NOT independent: replicates inflate sample size."
        ),
        "n_outcomes": n_outcomes,
        "pct_outcomes_correctly_covered": pct,
        "per_study_coverage": per_study_coverage,
        "notes": (
            "No kappa computed (replication violates independence). Use "
            "primary metrics for any statistical claim."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def run_hero4_evaluation(
    config_path: Path,
    output_dir: Path,
    agent_instance=None,
) -> dict:
    """Run the Hero 4 evaluation and return the result dict (also written to disk).

    Args:
        config_path: Path to the eval YAML config.
        output_dir: Directory where ``hero4_results_*.json`` will be written.
        agent_instance: Optional pre-instantiated ``Agent6Quality`` whose model
            is already loaded (used by ``run_heroes_micro.py`` to share VRAM
            across heroes). When ``None``, a new instance is created and its
            VRAM is freed before returning (standalone mode).
    """
    _seed_torch(42)

    eval_cfg = load_eval_config(config_path)
    sample_path = (PROJECT_ROOT / eval_cfg["sample"]).resolve()
    gold_screening_path = (PROJECT_ROOT / eval_cfg["gold_screening"]).resolve()
    gold_rob_path = (PROJECT_ROOT / eval_cfg["gold_rob"]).resolve()
    ref_map_path = (PROJECT_ROOT / eval_cfg["ref_to_covidence"]).resolve()
    pdfs_dir = (PROJECT_ROOT / eval_cfg["pdfs_dir"]).resolve()

    # ── 1. Load CSVs ───────────────────────────────────────────────────────
    sample_df = pd.read_csv(sample_path, dtype={"covidence_id": str})
    gold_screening_df = pd.read_csv(
        gold_screening_path, dtype={"covidence_id": str}
    )
    gold_rob_df = pd.read_csv(gold_rob_path)
    ref_map_df = pd.read_csv(ref_map_path)

    includes_df = _filter_hero4_includes(sample_df)
    include_ids_list = includes_df["covidence_id"].astype(str).tolist()
    include_ids: set[str] = set(include_ids_list)
    sample_size = int(len(includes_df))

    errors: list = []
    papers, skipped = _build_papers(
        includes_df, gold_screening_df, pdfs_dir, errors
    )
    eval_n = len(papers)

    # ── 2. Resolve ref_number -> covidence_id and build gold subset ────────
    ref_to_cid, target_refs = _build_ref_to_covidence(ref_map_df, include_ids)
    gold_subset, gold_debug = _build_gold_subset(
        gold_rob_df, ref_to_cid, target_refs
    )

    if eval_n == 0:
        return _emit_empty_envelope(
            eval_cfg=eval_cfg,
            sample_size=sample_size,
            output_dir=output_dir,
            errors=errors,
            gold_debug=gold_debug,
            notes=(
                "No se pudo procesar ningún PDF: ningún hero4 include tenía "
                "PDF utilizable. Verifica `has_pdf` y la existencia de los "
                "ficheros en pdfs_dir."
            ),
        )

    # ── 3. Agent6 setup ────────────────────────────────────────────────────
    from agents.agent6_quality import Agent6Quality

    a6_params = load_agent_params("agent6_quality")
    owns_agent = agent_instance is None
    if agent_instance is None:
        agent = Agent6Quality()
    else:
        agent = agent_instance

    # Record effective generation parameters. Agent6._simple_generate uses
    # ``self.temperature`` which is 0.2 by default (TEMPERATURE_CONFIG).
    llm_config = {
        "temperature": agent.temperature,
        "top_p": getattr(agent, "top_p", None),
        "max_tokens": 800,
        "repetition_penalty": a6_params.get("repetition_penalty", 1.3),
        "seed": 42,
        "max_input_chars_assessment": agent._assessment_max_chars,
        "max_input_chars_design": agent._design_max_chars,
    }

    model_used = agent.model_path or "unknown"
    if isinstance(model_used, str) and "qwen" in model_used.lower():
        model_display = "Qwen2.5-7B-Instruct (4-bit BitsAndBytes)"
    else:
        model_display = str(model_used)

    # ── 4. Run quality assessment ──────────────────────────────────────────
    start_ts = time.time()
    try:
        assess_out = agent.assess_all(papers, PAULA_SPORT_PROTOCOL)
    except Exception as e:
        errors.append({
            "input_id": "<batch>",
            "error_type": "ModelError",
            "message": f"assess_all raised: {type(e).__name__}: {e}",
        })
        assess_out = {"papers": [], "stats": {}}
    elapsed = round(time.time() - start_ts, 2)

    a6_by_id: dict[str, dict] = {}
    for p in assess_out.get("papers", []):
        pid = str(p.get("pmid", ""))
        if pid:
            a6_by_id[pid] = p

    # ── 5. Aggregate gold by paper + build primary/secondary pairs ─────────
    # gold_by_cid: covidence_id -> list of {outcome_name, gold_rob} entries
    gold_by_cid: dict[str, list[dict]] = defaultdict(list)
    for _, row in gold_subset.iterrows():
        cid = str(row["_derived_covidence_id"])
        outcome_name = str(row.get("outcome") or "").strip()
        rob_judgment = str(row.get("rob_judgment") or "").strip()
        if rob_judgment not in ROB_ORDINAL:
            errors.append({
                "input_id": f"{cid}::{outcome_name}",
                "error_type": "ValidationError",
                "message": (
                    f"Gold rob_judgment {rob_judgment!r} outside expected scale "
                    f"{ROB_SCALE}; row skipped."
                ),
            })
            continue
        gold_by_cid[cid].append({
            "outcome_name": outcome_name,
            "gold_rob": rob_judgment,
            "obj_sf": str(row.get("obj_sf") or "").strip(),
            "instrument_human": str(row.get("instrument_human") or "").strip(),
            "ref_number": str(row.get("ref_number") or "").strip(),
        })

    # Primary pairs: one per study (worst-case aggregation)
    primary_pairs: list[tuple[str, str]] = []
    # Secondary pairs: one per outcome (replication of A6 overall)
    secondary_pairs: list[dict] = []
    per_study_coverage: list[dict] = []
    # raw_predictions: both levels concatenated
    raw_predictions: list = []

    for cid in include_ids_list:
        a6_result = a6_by_id.get(cid)
        gold_outcomes = gold_by_cid.get(cid, [])

        if a6_result is None:
            errors.append({
                "input_id": cid,
                "error_type": "missing_a6_result",
                "message": (
                    "Agent6 returned no entry for this covidence_id; "
                    "study excluded from primary and secondary pairs."
                ),
            })
            continue

        quality = a6_result.get("quality_assessment") or {}
        a6_overall_raw = quality.get("overall")
        a6_overall_norm = _normalize_a6_overall(a6_overall_raw)
        a6_tool = quality.get("tool", "")
        design = a6_result.get("study_design_detected", "")
        weight = a6_result.get("quality_weight")

        if a6_overall_norm is None:
            errors.append({
                "input_id": cid,
                "error_type": "ValidationError",
                "message": (
                    f"A6 overall {a6_overall_raw!r} could not be normalised to "
                    f"{ROB_SCALE} (tool={a6_tool!r}); study excluded from "
                    "primary and secondary pairs."
                ),
            })
            continue

        if not gold_outcomes:
            errors.append({
                "input_id": cid,
                "error_type": "no_gold_outcomes",
                "message": (
                    "No gold RoB outcomes found for this covidence_id via the "
                    "high-confidence ref->covidence mapping."
                ),
            })
            continue

        # --- Primary (one pair, worst-case gold) ---
        gold_worst = _worst_case_gold([o["gold_rob"] for o in gold_outcomes])
        if gold_worst is None:
            errors.append({
                "input_id": cid,
                "error_type": "ValidationError",
                "message": "All gold RoB judgments invalid; primary pair skipped.",
            })
            continue
        primary_pairs.append((gold_worst, a6_overall_norm))

        raw_predictions.append({
            "input_id": cid,
            "gold": gold_worst,
            "predicted": a6_overall_norm,
            "match": gold_worst == a6_overall_norm,
            "details": {
                "level": "paper",
                "a6_overall_raw": a6_overall_raw,
                "a6_tool": a6_tool,
                "design_detected": design,
                "quality_weight": weight,
                "gold_aggregation": "worst_case",
                "gold_outcomes_collapsed": gold_outcomes,
                "raw_llm_response": (
                    quality.get("overall_justification", "") or ""
                ),
                "field_scores": {
                    "domains": quality.get("domains"),
                    "items": quality.get("items"),
                    "yes_count": quality.get("yes_count"),
                },
            },
        })

        # --- Secondary (one pair per outcome, A6 overall replicated) ---
        per_study_failures = 0
        per_study_outcome_records: list[dict] = []
        for outcome in gold_outcomes:
            gold = outcome["gold_rob"]
            match = gold == a6_overall_norm
            if not match:
                per_study_failures += 1
            secondary_pairs.append({
                "covidence_id": cid,
                "outcome_name": outcome["outcome_name"],
                "gold": gold,
                "predicted": a6_overall_norm,
                "match": match,
            })
            per_study_outcome_records.append({
                "outcome_name": outcome["outcome_name"],
                "gold_rob": gold,
                "match": match,
            })

            raw_predictions.append({
                "input_id": f"{cid}::{outcome['outcome_name']}",
                "gold": gold,
                "predicted": a6_overall_norm,
                "match": match,
                "details": {
                    "level": "outcome",
                    "covidence_id": cid,
                    "outcome_name": outcome["outcome_name"],
                    "obj_sf": outcome.get("obj_sf"),
                    "instrument_human": outcome.get("instrument_human"),
                    "ref_number": outcome.get("ref_number"),
                    "a6_overall_raw": a6_overall_raw,
                    "a6_tool": a6_tool,
                    "design_detected": design,
                    "quality_weight": weight,
                    "raw_llm_response": "",
                    "field_scores": {},
                },
            })

        per_study_coverage.append({
            "covidence_id": cid,
            "a6_overall": a6_overall_norm,
            "a6_overall_raw": a6_overall_raw,
            "a6_tool": a6_tool,
            "design_detected": design,
            "gold_outcomes": per_study_outcome_records,
            "failures": per_study_failures,
            "total_outcomes": len(gold_outcomes),
        })

    # ── 6. Compute primary and secondary metrics ───────────────────────────
    primary_metrics = _compute_primary_metrics(primary_pairs)
    secondary_metrics = _compute_secondary_metrics(
        secondary_pairs, per_study_coverage
    )

    # Tool distribution for context (not a metric per se).
    tool_counter: dict[str, int] = defaultdict(int)
    for entry in per_study_coverage:
        tool_counter[entry["a6_tool"] or "unknown"] += 1

    metrics = {
        "primary": primary_metrics,
        "secondary": secondary_metrics,
        "a6_tool_distribution": dict(tool_counter),
        "gold_subset_debug": gold_debug,
        "scale_normalisation": {
            "ROB_SCALE": ROB_SCALE,
            "ROB_ORDINAL": ROB_ORDINAL,
            "a6_overall_to_canonical": A6_OVERALL_NORMALISATION,
        },
        "notes": (
            "Hero 4 evalúa A6 con dos granularidades: PRIMARY (worst-case por "
            "paper, métrica de la memoria) y SECONDARY (one-to-many sin κ). "
            "A6 emite un único `overall` por paper; lo normalizamos a "
            "{L, SC, H}. ROBINS-I 'No information' se mapea a SC porque su "
            "quality_weight (0.3) lo sitúa entre Low (1.0) y Serious/Critical."
        ),
    }

    # ── 7. Final JSON envelope ─────────────────────────────────────────────
    notes = (
        f"Hero 4 evalúa Agent6.assess_all sobre los {sample_size} includes "
        "hero4 del sample_50 (10 estudios). Cobertura gold construida via "
        "mapeo high-confidence ref_number -> covidence_id: "
        f"{len(gold_subset)} filas gold_rob (sobre {len(gold_rob_df)} totales) "
        f"distribuidas en {len(gold_by_cid)} estudios. A6 produjo "
        f"{len(assess_out.get('papers', []))} assessments. "
        f"Primary pairs: {len(primary_pairs)} / Secondary pairs: "
        f"{len(secondary_pairs)}."
    )

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
        # Top-level confusion_matrix mirrors metrics.primary.confusion_matrix
        # so the canonical schema (docs/heroes_output_schema.md §3.4) still
        # holds even though we extend `metrics` with primary/secondary blocks.
        "confusion_matrix": primary_metrics["confusion_matrix"],
        "raw_predictions": raw_predictions,
        "errors": errors,
        "notes": notes,
    }

    # ── 8. Persist JSON ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"hero4_results_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    _print_short_report(result, out_path, skipped=skipped)

    # Free VRAM before exit (Agent6 holds the model after assess_all).
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
    gold_debug: dict,
    notes: str,
) -> dict:
    """Emit a minimal but schema-conformant JSON when no papers are evaluable."""
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    empty_cm = _empty_confusion_matrix()
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
            "max_tokens": 800,
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
            "primary": {
                "description": (
                    "Worst-case aggregation: gold per-paper = "
                    "max(per-outcome) on ordinal L<SC<H. One pair per paper."
                ),
                "n_pairs": 0,
                "cohens_kappa_quadratic": None,
                "cohens_kappa_linear": None,
                "accuracy_flat": 0.0,
                "confusion_matrix": empty_cm,
                "notes": "No primary pairs (no papers evaluated).",
            },
            "secondary": {
                "description": (
                    "One-to-many replication: A6 overall replicated per "
                    "outcome of the same study."
                ),
                "n_outcomes": 0,
                "pct_outcomes_correctly_covered": 0.0,
                "per_study_coverage": [],
                "notes": "No secondary pairs (no papers evaluated).",
            },
            "a6_tool_distribution": {},
            "gold_subset_debug": gold_debug,
            "scale_normalisation": {
                "ROB_SCALE": ROB_SCALE,
                "ROB_ORDINAL": ROB_ORDINAL,
                "a6_overall_to_canonical": A6_OVERALL_NORMALISATION,
            },
            "notes": "No papers could be evaluated.",
        },
        "confusion_matrix": empty_cm,
        "raw_predictions": [],
        "errors": errors,
        "notes": notes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"hero4_results_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[WARN] Hero 4 emitted empty envelope -> {out_path}")
    return result


def _print_short_report(
    result: dict,
    out_path: Path,
    skipped: list | None = None,
) -> None:
    """Print a brief Spanish summary to stdout."""
    m = result["metrics"]
    prim = m.get("primary", {})
    sec = m.get("secondary", {})
    skipped = skipped or []
    print()
    print("=" * 64)
    print("HERO 4 (A6 — Risk of Bias): RESUMEN")
    print("=" * 64)
    print(f"Sample size (includes) : {result['sample_size']}")
    print(f"Papers evaluados       : {result.get('eval_size', 0)}")
    print(f"Saltados               : {len(skipped)}")
    print(f"Modelo                 : {result['model_used']}")
    print(f"Tiempo total           : {result['elapsed_seconds']}s")
    print()
    print("MÉTRICA PRIMARIA (worst-case, va a la memoria):")
    print(f"   Pares (paper-level)  : {prim.get('n_pairs', 0)}")
    print(f"   κ cuadrática         : {prim.get('cohens_kappa_quadratic')}")
    print(f"   κ lineal             : {prim.get('cohens_kappa_linear')}")
    print(f"   Accuracy (flat)      : {prim.get('accuracy_flat')}")
    print()
    print("MÉTRICA SECUNDARIA (one-to-many, diagnóstica):")
    print(f"   Pares (outcome-level): {sec.get('n_outcomes', 0)}")
    print(
        "   % outcomes cubiertos : "
        f"{sec.get('pct_outcomes_correctly_covered')}"
    )
    tool_dist = m.get("a6_tool_distribution", {})
    if tool_dist:
        print()
        print("Distribución de herramientas A6:")
        for tool, n in tool_dist.items():
            print(f"   {tool}: {n}")
    debug = m.get("gold_subset_debug", {})
    if debug:
        print()
        print(
            "Subset gold construido: "
            f"{debug.get('subset_rows', 0)} filas "
            f"({debug.get('total_gold_rob_rows', 0)} totales gold_rob; "
            f"refs target = {len(debug.get('high_conf_refs_for_includes', []))})."
        )
        if debug.get("multi_cid_rows"):
            print(
                "   Aviso: filas con múltiples covidence_id resueltos = "
                f"{len(debug['multi_cid_rows'])}"
            )
    if skipped:
        print()
        print("Skipped papers:")
        for s in skipped[:5]:
            print(f"   - {s.get('covidence_id')} ({s.get('reason')})")
    print()
    print(
        "Caveat: N=10 da poder estadístico bajo para κ con 3 clases. "
        "IC amplios esperables."
    )
    print(f"\nOutput JSON guardado en: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hero 4 evaluator — Agent6 RoB vs Paula gold standard."
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
        help="Output directory for hero4_results_*.json (default: outputs/eval/micro/).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()

    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2

    run_hero4_evaluation(config_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
