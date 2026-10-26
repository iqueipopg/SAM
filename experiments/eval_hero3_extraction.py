# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Hero 3 evaluator -- Agent5 Data Extraction vs Paula's gold standard.

Sprint:  T16.D.5.D
Schema:  docs/heroes_output_schema.md (section 3.3)
Inputs:
  - config/eval/micro.yaml          : paths to sample / gold / mapping / PDFs
  - data/gold_standard/deporte/samples/micro/sample_50.csv
    (filtered to the 10 rows tagged with `hero3` in the `purpose` column)
  - data/gold_standard/deporte/unified/gold_screening_unified.csv
    (joined to get title / abstract / authors / year / journal)
  - data/gold_standard_normalized/gold_metrics.csv
    (the 824-row reference table with the 6 numeric fields per outcome)
  - data/gold_standard/deporte/unified/arm_to_parent_mapping.csv
    (arm_label -> parent_covidence_id, confidence='high' only)
  - data/gold_standard/deporte/pdfs_todos/*.pdf
  - config/agents.json              : per-agent LLM params (agent5_extraction)

Output:
  outputs/eval/micro/hero3_results_{YYYYMMDD_HHMMSS}.json

Pipeline overview
-----------------
1. Load sample_50, keep only the 10 rows tagged with `hero3`.
2. For each include, extract local PDF text (shared `_eval_common.extract_pdf_text`).
3. Build paper dicts populated with `full_text`, `title`, `abstract`,
   `authors`, `year`, `journal`, `pmid` (= covidence_id), `doi`.
4. Call `Agent5Extraction.extract_all(papers, PAULA_SPORT_PROTOCOL)` ONCE.
5. Load gold_metrics + arm_to_parent_mapping (confidence='high').
   Filter gold rows to the 10 hero3 covidence_ids by computing a per-row
   `effective_parent` via forward-fill: canonical rows starting with `#` are
   their own parent; non-canonical arm rows (e.g. `E2`, `E3`, `GC (...)`,
   `#1354RCT 2`) inherit the nearest preceding canonical parent. The
   arm_to_parent mapping is consulted only to confirm `confidence='high'`
   for any non-canonical row; rows whose (arm_label, parent) pair is absent
   from the high-confidence mapping (or whose mapping confidence is medium
   / unresolved) are dropped.
6. For each `(parent_covidence_id, outcome_name, timepoint)` triple in the
   filtered gold subset, search the matching A5 outcomes (by parent
   covidence_id) and try to match outcome_name + timepoint heuristically.
7. Compare the 6 numeric fields (N_GC, N_GI, X_GC, X_GI, SD_GC, SD_GI) to
   Agent5's `n_control`, `n_intervention`, `mean_control`, `mean_intervention`,
   `sd_control`, `sd_intervention` with tolerances:
       N  : exact integer match
       X  : abs(pred-gold) / abs(gold) <= 0.05  (fallback abs<=1e-3 if gold==0)
       SD : abs(pred-gold) / abs(gold) <= 0.10  (fallback abs<=1e-3 if gold==0)
   A field is `field_failed` if either side is missing/NaN.

A5 outcome dict reference (relevant fields emitted by `agent5_extraction.py`)
----------------------------------------------------------------------------
Per ``AGENT5_OUTCOMES_TEMPLATE`` and ``_normalize_outcome_fields`` /
``_extract_numeric_fields``:

    name                 -> heuristic outcome label (free-form English)
    effect_size_type     -> SMD / OR / RR / HR / MD / PROP
    measure_type         -> uppercase canonical (== effect_size_type)
    effect_size_value, value
    ci_lower, ci_upper, p_value, se
    n_intervention, n_control, n_total
    mean_intervention, sd_intervention
    mean_control, sd_control
    numeric_complete     -> bool flag
    n_per_group          -> {"intervention": ..., "control": ...}

A5 does NOT emit `timepoint`, `instrument`, `outcome_category` natively;
those exist only in the gold standard. We therefore parse a hint of timepoint
from the A5 `name` field when present (e.g. "post-intervention", "baseline").

Match rules
-----------
- Outcome name match: lowercase, strip punctuation/whitespace, then check
  bidirectional substring containment OR token-overlap >= 0.6. Conservative:
  prefer "no_match" over a false positive.
- Timepoint match: a small bidirectional mapping (BASELINE<->baseline,
  POST<->post/end/post-intervention, FU*<->follow-up/follow up/at N months).
  If A5 does not encode a timepoint in `name`, we treat the timepoint as
  "unknown" and DO NOT exclude the candidate solely on timepoint mismatch
  (we report timepoint_match=False but still compare numeric fields).
- outcome_match across the 6 fields:
    * "exact"   -> all 6 fields pass tolerance (none failed/missing)
    * "partial" -> at least 1 field passes and at least 1 fails/missing
    * "no_match"-> no outcome matched, or 0 fields passed
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is on sys.path so `agents` and `experiments` packages
# resolve when launching as `python experiments/eval_hero3_extraction.py`.
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
HERO_ID = "hero3"
AGENT_TAG = "A5"

# Tolerances per docs/heroes_output_schema.md section 3.3
TOL_X_REL = 0.05   # ±5% for means
TOL_SD_REL = 0.10  # ±10% for SDs
ABS_FALLBACK = 1e-3  # used when gold value == 0

# Marginal extraction: PDF extractions below this many chars are flagged in
# errors[] as `marginal_extraction`. The paper is NOT skipped — downstream
# extraction proceeds — but the marker lets post-hoc analysis identify cases
# where the numeric extraction is likely unreliable (e.g. mostly-empty PDFs
# such as appendices or pages with embedded images that defeat the cascade).
MARGINAL_EXTRACTION_THRESHOLD = 1000

# Token-overlap threshold for outcome-name match fallback (conservative).
NAME_TOKEN_OVERLAP_THRESHOLD = 0.6

# Tokens that are too common to disambiguate outcome names on their own.
# When the only overlapping tokens between gold and predicted names are
# unit / common-noun tokens, we DO NOT consider the names matched: e.g.
# "MVPA min/day" and "Sedentary min/day" overlap on {min, day} which is
# below the discrimination threshold. Adding the head measure token
# ("MVPA" vs "Sedentary") is what makes the match unambiguous.
_LOW_INFO_TOKENS = frozenset({
    "min", "day", "week", "month", "year", "days", "weeks", "months", "years",
    "hour", "hours", "second", "seconds",
    "min1", "week1", "weekly", "daily",
    "per", "of", "the", "a", "an", "in", "at", "to", "and", "or",
    "score", "scores", "value", "values", "level", "levels",
    "count", "counts", "number", "rate", "ratio",
    "change", "delta", "diff", "difference",
    "kcal", "kcla", "mets", "kg", "lb", "lbs", "m", "cm", "mm",
    "1", "2", "3", "4", "5",
})

# The 6 gold numeric fields and their A5 counterparts. The dict value is a
# tuple (a5_field_name, tolerance_type) where tolerance_type is one of:
#   "exact_int" -> exact integer match
#   "rel_x"     -> ±TOL_X_REL relative
#   "rel_sd"    -> ±TOL_SD_REL relative
FIELD_MAPPING: dict[str, tuple[str, str]] = {
    "N_GC":  ("n_control", "exact_int"),
    "N_GI":  ("n_intervention", "exact_int"),
    "X_GC":  ("mean_control", "rel_x"),
    "X_GI":  ("mean_intervention", "rel_x"),
    "SD_GC": ("sd_control", "rel_sd"),
    "SD_GI": ("sd_intervention", "rel_sd"),
}

# Coarse timepoint families. Each canonical bucket lists token markers found in
# free-form text on the predicted side. Used by `_normalize_timepoint` and
# `_timepoints_match` to bridge gold's uppercase labels (BASELINE / POST / FU*)
# with A5's free-form name text. Anything outside these buckets is treated as
# "unknown" and does NOT veto a numeric comparison (we record the mismatch but
# still compute the field-level tolerances).
TIMEPOINT_FAMILIES: dict[str, list[str]] = {
    "baseline": ["baseline", "pre-intervention", "pretest", "pre test", "t0"],
    "post": [
        "post", "post-intervention", "post intervention", "endpoint",
        "end of intervention", "end of treatment", "post-treatment", "t1",
        "immediate", "at end", "end-point",
    ],
    "followup": [
        "follow-up", "follow up", "followup", "fu", "month follow", "months follow",
        "week follow", "weeks follow", "long-term", "long term",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — gold subset construction
# ──────────────────────────────────────────────────────────────────────────────

def _is_canonical_covidence(cid: object) -> bool:
    """Return True if a covidence_id looks like a canonical study identifier.

    Canonical IDs start with '#'. Arm labels (E2, E3, GC ..., '#1354RCT 2') are
    handled separately. The '#1354RCT 2' edge case starts with '#' so it is
    treated as canonical by this rule, which matches its self-parent mapping.
    """
    return isinstance(cid, str) and cid.startswith("#")


def _filter_hero3_includes(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only sample rows tagged with 'hero3' in the `purpose` column."""
    purpose = sample_df["purpose"].fillna("").astype(str)
    mask = purpose.str.contains("hero3", case=False, na=False)
    return sample_df[mask].reset_index(drop=True)


def _build_arm_high_confidence_pairs(arm_df: pd.DataFrame) -> set[tuple[str, str]]:
    """Return the set of (arm_label, parent_covidence_id) pairs flagged 'high'.

    Any pair missing from this set should be ignored when filtering gold
    metrics rows. The 'unresolved' / 'medium' buckets are not currently
    populated in arm_to_parent_mapping.csv but we honour the contract.
    """
    high = arm_df[arm_df["confidence"].astype(str).str.lower() == "high"]
    pairs: set[tuple[str, str]] = set()
    for _, row in high.iterrows():
        pairs.add((str(row["arm_label"]), str(row["parent_covidence_id"])))
    return pairs


def _assign_effective_parents(
    gold_df: pd.DataFrame,
    high_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    """Add an `_effective_parent` column to gold_metrics.

    Logic:
      - Canonical rows (covidence_id starts with '#') are their own parent.
      - Non-canonical rows inherit the most recent preceding canonical
        covidence_id (forward fill). The mapping is consulted to confirm the
        (arm_label, parent) pair has confidence='high'; if not, the row's
        `_effective_parent` is set to None and the row will be excluded.
      - Rows preceding the first canonical id get `_effective_parent=None`.
    """
    parents: list[Optional[str]] = []
    last_canonical: Optional[str] = None
    for cid in gold_df["covidence_id"].tolist():
        cid_s = str(cid) if cid is not None else ""
        if _is_canonical_covidence(cid_s):
            last_canonical = cid_s
            parents.append(cid_s)
            continue
        if last_canonical is None:
            parents.append(None)
            continue
        # Non-canonical row: only accept if (arm_label, parent) is high-conf
        if (cid_s, last_canonical) in high_pairs:
            parents.append(last_canonical)
        else:
            parents.append(None)
    out = gold_df.copy()
    # dtype=object keeps None as None; pandas 3 would infer a string dtype and turn it into NaN
    out["_effective_parent"] = pd.Series(parents, index=out.index, dtype=object)
    return out


def _build_gold_subset(
    gold_df: pd.DataFrame,
    arm_df: pd.DataFrame,
    include_ids: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Return the gold_metrics rows whose effective parent is in `include_ids`.

    Returns:
        (subset_df, debug_info). `subset_df` keeps the original gold columns plus
        `_effective_parent`. `debug_info` exposes counters useful for the
        evaluator report.
    """
    high_pairs = _build_arm_high_confidence_pairs(arm_df)
    annotated = _assign_effective_parents(gold_df, high_pairs)
    include_set = set(include_ids)
    mask = annotated["_effective_parent"].isin(include_set)
    subset = annotated[mask].reset_index(drop=True)
    debug = {
        "total_gold_rows": int(len(gold_df)),
        "high_conf_mapping_pairs": int(len(high_pairs)),
        "subset_rows": int(len(subset)),
        "subset_by_parent": (
            subset["_effective_parent"].value_counts().to_dict() if len(subset) else {}
        ),
    }
    return subset, debug


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — outcome-name and timepoint matching
# ──────────────────────────────────────────────────────────────────────────────

_PUNC_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _normalize_outcome_name(name: object) -> str:
    """Lower-case, strip punctuation and collapse whitespace.

    Used as a coarse normaliser for outcome-name comparison. Keeps unit
    suffixes (e.g. "min/day") because they are often the distinguishing
    signal between similarly named outcomes.
    """
    s = str(name or "").lower().strip()
    s = s.translate(_PUNC_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _outcome_names_match(gold_name: str, pred_name: str) -> bool:
    """Heuristic match between a gold outcome_name and an A5 outcome `name`.

    Three-stage matching, conservative on purpose:
      1. Normalised string containment (either direction). This handles
         "MVPA (min/day)" <-> "MVPA min/day" / "MVPA min day".
      2. Token-overlap ratio against the smaller token set, threshold
         NAME_TOKEN_OVERLAP_THRESHOLD (0.6). Unit / common-noun tokens
         listed in `_LOW_INFO_TOKENS` are filtered out FIRST so that pairs
         like "MVPA min/day" vs "Sedentary min/day" do not match purely
         on shared unit tokens.
      3. Require at least one *informative* shared token (post-filter).
    """
    g = _normalize_outcome_name(gold_name)
    p = _normalize_outcome_name(pred_name)
    if not g or not p:
        return False
    if g == p:
        return True
    if g in p or p in g:
        return True
    g_tokens_all = set(g.split())
    p_tokens_all = set(p.split())
    if not g_tokens_all or not p_tokens_all:
        return False
    g_tokens = {t for t in g_tokens_all if t not in _LOW_INFO_TOKENS}
    p_tokens = {t for t in p_tokens_all if t not in _LOW_INFO_TOKENS}
    if not g_tokens or not p_tokens:
        return False
    informative_shared = g_tokens & p_tokens
    if not informative_shared:
        return False
    smaller = min(len(g_tokens), len(p_tokens))
    ratio = len(informative_shared) / smaller if smaller > 0 else 0.0
    return ratio >= NAME_TOKEN_OVERLAP_THRESHOLD


def _normalize_timepoint(label: object) -> str:
    """Map a free-form timepoint label to one of {baseline, post, followup, unknown}.

    Gold uses uppercase tokens (BASELINE / POST / FU3M / FU6M / ...). A5
    embeds timepoint hints inside `name`. This normaliser is intentionally
    permissive: if no marker is found we return "unknown" rather than
    invent a bucket.
    """
    s = str(label or "").lower().strip()
    if not s:
        return "unknown"
    if s == "baseline" or s.startswith("base"):
        return "baseline"
    if s == "post" or s.startswith("post"):
        return "post"
    if s.startswith("fu") or "follow" in s:
        return "followup"
    for bucket, markers in TIMEPOINT_FAMILIES.items():
        for m in markers:
            if m in s:
                return bucket
    return "unknown"


def _timepoints_match(gold_timepoint: str, predicted_name: str) -> bool:
    """True iff gold timepoint and A5 name agree on the canonical bucket.

    A5 lacks an explicit timepoint field, so the predicted side is derived
    from the outcome `name` text. If the predicted bucket is "unknown" we
    return True (do not veto): the evaluator can still compare numeric
    fields and the mismatch is recorded in details.
    """
    g_bucket = _normalize_timepoint(gold_timepoint)
    p_bucket = _normalize_timepoint(predicted_name)
    if p_bucket == "unknown":
        return True
    return g_bucket == p_bucket


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — numeric tolerance checks
# ──────────────────────────────────────────────────────────────────────────────

def _coerce_float(value: object) -> Optional[float]:
    """Coerce `value` to float; return None on NaN / null / unparseable input."""
    if value is None:
        return None
    if isinstance(value, float) and (value != value):  # NaN check
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value).strip()
        if not s or s.lower() in ("nan", "none", "n/a", "na", "null"):
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def _check_field(
    tol_type: str,
    gold_value: object,
    pred_value: object,
) -> tuple[bool, bool]:
    """Compare gold and predicted values according to `tol_type`.

    Returns ``(passed, missing)``. ``passed=True`` means the field is within
    tolerance. ``missing=True`` means at least one side is unparseable / NaN;
    in that case ``passed`` is False by definition (we cannot pass a field
    we cannot compare).
    """
    g = _coerce_float(gold_value)
    p = _coerce_float(pred_value)
    if g is None or p is None:
        return False, True
    if tol_type == "exact_int":
        try:
            return int(g) == int(p), False
        except (ValueError, OverflowError):
            return False, False
    if tol_type == "rel_x":
        rel = TOL_X_REL
    elif tol_type == "rel_sd":
        rel = TOL_SD_REL
    else:
        return False, False
    if g == 0:
        return abs(p - g) <= ABS_FALLBACK, False
    return abs(p - g) / abs(g) <= rel, False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — paper construction
# ──────────────────────────────────────────────────────────────────────────────

def _build_papers(
    includes_df: pd.DataFrame,
    gold_screening_df: pd.DataFrame,
    pdfs_dir: Path,
    errors: list,
    canonical_pdf_by_cid: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build paper dicts ready for `Agent5Extraction.extract_all`.

    Each dict carries: pmid (= covidence_id), title, abstract, authors, year,
    journal, doi, full_text, text_source='local_pdf'. Rows whose PDF cannot be
    extracted are skipped with an `errors[]` entry.

    When `canonical_pdf_by_cid` is provided (T16.D.9.C fix), the canonical
    `pdf_used` from `extracted_metadata.csv` overrides the per-sample
    `pdf_filename` whenever a non-null canonical value exists for the
    covidence_id. This avoids the protocol/appendix mismatch documented in
    T16.D.8.C (sample's pdf_filename pointed to orphan protocol PDFs for
    #13950 and #5298). When the canonical lookup yields None / NaN, the
    sample's `pdf_filename` is used as fallback so that covidence_ids
    recovered manually (e.g. #3490 in T16.D.3.bis) remain processable.
    """
    gold_idx = gold_screening_df.set_index("covidence_id")
    papers: list[dict] = []
    skipped: list[dict] = []
    canonical_pdf_by_cid = canonical_pdf_by_cid or {}

    for _, row in includes_df.iterrows():
        cid = str(row["covidence_id"])
        pdf_filename_sample = row.get("pdf_filename")
        has_pdf = bool(row.get("has_pdf", False))

        canonical_pdf = canonical_pdf_by_cid.get(cid)
        if canonical_pdf is not None and (
            isinstance(canonical_pdf, float) and pd.isna(canonical_pdf)
        ):
            canonical_pdf = None

        if canonical_pdf:
            if pdf_filename_sample and str(pdf_filename_sample) != str(canonical_pdf):
                errors.append({
                    "input_id": cid,
                    "error_type": "pdf_canonical_override",
                    "message": (
                        f"Canonical pdf_used ({canonical_pdf!r}) overrides "
                        f"sample pdf_filename ({pdf_filename_sample!r}). "
                        "T16.D.9.C fix: prefer extracted_metadata.csv.pdf_used."
                    ),
                })
            pdf_filename = canonical_pdf
            has_pdf = True
        else:
            pdf_filename = pdf_filename_sample

        if not has_pdf or pdf_filename is None or (
            isinstance(pdf_filename, float) and pd.isna(pdf_filename)
        ):
            errors.append({
                "input_id": cid,
                "error_type": "no_pdf_available",
                "message": (
                    f"Sample row marked has_pdf=False (pdf_filename={pdf_filename!r}); "
                    "skipped from extraction evaluation."
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
                    "skipped; downstream numeric extraction may be unreliable."
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
            # Optional columns — gold_screening_unified may or may not carry them
            for src_key, dst_setter in (
                ("authors", lambda v: v),
                ("year", lambda v: v),
                ("journal", lambda v: v),
                ("doi", lambda v: v),
            ):
                if src_key in gold_row.index:
                    val = gold_row.get(src_key)
                    if val is not None and not (isinstance(val, float) and pd.isna(val)):
                        s = str(val).strip()
                        if s.lower() not in ("nan", "none"):
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
# Helpers — metrics aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_by(category_pairs: list[tuple[str, bool]]) -> dict[str, dict]:
    """Aggregate `[(category, passed)]` pairs into ``{cat: {correct,total,pct_correct}}``."""
    out: dict[str, dict] = {}
    for cat, passed in category_pairs:
        if cat not in out:
            out[cat] = {"correct": 0, "total": 0, "pct_correct": 0.0}
        out[cat]["total"] += 1
        if passed:
            out[cat]["correct"] += 1
    for cat, stats in out.items():
        stats["pct_correct"] = (
            round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def run_hero3_evaluation(
    config_path: Path,
    output_dir: Path,
    agent_instance=None,
) -> dict:
    """Run the Hero 3 evaluation and return the result dict (also written to disk).

    Args:
        config_path: Path to the eval YAML config.
        output_dir: Directory where ``hero3_results_*.json`` will be written.
        agent_instance: Optional pre-instantiated ``Agent5Extraction`` whose model
            is already loaded (used by ``run_heroes_micro.py`` to share VRAM
            across heroes). When ``None``, a new instance is created and its
            VRAM is freed before returning (standalone mode).
    """
    _seed_torch(42)

    eval_cfg = load_eval_config(config_path)
    sample_path = (PROJECT_ROOT / eval_cfg["sample"]).resolve()
    gold_screening_path = (PROJECT_ROOT / eval_cfg["gold_screening"]).resolve()
    gold_metrics_path = (PROJECT_ROOT / eval_cfg["gold_metrics"]).resolve()
    arm_mapping_path = (PROJECT_ROOT / eval_cfg["arm_to_parent_mapping"]).resolve()
    pdfs_dir = (PROJECT_ROOT / eval_cfg["pdfs_dir"]).resolve()

    # ── 1. Load CSVs ───────────────────────────────────────────────────────
    sample_df = pd.read_csv(sample_path, dtype={"covidence_id": str})
    gold_screening_df = pd.read_csv(gold_screening_path, dtype={"covidence_id": str})
    gold_metrics_df = pd.read_csv(gold_metrics_path, dtype={"covidence_id": str})
    arm_df = pd.read_csv(arm_mapping_path, dtype=str)

    # T16.D.9.C fix: optional canonical PDF mapping from extracted_metadata.csv
    canonical_pdf_by_cid: dict[str, str] = {}
    extracted_meta_cfg = eval_cfg.get("extracted_metadata")
    if extracted_meta_cfg:
        extracted_meta_path = (PROJECT_ROOT / extracted_meta_cfg).resolve()
        if extracted_meta_path.exists():
            em_df = pd.read_csv(extracted_meta_path, dtype={"covidence_id": str})
            if "pdf_used" in em_df.columns:
                for _, em_row in em_df.iterrows():
                    cid = str(em_row.get("covidence_id", "") or "").strip()
                    pdf_used = em_row.get("pdf_used")
                    if cid and pdf_used is not None and not (
                        isinstance(pdf_used, float) and pd.isna(pdf_used)
                    ):
                        canonical_pdf_by_cid[cid] = str(pdf_used)

    includes_df = _filter_hero3_includes(sample_df)
    include_ids = includes_df["covidence_id"].astype(str).tolist()
    sample_size = int(len(includes_df))

    errors: list = []
    papers, skipped = _build_papers(
        includes_df, gold_screening_df, pdfs_dir, errors,
        canonical_pdf_by_cid=canonical_pdf_by_cid,
    )
    eval_n = len(papers)

    # ── 2. Build the gold subset (direct + arm-mapped) ─────────────────────
    gold_subset, gold_debug = _build_gold_subset(gold_metrics_df, arm_df, include_ids)

    if eval_n == 0:
        return _emit_empty_envelope(
            eval_cfg=eval_cfg,
            sample_size=sample_size,
            output_dir=output_dir,
            errors=errors,
            notes=(
                "No se pudo procesar ningún PDF: ningún hero3 include tenía "
                "PDF utilizable. Verifica `has_pdf` y la existencia de los "
                "ficheros en pdfs_dir."
            ),
        )

    # ── 3. Agent5 setup ────────────────────────────────────────────────────
    from agents.agent5_extraction import Agent5Extraction

    a5_params = load_agent_params("agent5_extraction")
    owns_agent = agent_instance is None
    if agent_instance is None:
        agent = Agent5Extraction()
    else:
        agent = agent_instance

    # Agent5 hardcodes temperature=0.1 inside `_extract_once`. Record the
    # effective values rather than overrides we cannot inject.
    llm_config = {
        "temperature": 0.1,
        "top_p": getattr(agent, "top_p", None),
        "max_tokens": 600,
        "repetition_penalty": a5_params.get("repetition_penalty", 1.3),
        "seed": 42,
        "dual_extraction": agent.dual_extraction,
        "concordance_threshold_numeric": agent.threshold_numeric,
        "concordance_threshold_sample": agent.threshold_sample,
    }

    model_used = agent.model_path or "unknown"
    if isinstance(model_used, str) and "qwen" in model_used.lower():
        model_display = "Qwen2.5-7B-Instruct (4-bit BitsAndBytes)"
    else:
        model_display = str(model_used)

    # ── 4. Run extraction ──────────────────────────────────────────────────
    start_ts = time.time()
    try:
        extract_out = agent.extract_all(papers, PAULA_SPORT_PROTOCOL)
    except Exception as e:
        errors.append({
            "input_id": "<batch>",
            "error_type": "ModelError",
            "message": f"extract_all raised: {type(e).__name__}: {e}",
        })
        extract_out = {"papers": [], "stats": {}}
    elapsed = round(time.time() - start_ts, 2)

    a5_by_id: dict[str, dict] = {}
    for p in extract_out.get("papers", []):
        pid = str(p.get("pmid", ""))
        if pid:
            a5_by_id[pid] = p

    # Surface any A5 papers that failed entirely as errors[]
    for paper in papers:
        cid = paper["pmid"]
        extracted = a5_by_id.get(cid)
        if extracted is None:
            errors.append({
                "input_id": cid,
                "error_type": "extraction_failed",
                "message": "Agent5 returned no entry for this covidence_id.",
            })
        elif extracted.get("extraction_quality") == "failed":
            errors.append({
                "input_id": cid,
                "error_type": "extraction_failed",
                "message": (
                    "Agent5 marked extraction_quality='failed' "
                    f"(notes={extracted.get('extraction_notes', '')!r})."
                ),
            })

    # ── 5. Walk the gold subset and score each triple ──────────────────────
    raw_predictions: list = []
    field_pass_pairs: list[tuple[str, bool]] = []          # (field_type, passed)
    cat_pairs: list[tuple[str, bool]] = []                 # (category, passed)
    timepoint_pairs: list[tuple[str, bool]] = []           # (timepoint, passed)

    n_outcome_exact = 0
    n_outcome_partial = 0
    n_outcome_no_match = 0
    total_fields_expected = 0
    total_fields_passed = 0
    total_fields_failed = 0
    total_fields_missing = 0

    for _, gold_row in gold_subset.iterrows():
        parent = str(gold_row["_effective_parent"])
        gold_outcome_name = str(gold_row.get("outcome_name") or "").strip()
        gold_timepoint = str(gold_row.get("timepoint") or "").strip()
        gold_arm_label = str(gold_row.get("covidence_id") or "").strip()
        outcome_category_raw = (
            str(gold_row.get("instrument_type") or "").strip().lower()
        )
        if "object" in outcome_category_raw:
            outcome_category = "objective"
        elif "self" in outcome_category_raw or "report" in outcome_category_raw:
            outcome_category = "self_report"
        else:
            outcome_category = outcome_category_raw or "unknown"

        gold_values = {
            field: gold_row.get(field) for field in FIELD_MAPPING
        }

        input_id = f"{parent}::{gold_outcome_name}::{gold_timepoint}"

        extracted = a5_by_id.get(parent)
        a5_outcomes: list[dict] = []
        if extracted is not None:
            a5_outcomes = extracted.get("outcomes") or []

        # Find candidate outcomes by name (and optionally timepoint).
        candidates: list[tuple[int, dict, bool]] = []
        for idx, outcome in enumerate(a5_outcomes):
            pred_name = outcome.get("name", "")
            if _outcome_names_match(gold_outcome_name, pred_name):
                tp_ok = _timepoints_match(gold_timepoint, pred_name)
                candidates.append((idx, outcome, tp_ok))

        chosen: Optional[dict] = None
        chosen_idx: Optional[int] = None
        tp_match: Optional[bool] = None
        if candidates:
            # Prefer a candidate whose timepoint bucket agrees with gold.
            tp_preferred = [c for c in candidates if c[2]]
            pick = tp_preferred[0] if tp_preferred else candidates[0]
            chosen_idx, chosen, tp_match = pick

        pred_values: dict = {f: None for f in FIELD_MAPPING}
        failed_fields: list[str] = []
        field_results: dict[str, dict] = {}
        any_passed = False
        any_failed = False

        if chosen is None:
            outcome_match = "no_match"
            n_outcome_no_match += 1
            for field, (a5_field, tol_type) in FIELD_MAPPING.items():
                _passed, _missing = _check_field(tol_type, gold_values[field], None)
                field_results[field] = {
                    "gold": gold_values[field],
                    "predicted": None,
                    "passed": False,
                    "missing": True,
                    "tol_type": tol_type,
                }
                # Even on no_match we must still account for the field counts so
                # `pct_fields_correct_overall` is denominator-stable across runs.
                field_type = field[:2] if field.startswith("SD") else field[:1]  # N/X/SD
                field_pass_pairs.append((field_type, False))
                cat_pairs.append((outcome_category, False))
                timepoint_pairs.append((gold_timepoint, False))
                total_fields_expected += 1
                total_fields_missing += 1
                failed_fields.append(field)
        else:
            n_passed_here = 0
            n_failed_here = 0
            n_missing_here = 0
            for field, (a5_field, tol_type) in FIELD_MAPPING.items():
                pred = chosen.get(a5_field)
                # mean_intervention/sd_intervention paths fall back to n_per_group
                if pred is None and a5_field in ("n_intervention", "n_control"):
                    n_per_group = chosen.get("n_per_group") or {}
                    if isinstance(n_per_group, dict):
                        if a5_field == "n_intervention":
                            pred = n_per_group.get("intervention")
                        else:
                            pred = n_per_group.get("control")
                pred_values[field] = pred
                passed, missing = _check_field(tol_type, gold_values[field], pred)
                field_results[field] = {
                    "gold": gold_values[field],
                    "predicted": pred,
                    "passed": passed,
                    "missing": missing,
                    "tol_type": tol_type,
                }
                field_type = field[:2] if field.startswith("SD") else field[:1]
                field_pass_pairs.append((field_type, passed))
                cat_pairs.append((outcome_category, passed))
                timepoint_pairs.append((gold_timepoint, passed))
                total_fields_expected += 1
                if passed:
                    total_fields_passed += 1
                    n_passed_here += 1
                    any_passed = True
                else:
                    if missing:
                        total_fields_missing += 1
                        n_missing_here += 1
                    else:
                        total_fields_failed += 1
                        n_failed_here += 1
                    failed_fields.append(field)
                    any_failed = True

            # Decide outcome match label
            if n_passed_here == len(FIELD_MAPPING):
                outcome_match = "exact"
                n_outcome_exact += 1
            elif any_passed:
                outcome_match = "partial"
                n_outcome_partial += 1
            else:
                outcome_match = "no_match"
                n_outcome_no_match += 1

        gold_payload = {
            "covidence_id_row": gold_arm_label,
            "parent_covidence_id": parent,
            "outcome_name": gold_outcome_name,
            "timepoint": gold_timepoint,
            "outcome_category": outcome_category,
            "instrument": (
                str(gold_row.get("instrument") or "").strip()
                if gold_row.get("instrument") is not None else ""
            ),
            "N_GC": gold_values["N_GC"],
            "N_GI": gold_values["N_GI"],
            "X_GC": gold_values["X_GC"],
            "X_GI": gold_values["X_GI"],
            "SD_GC": gold_values["SD_GC"],
            "SD_GI": gold_values["SD_GI"],
        }
        predicted_payload: Optional[dict]
        if chosen is None:
            predicted_payload = None
        else:
            predicted_payload = {
                "name": chosen.get("name"),
                "measure_type": chosen.get("measure_type") or chosen.get("effect_size_type"),
                "n_intervention": pred_values["N_GI"],
                "n_control": pred_values["N_GC"],
                "mean_intervention": pred_values["X_GI"],
                "mean_control": pred_values["X_GC"],
                "sd_intervention": pred_values["SD_GI"],
                "sd_control": pred_values["SD_GC"],
                "numeric_complete": chosen.get("numeric_complete"),
                "matched_outcome_index": chosen_idx,
            }

        raw_response_excerpt = (
            extracted.get("extraction_notes", "")
            if extracted is not None else ""
        )

        raw_predictions.append({
            "input_id": input_id,
            "gold": gold_payload,
            "predicted": predicted_payload,
            "match": outcome_match == "exact",
            "details": {
                "outcome_match": outcome_match,
                "timepoint_match": tp_match,
                "failed_fields": failed_fields,
                "field_scores": field_results,
                "extraction_quality": (
                    extracted.get("extraction_quality") if extracted else "n/a"
                ),
                "sections_used": (
                    extracted.get("sections_used", []) if extracted else []
                ),
                "raw_llm_response": raw_response_excerpt,
            },
        })

    # ── 6. Aggregate metrics ───────────────────────────────────────────────
    by_field_type = _aggregate_by(field_pass_pairs)
    by_outcome_category = _aggregate_by(cat_pairs)
    by_timepoint = _aggregate_by(timepoint_pairs)

    pct_overall = (
        round(total_fields_passed / total_fields_expected, 4)
        if total_fields_expected > 0 else 0.0
    )

    # Compact extractions_detail: per-triple summary mirroring schema 3.3.
    extractions_detail = []
    for entry in raw_predictions:
        extractions_detail.append({
            "input_id": entry["input_id"],
            "covidence_id": entry["gold"]["parent_covidence_id"],
            "outcome_name": entry["gold"]["outcome_name"],
            "timepoint": entry["gold"]["timepoint"],
            "outcome_match": entry["details"]["outcome_match"],
            "gold_values": {
                k: entry["gold"][k] for k in ("N_GC", "N_GI", "X_GC", "X_GI", "SD_GC", "SD_GI")
            },
            "pred_values": (
                {
                    k: (entry["predicted"] or {}).get(k) for k in (
                        "N_GC", "N_GI", "X_GC", "X_GI", "SD_GC", "SD_GI",
                    )
                } if entry["predicted"] else None
            ),
            "failed_fields": entry["details"]["failed_fields"],
        })

    metrics = {
        "pct_fields_correct_overall": pct_overall,
        "fields_total": total_fields_expected,
        "fields_correct": total_fields_passed,
        "fields_failed": total_fields_failed,
        "fields_missing": total_fields_missing,
        "outcome_match_counts": {
            "exact": n_outcome_exact,
            "partial": n_outcome_partial,
            "no_match": n_outcome_no_match,
        },
        "by_field_type": by_field_type,
        "by_outcome_category": by_outcome_category,
        "by_timepoint": by_timepoint,
        "extractions_detail": extractions_detail,
        "gold_subset_debug": gold_debug,
        "notes": (
            "Tolerancias D3: N exact, X ±5%, SD ±10%. "
            "Outcome name matching es heurístico (substring bidireccional + token overlap >= 0.6). "
            "Timepoint match no veta la comparación numérica cuando A5 no emite timepoint "
            "(predicción 'unknown' siempre se acepta). Hero 3 es regresión: "
            "confusion_matrix = null."
        ),
    }

    notes = (
        "Hero 3 evalúa Agent5.extract_all sobre los 10 includes hero3 del "
        "sample_50. La cobertura gold se construye combinando covidence_ids "
        f"directos ({sum(1 for cid in gold_subset['covidence_id'] if _is_canonical_covidence(cid))} "
        "filas canónicas) más arm-labels mapeadas con confidence='high' "
        f"({sum(1 for cid in gold_subset['covidence_id'] if not _is_canonical_covidence(cid))} "
        "filas de brazos). Tamaño subset gold: "
        f"{len(gold_subset)} filas. A5 produjo "
        f"{len(extract_out.get('papers', []))} estudios extraídos."
    )

    # ── 7. Final JSON envelope ─────────────────────────────────────────────
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
        "confusion_matrix": None,             # Hero 3 is regression, not classification
        "raw_predictions": raw_predictions,
        "errors": errors,
        "notes": notes,
    }

    # ── 8. Persist JSON ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"hero3_results_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    _print_short_report(result, out_path, skipped=skipped)

    # Free VRAM before exit (Agent5 holds the model after extract_all).
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
            "temperature": 0.1,
            "top_p": None,
            "max_tokens": 600,
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
            "pct_fields_correct_overall": 0.0,
            "fields_total": 0,
            "fields_correct": 0,
            "fields_failed": 0,
            "fields_missing": 0,
            "outcome_match_counts": {"exact": 0, "partial": 0, "no_match": 0},
            "by_field_type": {},
            "by_outcome_category": {},
            "by_timepoint": {},
            "extractions_detail": [],
            "gold_subset_debug": {},
            "notes": "No papers could be evaluated.",
        },
        "confusion_matrix": None,
        "raw_predictions": [],
        "errors": errors,
        "notes": notes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"hero3_results_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[WARN] Hero 3 emitted empty envelope -> {out_path}")
    return result


def _print_short_report(result: dict, out_path: Path, skipped: list | None = None) -> None:
    """Print a brief Spanish summary to stdout."""
    m = result["metrics"]
    skipped = skipped or []
    print()
    print("=" * 64)
    print("HERO 3 (A5 — Extracción de Datos): RESUMEN")
    print("=" * 64)
    print(f"Sample size (includes) : {result['sample_size']}")
    print(f"Papers evaluados       : {result.get('eval_size', 0)}")
    print(f"Saltados               : {len(skipped)}")
    print(f"Modelo                 : {result['model_used']}")
    print(f"Tiempo total           : {result['elapsed_seconds']}s")
    print()
    print(f"% campos correctos     : {m.get('pct_fields_correct_overall')}")
    print(f"Campos totales         : {m.get('fields_total')}")
    print(f"  Correctos            : {m.get('fields_correct')}")
    print(f"  Fallidos             : {m.get('fields_failed')}")
    print(f"  Ausentes             : {m.get('fields_missing')}")
    print()
    counts = m.get("outcome_match_counts", {})
    print(
        "Matches por outcome    : "
        f"exact={counts.get('exact', 0)} / partial={counts.get('partial', 0)} / "
        f"no_match={counts.get('no_match', 0)}"
    )
    bft = m.get("by_field_type", {})
    if bft:
        print()
        print("Desglose por tipo de campo:")
        for k in ("N", "X", "SD"):
            stats = bft.get(k, {})
            if stats:
                print(
                    f"   {k:<3}: {stats.get('correct', 0)}/{stats.get('total', 0)} "
                    f"({stats.get('pct_correct', 0.0)})"
                )
    debug = m.get("gold_subset_debug", {})
    if debug:
        print()
        print(
            "Subset gold construido: "
            f"{debug.get('subset_rows', 0)} filas "
            f"({debug.get('total_gold_rows', 0)} totales; "
            f"{debug.get('high_conf_mapping_pairs', 0)} pares high-confidence)."
        )
    if skipped:
        print()
        print("Skipped papers:")
        for s in skipped[:5]:
            print(f"   - {s.get('covidence_id')} ({s.get('reason')})")
    print(f"\nOutput JSON guardado en: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hero 3 evaluator — Agent5 extraction vs Paula gold standard."
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
        help="Output directory for hero3_results_*.json (default: outputs/eval/micro/).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()

    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2

    run_hero3_evaluation(config_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
