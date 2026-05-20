"""T17.B.1 evaluator — Agent4 full-text screening (reporter/decider split).

Sprint:   T17.B.1 (Phase 3 — smoke; Phase 4 — full dev sweep)
Schema:   docs/iterations/T17.B.1_hypothesis.md
Inputs:
  - experiments/T17.B.1/partition.json
        Stratified 70/30 dev/heldout split over the 553-paper baseline corpus.
  - data/gold_standard/deporte/samples/medio/sample_medio_a4.csv
        Per-paper has_pdf / pdf_filename / human_label_ft (=gold_decision).
  - data/gold_standard/deporte/unified/gold_screening_unified.csv
        title + abstract per covidence_id.
  - data/gold_standard/deporte/pdfs_todos/*.pdf
        Local PDFs (same set used by hero2/T16.E.2.B).
  - agents/prompts.py
        AGENT4_REPORTER_TEMPLATE_V1/V2 + AGENT4_DECIDER_TEMPLATE_V1/V3/V4.

Output:
  - experiments/T17.B.1/runs/<variant>_<split>_<UTC>.json
        Per-paper records + aggregate metrics + cache hit telemetry.
  - experiments/T17.B.1/cache/{reporter,decider}/<key>.json
        Per-call cache entries (atomic write via .tmp + os.replace).

Variants:
  V1 = Reporter V1 + Decider V1   (sanity, equivalent to T17.B.1 lock-in defaults)
  V2 = Reporter V2 + Decider V1   (reporter emits exclusion_signals)
  V3 = Reporter V1 + Decider V3   (decider receives chain-of-thought prelude)
  V4 = Reporter V1 + Decider V4   (decider with stricter exclusion criteria)

The evaluator does NOT touch agents/agent4_fulltext.py, agents/agent4_fulltext_split.py,
agents/prompts.py, app.py, api.py, config.yaml or config/agents.json (read-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Ensure project root on sys.path before importing `agents` or `experiments`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._eval_common import (  # noqa: E402
    PAULA_SPORT_PROTOCOL,
    extract_pdf_text as _extract_pdf_text_shared,
    gpu_info as _gpu_info,
    load_agent_params,
    seed_torch as _seed_torch,
    truncate as _truncate,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PARTITION = PROJECT_ROOT / "experiments" / "T17.B.1" / "partition.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "T17.B.1" / "runs"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "experiments" / "T17.B.1" / "cache"
DEFAULT_SAMPLE_CSV = (
    PROJECT_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "samples"
    / "medio"
    / "sample_medio_a4.csv"
)
DEFAULT_GOLD_CSV = (
    PROJECT_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "unified"
    / "gold_screening_unified.csv"
)
DEFAULT_PDFS_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_todos"

VARIANT_TEMPLATES = {
    "V1": ("AGENT4_REPORTER_TEMPLATE_V1", "AGENT4_DECIDER_TEMPLATE_V1"),
    "V2": ("AGENT4_REPORTER_TEMPLATE_V2", "AGENT4_DECIDER_TEMPLATE_V1"),
    "V3": ("AGENT4_REPORTER_TEMPLATE_V1", "AGENT4_DECIDER_TEMPLATE_V3"),
    "V4": ("AGENT4_REPORTER_TEMPLATE_V1", "AGENT4_DECIDER_TEMPLATE_V4"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — partition + sub-sampling
# ──────────────────────────────────────────────────────────────────────────────

def _load_partition(partition_path: Path) -> dict[str, Any]:
    if not partition_path.is_file():
        raise FileNotFoundError(f"Partition file not found: {partition_path}")
    with open(partition_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("dev", "heldout"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"Partition missing '{key}' list: {partition_path}")
    return data


def _load_sample_index(sample_csv: Path) -> dict[str, dict[str, Any]]:
    """Index sample_medio_a4.csv by covidence_id.

    Provides per-paper gold_decision (human_label_ft), gold_reason, has_pdf,
    pdf_filename — i.e. the metadata we need to drive Phase B without Phase A.
    """
    df = pd.read_csv(sample_csv, dtype={"covidence_id": str})
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        cid = str(row["covidence_id"]).strip()
        out[cid] = {
            "covidence_id": cid,
            "gold_decision": str(row.get("human_label_ft", "") or "").strip().lower(),
            "gold_reason_raw": row.get("exclusion_reason_ft"),
            "has_pdf": bool(row.get("has_pdf", False)),
            "pdf_filename": row.get("pdf_filename"),
        }
    return out


def _load_gold_index(gold_csv: Path) -> dict[str, dict[str, Any]]:
    """Index gold_screening_unified.csv by covidence_id (title + abstract)."""
    df = pd.read_csv(gold_csv, dtype={"covidence_id": str})
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        cid = str(row["covidence_id"]).strip()
        title = str(row.get("title", "") or "").strip()
        abstract = str(row.get("abstract", "") or "").strip()
        if title.lower() in ("nan", "none"):
            title = ""
        if abstract.lower() in ("nan", "none"):
            abstract = ""
        out[cid] = {
            "title": title,
            "abstract": abstract,
            "year": row.get("year"),
            "authors": str(row.get("authors", "") or "").strip(),
            "doi": str(row.get("doi", "") or "").strip(),
        }
    return out


def _stratified_subset(
    target_ids: list[str],
    sample_index: dict[str, dict[str, Any]],
    max_samples: int,
    seed: int,
) -> list[str]:
    """Return a stratified subset of size <= max_samples.

    Stratifies by binary gold_decision (include / exclude) preserving the
    natural prevalence of `target_ids`. When the proportional split would
    yield < 2 INCLUDE samples, the include count is bumped to 2 (assuming
    at least 2 includes exist in the pool); this guarantees the smoke run
    exercises both classes even at very small N.
    """
    rng = random.Random(seed)

    includes = [
        cid for cid in target_ids
        if sample_index.get(cid, {}).get("gold_decision") == "include"
    ]
    excludes = [
        cid for cid in target_ids
        if sample_index.get(cid, {}).get("gold_decision") == "exclude"
    ]

    if max_samples >= len(target_ids):
        return list(target_ids)

    n_total = max(0, min(max_samples, len(target_ids)))
    if n_total == 0:
        return []

    # Proportional allocation by binary class.
    p_include = len(includes) / max(1, len(target_ids))
    n_include = int(round(n_total * p_include))
    n_include = min(n_include, len(includes))
    n_exclude = n_total - n_include
    n_exclude = min(n_exclude, len(excludes))

    # Minimum-include floor (only when feasible).
    min_include = 2
    if n_include < min_include and len(includes) >= min_include and n_total >= min_include:
        n_include = min_include
        n_exclude = n_total - n_include
        n_exclude = min(n_exclude, len(excludes))

    # Re-balance if exclude pool exhausted (corner case).
    if n_include + n_exclude < n_total:
        deficit = n_total - (n_include + n_exclude)
        bonus_include = min(deficit, len(includes) - n_include)
        n_include += bonus_include

    sampled_includes = rng.sample(includes, n_include) if n_include > 0 else []
    sampled_excludes = rng.sample(excludes, n_exclude) if n_exclude > 0 else []
    subset = sampled_includes + sampled_excludes
    rng.shuffle(subset)
    return subset


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — corpus loading
# ──────────────────────────────────────────────────────────────────────────────

def _build_paper(
    cid: str,
    sample_row: dict[str, Any],
    gold_row: dict[str, Any] | None,
    pdfs_dir: Path,
    errors: list,
) -> Optional[dict[str, Any]]:
    """Build one paper dict ready for the split agent (Phase B only).

    Returns None and appends to `errors` when the PDF is unavailable or
    cannot be extracted. The dict mirrors the schema Agent4FullText (and its
    Split subclass) consumes — `text_source='local_pdf'` short-circuits the
    HTTP cascade entirely.
    """
    if not sample_row.get("has_pdf"):
        errors.append({
            "covidence_id": cid,
            "error_type": "no_pdf_available",
            "message": "Sample row has has_pdf=False; skipped from evaluation.",
        })
        return None

    pdf_filename = sample_row.get("pdf_filename")
    if pdf_filename is None or (
        isinstance(pdf_filename, float) and pd.isna(pdf_filename)
    ):
        errors.append({
            "covidence_id": cid,
            "error_type": "no_pdf_filename",
            "message": "pdf_filename missing from sample row; skipped.",
        })
        return None

    pdf_path = pdfs_dir / str(pdf_filename)
    text, err = _extract_pdf_text_shared(pdf_path)
    if text is None:
        errors.append({
            "covidence_id": cid,
            "error_type": "pdf_extraction_failed",
            "message": f"PDF '{pdf_filename}' extraction failed: {err}",
        })
        return None

    title = (gold_row or {}).get("title", "") or ""
    abstract = (gold_row or {}).get("abstract", "") or ""
    year_raw = (gold_row or {}).get("year")
    try:
        year = str(int(float(year_raw))) if year_raw not in (None, "", "nan") else ""
    except (TypeError, ValueError):
        year = str(year_raw or "")

    paper = {
        # Agent4 uses `pmid` as the in-memory primary key; we mirror covidence_id
        # there so the rest of the pipeline stays oblivious to source.
        "pmid": cid,
        "covidence_id": cid,
        "title": title,
        "abstract": abstract,
        "full_text": text,
        "full_text_length": len(text),
        "text_source": "local_pdf",
        "year": year,
        "journal": "",
        "authors": (gold_row or {}).get("authors", ""),
        "doi": (gold_row or {}).get("doi", ""),
        # Evaluator-internal metadata (ignored by the agent):
        "_gold_decision": sample_row.get("gold_decision", ""),
        "_gold_reason_raw": (
            ""
            if sample_row.get("gold_reason_raw") is None
            else str(sample_row.get("gold_reason_raw"))
        ),
        "_pdf_filename": str(pdf_filename),
    }
    return paper


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — cache (two-level: reporter then decider)
# ──────────────────────────────────────────────────────────────────────────────

def _cache_key_to_path(cache_dir: Path, level: str, key: str) -> Path:
    """Return the per-entry JSON path under `cache_dir/level/`.

    Uses `__` as the separator inside the key to avoid filesystem oddities
    with `::` (which Linux allows but trips up some tooling).
    """
    return cache_dir / level / f"{key}.json"


def _read_cache_entry(cache_dir: Path, level: str, key: str) -> Optional[dict[str, Any]]:
    path = _cache_key_to_path(cache_dir, level, key)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache_entry(cache_dir: Path, level: str, key: str, payload: dict[str, Any]) -> None:
    path = _cache_key_to_path(cache_dir, level, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _hash16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _sanitize_id_for_filename(cid: str) -> str:
    """Strip `#` and any path-separator chars from a covidence_id for filenames."""
    safe = cid.replace("#", "").replace("/", "_").replace("\\", "_")
    safe = safe.replace(":", "_").replace(" ", "_")
    return safe or "unknown"


def _screen_with_cache(
    agent,
    paper: dict[str, Any],
    protocol: dict[str, Any],
    cache_dir: Path,
    variant: str,
    no_cache: bool,
) -> dict[str, Any]:
    """Run reporter -> decider with per-stage on-disk cache.

    Mirrors `Agent4FullTextSplit._screen_single_paper_split` so the on-disk
    semantics (parse-error fallback = conservative EXCLUDE) match the production
    code path. The only deviation: we cache the raw LLM output keyed by
    `(covidence_id, prompt_hash)`.
    """
    cid = paper.get("covidence_id") or paper.get("pmid") or "unknown"
    safe_cid = _sanitize_id_for_filename(str(cid))
    from_cache = {"reporter": False, "decider": False}

    # ----- Reporter -----
    reporter_prompt = agent._build_reporter_prompt(paper, protocol)
    reporter_hash = _hash16(reporter_prompt)
    reporter_key = f"{safe_cid}__{reporter_hash}"

    cached = None if no_cache else _read_cache_entry(cache_dir, "reporter", reporter_key)
    if cached is not None:
        reporter_raw = cached.get("raw", "")
        reporter_json = cached.get("reporter_json")
        reporter_ok = bool(cached.get("ok", False))
        from_cache["reporter"] = True
    else:
        reporter_raw = agent._generate_split(
            reporter_prompt,
            max_tokens=agent.reporter_max_tokens,
            temperature=agent.reporter_temperature,
            repetition_penalty=agent.reporter_repetition_penalty,
        )
        reporter_json = agent._parse_reporter(reporter_raw)
        reporter_ok = reporter_json is not None
        if not no_cache:
            _write_cache_entry(cache_dir, "reporter", reporter_key, {
                "covidence_id": cid,
                "variant": variant,
                "prompt_hash": reporter_hash,
                "raw": reporter_raw,
                "reporter_json": reporter_json,
                "ok": reporter_ok,
                "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

    if not reporter_ok:
        return {
            "fulltext_decision": "EXCLUDE",
            "reporter_ok": False,
            "decider_ok": False,
            "confidence": "low",
            "reasoning": "reporter parse failed",
            "exclusion_reason": "reporter_parse_error",
            "reporter": None,
            "decider": None,
            "reporter_raw": reporter_raw,
            "decider_raw": "",
            "screening_confidence": 0.0,
            "borderline": True,
            "from_cache": from_cache,
        }

    # ----- Decider -----
    decider_prompt = agent._build_decider_prompt(reporter_json, protocol)
    decider_hash = _hash16(decider_prompt)
    decider_key = f"{safe_cid}__{reporter_hash}__{decider_hash}"

    cached = None if no_cache else _read_cache_entry(cache_dir, "decider", decider_key)
    if cached is not None:
        decider_raw = cached.get("raw", "")
        decider_json = cached.get("decider_json")
        decider_ok = bool(cached.get("ok", False))
        from_cache["decider"] = True
    else:
        decider_raw = agent._generate_split(
            decider_prompt,
            max_tokens=agent.decider_max_tokens,
            temperature=agent.decider_temperature,
            repetition_penalty=agent.decider_repetition_penalty,
        )
        decider_json = agent._parse_decider(decider_raw)
        decider_ok = decider_json is not None
        if not no_cache:
            _write_cache_entry(cache_dir, "decider", decider_key, {
                "covidence_id": cid,
                "variant": variant,
                "reporter_prompt_hash": reporter_hash,
                "prompt_hash": decider_hash,
                "raw": decider_raw,
                "decider_json": decider_json,
                "ok": decider_ok,
                "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

    if not decider_ok:
        return {
            "fulltext_decision": "EXCLUDE",
            "reporter_ok": True,
            "decider_ok": False,
            "confidence": "low",
            "reasoning": "decider parse failed",
            "exclusion_reason": "decider_parse_error",
            "reporter": reporter_json,
            "decider": None,
            "reporter_raw": reporter_raw,
            "decider_raw": decider_raw,
            "screening_confidence": 0.0,
            "borderline": True,
            "from_cache": from_cache,
        }

    decision = decider_json["decision"]
    confidence = decider_json.get("confidence", "medium")
    reason = decider_json.get("reason", "")

    if confidence == "high":
        screening_confidence = 1.0
    elif confidence == "medium":
        screening_confidence = 0.6
    else:
        screening_confidence = 0.3
    borderline = confidence == "low"

    return {
        "fulltext_decision": decision,
        "reporter_ok": True,
        "decider_ok": True,
        "confidence": confidence,
        "reasoning": reason,
        "exclusion_reason": reason if decision == "EXCLUDE" else "",
        "reporter": reporter_json,
        "decider": decider_json,
        "reporter_raw": reporter_raw,
        "decider_raw": decider_raw,
        "screening_confidence": screening_confidence,
        "borderline": borderline,
        "from_cache": from_cache,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — metrics
# ──────────────────────────────────────────────────────────────────────────────

def _compute_binary_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute binary metrics treating gold=include as the positive class.

    Records where both reporter_ok and decider_ok are True are counted as
    "valid"; parse failures still get a predicted_decision (EXCLUDE by the
    conservative fallback policy) and ARE included in the metrics — they would
    show up as FN whenever gold=include, which is the honest accounting.
    """
    n_total = len(records)

    # All records contribute to the metrics (parse failures fall back to EXCLUDE).
    n_excluded_by_parse_error = sum(
        1 for r in records
        if not r.get("reporter_ok", False) or not r.get("decider_ok", False)
    )

    tp = fp = fn = tn = 0
    for r in records:
        gold = (r.get("gold_decision") or "").strip().lower()
        pred = (r.get("predicted_decision") or "").strip().upper()
        gold_bin = "include" if gold == "include" else "exclude"
        pred_bin = "include" if pred == "INCLUDE" else "exclude"
        if gold_bin == "include" and pred_bin == "include":
            tp += 1
        elif gold_bin == "include" and pred_bin == "exclude":
            fn += 1
        elif gold_bin == "exclude" and pred_bin == "include":
            fp += 1
        else:
            tn += 1

    n_valid = tp + fp + fn + tn
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / n_valid if n_valid > 0 else 0.0
    if precision + sensitivity > 0:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    else:
        f1 = 0.0

    kappa = None
    try:
        from sklearn.metrics import cohen_kappa_score
        y_true = []
        y_pred = []
        for r in records:
            gold = (r.get("gold_decision") or "").strip().lower()
            pred = (r.get("predicted_decision") or "").strip().upper()
            y_true.append(1 if gold == "include" else 0)
            y_pred.append(1 if pred == "INCLUDE" else 0)
        if y_true and len(y_true) == len(y_pred):
            kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = None

    return {
        "n": n_valid,
        "n_total": n_total,
        "n_excluded_by_parse_error": n_excluded_by_parse_error,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "F1": round(f1, 4),
        "kappa": round(kappa, 4) if kappa is not None else None,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — environment
# ──────────────────────────────────────────────────────────────────────────────

def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def _resolve_templates(variant: str):
    """Resolve a variant code (V1..V4) into (reporter_template, decider_template).

    Returns a tuple of (reporter_attr_name, decider_attr_name, reporter_value,
    decider_value). The names are kept so they end up in the output JSON for
    audit; the values are passed to the agent constructor.
    """
    if variant not in VARIANT_TEMPLATES:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose one of: {list(VARIANT_TEMPLATES)}"
        )
    reporter_name, decider_name = VARIANT_TEMPLATES[variant]
    from agents import prompts as _prompts_mod  # lazy import (prompts is heavy)
    reporter_tpl = getattr(_prompts_mod, reporter_name)
    decider_tpl = getattr(_prompts_mod, decider_name)
    return reporter_name, decider_name, reporter_tpl, decider_tpl


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(
    variant: str,
    split: str,
    max_samples: Optional[int],
    partition_path: Path,
    output_dir: Path,
    cache_dir: Path,
    sample_csv: Path,
    gold_csv: Path,
    pdfs_dir: Path,
    seed: int,
    no_cache: bool,
) -> dict[str, Any]:
    """Run the T17.B.1 evaluator and persist a JSON artefact under `output_dir`."""
    _seed_torch(seed)
    started_at = datetime.now(timezone.utc)
    started_iso = started_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    partition = _load_partition(partition_path)
    if split not in ("dev", "heldout"):
        raise ValueError(f"--split must be 'dev' or 'heldout', got {split!r}")
    target_ids = list(partition[split])

    sample_index = _load_sample_index(sample_csv)
    gold_index = _load_gold_index(gold_csv)

    # Apply max-samples (stratified) if requested.
    if max_samples is not None and max_samples < len(target_ids):
        target_ids = _stratified_subset(target_ids, sample_index, max_samples, seed)

    # Build paper dicts for evaluable rows.
    errors: list[dict[str, Any]] = []
    papers: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for cid in target_ids:
        sample_row = sample_index.get(cid)
        if sample_row is None:
            errors.append({
                "covidence_id": cid,
                "error_type": "missing_from_sample_csv",
                "message": f"covidence_id {cid} not present in {sample_csv.name}",
            })
            skipped.append({"covidence_id": cid, "reason": "missing_from_sample_csv"})
            continue
        gold_row = gold_index.get(cid)
        paper = _build_paper(cid, sample_row, gold_row, pdfs_dir, errors)
        if paper is None:
            skipped.append({"covidence_id": cid, "reason": "build_failed"})
            continue
        papers.append(paper)

    n_total = len(target_ids)
    n_eval = len(papers)

    print()
    print("=" * 72)
    print(f"[T17.B.1.{variant}] split={split}  target={n_total}  evaluables={n_eval}  "
          f"skipped={len(skipped)}")
    print(f"Partition: {partition_path}")
    print(f"Sample CSV: {sample_csv.name}")
    print(f"PDFs dir : {pdfs_dir}")
    print(f"Cache    : {cache_dir}  (no_cache={no_cache})")
    print("=" * 72)

    if n_eval == 0:
        print("[WARN] Sin papers evaluables. Emitiendo envelope vacío.")
        return _persist_result(
            variant=variant,
            split=split,
            reporter_tpl_name="",
            decider_tpl_name="",
            n_target=n_total,
            n_eval=0,
            records=[],
            metrics=_compute_binary_metrics([]),
            started_iso=started_iso,
            finished_iso=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            elapsed_seconds=0.0,
            errors=errors,
            skipped=skipped,
            output_dir=output_dir,
            cache_dir=cache_dir,
            seed=seed,
            partition_path=partition_path,
            partition_meta={
                "seed": partition.get("seed"),
                "git_commit": partition.get("git_commit"),
                "n_dev": partition.get("n_dev"),
                "n_heldout": partition.get("n_heldout"),
            },
        )

    # Resolve variant -> templates.
    reporter_tpl_name, decider_tpl_name, reporter_tpl, decider_tpl = _resolve_templates(variant)

    # Instantiate agent.
    from agents.agent4_fulltext_split import Agent4FullTextSplit

    agent = Agent4FullTextSplit(
        ensemble_size=1,
        reporter_prompt_template=reporter_tpl,
        decider_prompt_template=decider_tpl,
    )
    # Explicit load — keeps the single VRAM slot under the evaluator's control.
    agent.load_model()

    a4_params = load_agent_params("agent4_validation")
    llm_config = {
        "reporter_temperature": agent.reporter_temperature,
        "decider_temperature": agent.decider_temperature,
        "reporter_max_tokens": agent.reporter_max_tokens,
        "decider_max_tokens": agent.decider_max_tokens,
        "reporter_repetition_penalty": agent.reporter_repetition_penalty,
        "decider_repetition_penalty": agent.decider_repetition_penalty,
        "ensemble_size": agent.ensemble_size,
        "voting_threshold": agent.voting_threshold,
        "max_input_chars": a4_params.get("max_input_chars", 6000),
        "seed": seed,
    }

    model_path = agent.model_path or "unknown"
    model_display = (
        "Qwen2.5-7B-Instruct (4-bit BitsAndBytes)"
        if isinstance(model_path, str) and "qwen" in model_path.lower()
        else str(model_path)
    )

    # Run per-paper loop with cache.
    records: list[dict[str, Any]] = []
    t0 = time.time()
    try:
        for idx, paper in enumerate(papers):
            cid = paper["covidence_id"]
            gold = paper["_gold_decision"]
            result = _screen_with_cache(
                agent=agent,
                paper=paper,
                protocol=PAULA_SPORT_PROTOCOL,
                cache_dir=cache_dir,
                variant=variant,
                no_cache=no_cache,
            )
            record = {
                "covidence_id": cid,
                "gold_decision": gold,
                "gold_reason_raw": paper.get("_gold_reason_raw", ""),
                "predicted_decision": result["fulltext_decision"],
                "reporter_ok": result["reporter_ok"],
                "decider_ok": result["decider_ok"],
                "confidence": result.get("confidence"),
                "reasoning": result.get("reasoning"),
                "exclusion_reason": result.get("exclusion_reason"),
                "reporter": result.get("reporter"),
                "decider": result.get("decider"),
                "reporter_raw": result.get("reporter_raw"),
                "decider_raw": result.get("decider_raw"),
                "screening_confidence": result.get("screening_confidence"),
                "borderline": result.get("borderline"),
                "from_cache": result.get("from_cache", {"reporter": False, "decider": False}),
                "text_source": paper.get("text_source"),
                "full_text_length": paper.get("full_text_length"),
                "pdf_filename": paper.get("_pdf_filename", ""),
            }
            records.append(record)

            cache_tag = (
                f"R:{'hit' if record['from_cache']['reporter'] else 'miss'}/"
                f"D:{'hit' if record['from_cache']['decider'] else 'miss'}"
            )
            print(
                f"[T17.B.1.{variant}] sample {idx + 1}/{n_eval} id={cid} "
                f"gold={gold.upper()} pred={record['predicted_decision']} "
                f"reporter_ok={record['reporter_ok']} decider_ok={record['decider_ok']} "
                f"conf={record['confidence']} cache={cache_tag}"
            )
    finally:
        try:
            agent.free_vram()
        except Exception:
            pass

    elapsed = round(time.time() - t0, 2)
    finished_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics = _compute_binary_metrics(records)

    # Telemetry: count cache files actually present.
    n_cache_reporter = sum(1 for _ in (cache_dir / "reporter").glob("*.json")) if (cache_dir / "reporter").is_dir() else 0
    n_cache_decider = sum(1 for _ in (cache_dir / "decider").glob("*.json")) if (cache_dir / "decider").is_dir() else 0

    print()
    print("=" * 72)
    print(f"[T17.B.1.{variant}] DONE n_eval={n_eval} elapsed={elapsed}s")
    print(
        f"  metrics: sens={metrics['sensitivity']} spec={metrics['specificity']} "
        f"acc={metrics['accuracy']} F1={metrics['F1']} kappa={metrics['kappa']}"
    )
    print(
        f"  confusion: TP={metrics['confusion']['TP']} FP={metrics['confusion']['FP']} "
        f"FN={metrics['confusion']['FN']} TN={metrics['confusion']['TN']}"
    )
    print(
        f"  parse-error fallbacks: "
        f"{metrics['n_excluded_by_parse_error']}/{metrics['n_total']}"
    )
    print(f"  cache files on disk: reporter={n_cache_reporter} decider={n_cache_decider}")
    print("=" * 72)

    return _persist_result(
        variant=variant,
        split=split,
        reporter_tpl_name=reporter_tpl_name,
        decider_tpl_name=decider_tpl_name,
        n_target=n_total,
        n_eval=n_eval,
        records=records,
        metrics=metrics,
        started_iso=started_iso,
        finished_iso=finished_iso,
        elapsed_seconds=elapsed,
        errors=errors,
        skipped=skipped,
        output_dir=output_dir,
        cache_dir=cache_dir,
        seed=seed,
        partition_path=partition_path,
        partition_meta={
            "seed": partition.get("seed"),
            "git_commit": partition.get("git_commit"),
            "n_dev": partition.get("n_dev"),
            "n_heldout": partition.get("n_heldout"),
        },
        llm_config=llm_config,
        model_display=model_display,
        n_cache_reporter=n_cache_reporter,
        n_cache_decider=n_cache_decider,
    )


def _persist_result(
    *,
    variant: str,
    split: str,
    reporter_tpl_name: str,
    decider_tpl_name: str,
    n_target: int,
    n_eval: int,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    started_iso: str,
    finished_iso: str,
    elapsed_seconds: float,
    errors: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    output_dir: Path,
    cache_dir: Path,
    seed: int,
    partition_path: Path,
    partition_meta: dict[str, Any],
    llm_config: dict[str, Any] | None = None,
    model_display: str | None = None,
    n_cache_reporter: int = 0,
    n_cache_decider: int = 0,
) -> dict[str, Any]:
    """Write the run output JSON and return its in-memory dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{variant}_{split}_{stamp}.json"

    payload = {
        "experiment": "T17.B.1",
        "variant": variant,
        "split": split,
        "n_target": n_target,
        "n_eval": n_eval,
        "templates": {
            "reporter": reporter_tpl_name,
            "decider": decider_tpl_name,
        },
        "partition_file": str(partition_path),
        "partition_meta": partition_meta,
        "model_used": model_display or "n/a",
        "llm_config": llm_config or {},
        "metrics": metrics,
        "confusion_matrix": metrics.get("confusion"),
        "cache_telemetry": {
            "files_reporter": n_cache_reporter,
            "files_decider": n_cache_decider,
        },
        "records": records,
        "errors": errors,
        "skipped": skipped,
        "started_at": started_iso,
        "finished_at": finished_iso,
        "elapsed_seconds": elapsed_seconds,
        "git_commit": _git_head(),
        "environment": {
            "python_version": platform.python_version(),
            "gpu_info": _gpu_info(),
        },
        "seed": seed,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nOutput JSON guardado en: {out_path}")
    return payload


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T17.B.1 evaluator — Agent4 reporter/decider split."
    )
    p.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANT_TEMPLATES.keys()),
        help="Variant code: V1 / V2 / V3 / V4 (selects reporter+decider templates).",
    )
    p.add_argument(
        "--split",
        required=True,
        choices=("dev", "heldout"),
        help="Partition split to evaluate.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples (stratified). Default: full split.",
    )
    p.add_argument(
        "--partition",
        type=Path,
        default=DEFAULT_PARTITION,
        help=f"Path to partition.json (default: {DEFAULT_PARTITION}).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for run JSONs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory root (default: {DEFAULT_CACHE_DIR}).",
    )
    p.add_argument(
        "--sample-csv",
        type=Path,
        default=DEFAULT_SAMPLE_CSV,
        help="Sample CSV with has_pdf + gold labels (default: sample_medio_a4.csv).",
    )
    p.add_argument(
        "--gold-csv",
        type=Path,
        default=DEFAULT_GOLD_CSV,
        help="Gold screening CSV (title/abstract).",
    )
    p.add_argument(
        "--pdfs-dir",
        type=Path,
        default=DEFAULT_PDFS_DIR,
        help="Directory of local PDFs.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for stratified sub-sampling and torch (default 42).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable read+write cache (forces fresh LLM calls).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_eval(
            variant=args.variant,
            split=args.split,
            max_samples=args.max_samples,
            partition_path=args.partition.resolve(),
            output_dir=args.output_dir.resolve(),
            cache_dir=args.cache_dir.resolve(),
            sample_csv=args.sample_csv.resolve(),
            gold_csv=args.gold_csv.resolve(),
            pdfs_dir=args.pdfs_dir.resolve(),
            seed=args.seed,
            no_cache=args.no_cache,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ERROR durante la evaluación: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
