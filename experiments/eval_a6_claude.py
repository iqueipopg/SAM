"""T17.C.3 (Part D) — deterministic harness (NO LLM) for the Claude track of A6 (RoB).

This script is the DETERMINISTIC glue around the A6 Claude run. It does NOT
invoke any LLM / subagent: ``Task(sam_a6_quality)`` is a harness tool that only
the orchestrator can call. This file only (a) PREPARES a run (init_run + per-paper
full-text cache + a gold sidecar) and (b) EVALUATES the committed per-paper A6
outputs against Paula's gold standard, recomputing the RoB `overall` IN CODE so
the Claude track is scored under exactly the same aggregation rules as Qwen.

It mirrors ``experiments/eval_a3_claude.py`` (the ``prepare`` / ``evaluate``
two-subcommand pattern) but targets A6 instead of A3, and reuses the gold-subset
construction and metric helpers from ``experiments/eval_hero4_rob.py`` so the
output JSON shares the SAME schema as ``hero4_results_*.json``.

────────────────────────────────────────────────────────────────────────────
MODULE-LEVEL DEPENDENCY NOTE (C.2 — SHARED)
────────────────────────────────────────────────────────────────────────────
The RoB aggregation doctrine (per-domain canonicalisation + overall recompute +
quality weight) lives in ONE place: ``experiments.eval_hero4_rob`` exports
``canonicalize_domain_verdict``, ``aggregate_rob2_overall``,
``aggregate_robins_i_overall``, ``aggregate_jbi_overall`` and
``compute_quality_weight``. This harness imports them (module top) so the Claude
track is scored under EXACTLY the same logic as the Qwen track — no parallel
copy of the whitelist/casing lives here.

Reuses (does NOT reimplement):
  - ``sam_orchestrator``: ``RunPaths``, ``init_run``, ``paper_id_to_slug``,
    ``RunPaths.paper_output("sam_a6_quality", cid)`` for per-paper outputs.
  - ``experiments._eval_common``: ``extract_pdf_text``, ``load_eval_config``.
  - ``experiments.eval_hero4_rob``: gold-subset construction
    (``_filter_hero4_includes``, ``_build_ref_to_covidence``,
    ``_build_gold_subset``, ``_worst_case_gold``), A6-overall normalisation
    (``_normalize_a6_overall``), metric computation (``_compute_primary_metrics``,
    ``_compute_secondary_metrics``) and the scale constants (``ROB_SCALE``,
    ``ROB_ORDINAL``, ``A6_OVERALL_NORMALISATION``).
  - ``config/eval/medio.yaml``: flattened ``samples.hero4`` -> ``cfg["sample"]``
    exactly as ``run_heroes_medio._materialise_per_hero_config`` does.

NO sacred zones are touched. Usage:
    python experiments/eval_a6_claude.py prepare [--config config/eval/medio.yaml]
    python experiments/eval_a6_claude.py evaluate <run_id> [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sam_orchestrator import (  # noqa: E402
    RunPaths,
    init_run,
    paper_id_to_slug,
)
from experiments._eval_common import (  # noqa: E402
    extract_pdf_text as _extract_pdf_text,
    load_eval_config,
)
from experiments.eval_hero4_rob import (  # noqa: E402
    A6_OVERALL_NORMALISATION,
    ROB_ORDINAL,
    ROB_SCALE,
    _build_gold_subset,
    _build_ref_to_covidence,
    _compute_primary_metrics,
    _compute_secondary_metrics,
    _filter_hero4_includes,
    _normalize_a6_overall,
    _worst_case_gold,
    aggregate_jbi_overall,
    aggregate_rob2_overall,
    aggregate_robins_i_overall,
    canonicalize_domain_verdict,
    compute_quality_weight,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = _REPO_ROOT / "config" / "eval" / "medio.yaml"
_HERO_ID = "hero4"
_AGENT_TAG = "A6"
_AGENT_ID = "sam_a6_quality"

# Per-run sidecar with the gold labels A6 will be scored against.
_META_FILENAME = "_a6_eval_meta.json"
# Sub-directory inside the run dir holding one full-text .txt per paper.
_FULLTEXT_CACHE_DIRNAME = "_fulltext_cache"

# Default clinical query (same Paula sport MA used across the Claude track).
_DEFAULT_QUERY = (
    "¿Mejoran las intervenciones de actividad física con componentes "
    "psicológicos los resultados de actividad física en adultos de edad de "
    "jubilación (50-70 años) que viven en la comunidad?"
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ──────────────────────────────────────────────────────────────────────────────
# Config loading — flatten medio.yaml samples.hero4 -> cfg["sample"]
# ──────────────────────────────────────────────────────────────────────────────

def _load_medio_cfg(config_path: Path) -> dict[str, Any]:
    """Load medio.yaml and flatten ``samples.hero4`` into ``cfg["sample"]``.

    Mirrors ``run_heroes_medio._materialise_per_hero_config`` so the rest of the
    code can read ``cfg["sample"]`` the same way the standalone evaluators do.
    """
    cfg = load_eval_config(config_path)
    samples_map = cfg.get("samples") or {}
    sample_path = samples_map.get(_HERO_ID)
    if not sample_path:
        raise KeyError(f"{config_path} missing samples.{_HERO_ID}")
    flat = dict(cfg)
    flat.pop("samples", None)
    flat["sample"] = sample_path
    flat["heroes"] = [_HERO_ID]
    return flat


# ──────────────────────────────────────────────────────────────────────────────
# Gold subset (worst-case per paper) — built via eval_hero4_rob helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_gold_by_cid(
    gold_rob_df: pd.DataFrame,
    ref_to_cid: dict[int, str],
    target_refs: set[int],
    errors: list[dict],
) -> tuple[dict[str, list[dict]], dict]:
    """Build ``covidence_id -> [ {outcome_name, gold_rob, ...} ]`` from gold_rob.

    Reuses ``_build_gold_subset`` (eval_hero4_rob) to attribute each gold_rob
    row to a covidence_id via the high-confidence ref->covidence mapping, then
    groups the rows per study. Rows whose ``rob_judgment`` falls outside
    ROB_SCALE are recorded in ``errors`` and dropped (same policy as Hero 4).
    """
    gold_subset, gold_debug = _build_gold_subset(
        gold_rob_df, ref_to_cid, target_refs
    )
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
    return gold_by_cid, gold_debug


# ──────────────────────────────────────────────────────────────────────────────
# prepare
# ──────────────────────────────────────────────────────────────────────────────

def cmd_prepare(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else _DEFAULT_CONFIG
    config_path = config_path.resolve()
    if not config_path.is_file():
        print(f"ERROR: no existe config {config_path}", file=sys.stderr)
        return 2

    cfg = _load_medio_cfg(config_path)
    sample_path = (_REPO_ROOT / cfg["sample"]).resolve()
    gold_rob_path = (_REPO_ROOT / cfg["gold_rob"]).resolve()
    ref_map_path = (_REPO_ROOT / cfg["ref_to_covidence"]).resolve()
    pdfs_dir = (_REPO_ROOT / cfg["pdfs_dir"]).resolve()

    # --- 1. Load CSVs + filter hero4 includes ---
    sample_df = pd.read_csv(sample_path, dtype={"covidence_id": str})
    gold_rob_df = pd.read_csv(gold_rob_path)
    ref_map_df = pd.read_csv(ref_map_path)

    includes_df = _filter_hero4_includes(sample_df)
    include_ids_list = includes_df["covidence_id"].astype(str).tolist()
    include_ids: set[str] = set(include_ids_list)

    # --- 2. Build worst-case gold subset by paper ---
    errors: list[dict] = []
    ref_to_cid, target_refs = _build_ref_to_covidence(ref_map_df, include_ids)
    gold_by_cid, gold_debug = _build_gold_by_cid(
        gold_rob_df, ref_to_cid, target_refs, errors
    )

    # --- 3. init_run (subset_spec=None; this run only holds the A6 cache) ---
    run_id = init_run(args.query or _DEFAULT_QUERY, subset_spec=None,
                      backend_default="claude")
    rp = RunPaths(run_id)
    cache_dir = rp.run_dir / _FULLTEXT_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 4. Extract full text for every include with has_pdf=True ---
    sidecar_papers: list[dict] = []
    n_cached = 0
    for _, row in includes_df.iterrows():
        cid = str(row["covidence_id"])
        slug = paper_id_to_slug(cid)
        has_pdf = str(row.get("has_pdf", "")).strip().lower() == "true"
        pdf_filename = row.get("pdf_filename")

        gold_outcomes = gold_by_cid.get(cid, [])
        gold_overall = _worst_case_gold(
            [o["gold_rob"] for o in gold_outcomes]
        )

        if not has_pdf or pdf_filename is None or (
            isinstance(pdf_filename, float) and pd.isna(pdf_filename)
        ):
            errors.append({
                "input_id": cid,
                "error_type": "no_pdf_available",
                "message": (
                    f"Sample row has_pdf={row.get('has_pdf')!r} / "
                    f"pdf_filename={pdf_filename!r}; no full-text cache written."
                ),
            })
        else:
            pdf_path = pdfs_dir / str(pdf_filename)
            text, err = _extract_pdf_text(pdf_path, min_chars=500)
            if text is None:
                errors.append({
                    "input_id": cid,
                    "error_type": "pdf_extraction_failed",
                    "message": f"PDF {pdf_filename!r} extraction failed: {err}",
                })
            else:
                # Write the COMPLETE extracted text (NO [:5000] truncation) so
                # the Claude RoB run reads the whole paper.
                (cache_dir / f"{slug}.txt").write_text(text, encoding="utf-8")
                n_cached += 1

        sidecar_papers.append({
            "covidence_id": cid,
            "slug": slug,
            "gold_overall": gold_overall,
            "gold_outcomes": gold_outcomes,
        })

    # --- 5. Sidecar with gold labels per paper ---
    meta = {
        "task": "T17.C.3.D_eval_claude_a6",
        "hero_id": _HERO_ID,
        "agent": _AGENT_ID,
        "config_path": str(config_path),
        "sample_path": str(sample_path),
        "gold_rob_path": str(gold_rob_path),
        "ref_map_path": str(ref_map_path),
        "pdfs_dir": str(pdfs_dir),
        "fulltext_cache_dir": _FULLTEXT_CACHE_DIRNAME,
        "n_includes": len(include_ids_list),
        "n_fulltext_cached": n_cached,
        "papers": sidecar_papers,
        "gold_subset_debug": gold_debug,
        "errors": errors,
    }
    (rp.run_dir / _META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 6. stdout report + instruction for the orchestrator ---
    print(f"run_id: {run_id}")
    print(f"includes hero4         : {len(include_ids_list)}")
    print(f"full-text cacheados    : {n_cached}")
    print(f"cache dir              : {cache_dir}")
    print(f"sidecar                : {rp.run_dir / _META_FILENAME}")
    if errors:
        print(f"errores (no fatales)   : {len(errors)}")
        for e in errors[:5]:
            print(f"  - {e['input_id']}: {e['error_type']}")
    print()
    print("Este arnés NO invoca al subagente. the orchestrator debe ahora, por cada paper:")
    print(f"  Task(sam_a6_quality) leyendo el texto en "
          f"{_FULLTEXT_CACHE_DIRNAME}/<slug>.txt y volcando el output per-paper")
    print(f"  en RunPaths({run_id}).paper_output('{_AGENT_ID}', <covidence_id>).")
    print("Tras commitear todos, correr:")
    print(f"  python experiments/eval_a6_claude.py evaluate {run_id}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# evaluate — domain canonicalisation + code-recomputed overall
# ──────────────────────────────────────────────────────────────────────────────

def _domain_verdict(entry: object) -> str:
    """Read the raw verdict from a domain/item entry.

    Qwen RoB2/ROBINS-I use ``judgment``; Qwen JBI uses ``answer``; the Claude
    QualityOutput schema uses ``judgment`` for all tools. We accept either so
    the harness is robust to the field-name divergence (the actual
    canonicalisation casing/whitelist is owned by ``canonicalize_domain_verdict``
    in eval_hero4_rob).
    """
    if not isinstance(entry, dict):
        return ""
    if "judgment" in entry:
        return str(entry.get("judgment") or "").strip()
    if "answer" in entry:
        return str(entry.get("answer") or "").strip()
    return ""


# Map the harness tool string (as committed by the Claude subagent) to the tool
# tag expected by the shared ``canonicalize_domain_verdict`` ("RoB2" / "ROBINS-I"
# / "JBI"). Unknown tools fall back to RoB2 (same conservative default the local
# copy used previously).
def _tool_tag(tool: str) -> str:
    t = (tool or "").strip().lower()
    if t in ("robins-i", "robins_i", "robinsi"):
        return "ROBINS-I"
    if t == "jbi":
        return "JBI"
    return "RoB2"


def _canonicalise_domains(tool: str, domains: dict) -> dict[str, dict]:
    """Canonicalise per-domain verdicts under the SAME whitelist/casing as Qwen.

    Returns ``{domain_name: {"judgment": canonical_verdict}}`` so the result can
    be handed directly to the shared ``aggregate_*`` helpers (which read
    ``.get("judgment")`` per domain). The casing/whitelist/fallback logic itself
    is owned by ``canonicalize_domain_verdict`` (eval_hero4_rob) — no normalisation
    is duplicated here.
    """
    tag = _tool_tag(tool)
    canonical: dict[str, dict] = {}
    for name, entry in (domains or {}).items():
        raw = _domain_verdict(entry)
        canonical[name] = {"judgment": canonicalize_domain_verdict(raw, tag)}
    return canonical


def cmd_evaluate(args: argparse.Namespace) -> int:
    run_id = args.run_id
    rp = RunPaths(run_id)
    meta_path = rp.run_dir / _META_FILENAME
    if not meta_path.is_file():
        print(f"ERROR: no existe {meta_path}", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    errors: list[dict] = list(meta.get("errors", []))
    gold_debug = meta.get("gold_subset_debug", {})

    primary_pairs: list[tuple[str, str]] = []
    secondary_pairs: list[dict] = []
    per_study_coverage: list[dict] = []
    raw_predictions: list[dict] = []
    tool_counter: dict[str, int] = defaultdict(int)
    pending_ids: list[str] = []

    for entry in meta.get("papers", []):
        cid = entry["covidence_id"]
        gold_overall = entry.get("gold_overall")
        gold_outcomes = entry.get("gold_outcomes", [])

        out_path = rp.paper_output(_AGENT_ID, cid)
        if not out_path.is_file():
            pending_ids.append(cid)
            continue

        a6 = json.loads(out_path.read_text(encoding="utf-8"))
        quality = a6.get("quality_assessment") or {}
        tool = str(quality.get("tool", "") or "").strip()
        domains = quality.get("domains") or {}
        design = a6.get("study_design_detected", "")

        # --- Canonicalise domain verdicts (Qwen-parity), then RECOMPUTE the
        #     overall IN CODE with the C.2 aggregate_* functions. JBI also
        #     returns the yes_count it derives from the 10 canonical items. ---
        canon = _canonicalise_domains(tool, domains)
        t = tool.strip().lower()
        jbi_yes_count: Optional[int] = None
        if t == "rob2":
            a6_overall_raw = aggregate_rob2_overall(canon)
        elif t in ("robins-i", "robins_i", "robinsi"):
            a6_overall_raw = aggregate_robins_i_overall(canon)
        elif t == "jbi":
            jbi_yes_count, a6_overall_raw = aggregate_jbi_overall(canon)
        else:
            a6_overall_raw = None

        weight = (
            compute_quality_weight(a6_overall_raw)
            if a6_overall_raw is not None else None
        )
        a6_overall_norm = _normalize_a6_overall(a6_overall_raw)
        tool_counter[tool or "unknown"] += 1

        if a6_overall_norm is None:
            errors.append({
                "input_id": cid,
                "error_type": "ValidationError",
                "message": (
                    f"Recomputed A6 overall {a6_overall_raw!r} could not be "
                    f"normalised to {ROB_SCALE} (tool={tool!r}); study excluded "
                    "from primary and secondary pairs."
                ),
            })
            continue

        if not gold_outcomes or gold_overall is None:
            errors.append({
                "input_id": cid,
                "error_type": "no_gold_outcomes",
                "message": (
                    "No gold RoB outcomes for this covidence_id via the "
                    "high-confidence ref->covidence mapping."
                ),
            })
            continue

        # --- Primary (one pair per paper, worst-case gold) ---
        primary_pairs.append((gold_overall, a6_overall_norm))
        raw_predictions.append({
            "input_id": cid,
            "gold": gold_overall,
            "predicted": a6_overall_norm,
            "match": gold_overall == a6_overall_norm,
            "details": {
                "level": "paper",
                "a6_overall_raw": a6_overall_raw,
                "a6_tool": tool,
                "design_detected": design,
                "quality_weight": weight,
                "jbi_yes_count": jbi_yes_count,
                "gold_aggregation": "worst_case",
                "gold_outcomes_collapsed": gold_outcomes,
                "raw_llm_response": quality.get("overall_justification", "") or "",
                "field_scores": {
                    "domains_canonical": canon,
                    "domains_raw": domains,
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
                    "a6_tool": tool,
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
            "a6_tool": tool,
            "design_detected": design,
            "gold_outcomes": per_study_outcome_records,
            "failures": per_study_failures,
            "total_outcomes": len(gold_outcomes),
        })

    # --- Metrics (reused from eval_hero4_rob) ---
    primary_metrics = _compute_primary_metrics(primary_pairs)
    secondary_metrics = _compute_secondary_metrics(
        secondary_pairs, per_study_coverage
    )

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
            "Track Claude (Task sam_a6_quality). El `overall` NO se toma del "
            "subagente: se RECOMPUTA POR CÓDIGO con aggregate_rob2_overall / "
            "aggregate_robins_i_overall / aggregate_jbi_overall (C.2) sobre los "
            "juicios de dominio ya canonicalizados con la misma whitelist/casing "
            "que Qwen (agent6_quality.py). Granularidad: PRIMARY (worst-case por "
            "paper, métrica de la memoria) y SECONDARY (one-to-many sin κ)."
        ),
    }

    n_papers = len(meta.get("papers", []))
    notes = (
        f"Eval Claude A6 (T17.C.3.D) sobre {n_papers} includes hero4 (medio). "
        f"Evaluados {len(per_study_coverage)} / pendientes {len(pending_ids)}. "
        f"Primary pairs: {len(primary_pairs)} / Secondary pairs: "
        f"{len(secondary_pairs)}. Overall recomputado por código (no por el "
        "subagente)."
    )

    result = {
        "hero_id": _HERO_ID,
        "agent": _AGENT_TAG,
        "track": "claude",
        "run_id": run_id,
        "sample_path": str(Path(meta.get("sample_path", ""))),
        "sample_size": n_papers,
        "eval_size": len(per_study_coverage),
        "n_pending": len(pending_ids),
        "pending_ids": pending_ids,
        "model_used": "claude (Task sam_a6_quality)",
        "llm_config": {
            "provider": "claude",
            "note": (
                "Juicios de dominio del subagente; overall/weight recomputados "
                "por código (aggregate_* + compute_quality_weight, C.2)."
            ),
        },
        "timestamp_utc": _utc_iso(),
        "environment": {
            "python_version": platform.python_version(),
            "gpu_info": None,
        },
        "metrics": metrics,
        # Top-level confusion_matrix mirrors metrics.primary.confusion_matrix,
        # same canonical 3x3 contract as hero4_results_*.json.
        "confusion_matrix": primary_metrics["confusion_matrix"],
        "raw_predictions": raw_predictions,
        "errors": errors,
        "notes": notes,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (
            _REPO_ROOT / "outputs" / "eval" / "medio"
            / f"hero4_claude_{_utc_stamp()}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # --- stdout summary ---
    prim = primary_metrics
    sec = secondary_metrics
    print("=" * 64)
    print("A6 CLAUDE TRACK — RESUMEN (Hero 4 / Risk of Bias)")
    print("=" * 64)
    print(f"run_id                 : {run_id}")
    print(f"papers (includes)      : {n_papers}")
    print(f"evaluados              : {len(per_study_coverage)}")
    print(f"pendientes             : {len(pending_ids)}")
    if pending_ids:
        print(f"  ids pendientes       : {', '.join(pending_ids)}")
    print()
    print("MÉTRICA PRIMARIA (worst-case, va a la memoria):")
    print(f"   Pares (paper-level)  : {prim.get('n_pairs', 0)}")
    print(f"   κ cuadrática         : {prim.get('cohens_kappa_quadratic')}")
    print(f"   κ lineal             : {prim.get('cohens_kappa_linear')}")
    print(f"   Accuracy (flat)      : {prim.get('accuracy_flat')}")
    print()
    print("MÉTRICA SECUNDARIA (one-to-many, diagnóstica):")
    print(f"   Pares (outcome-level): {sec.get('n_outcomes', 0)}")
    print(f"   % outcomes cubiertos : {sec.get('pct_outcomes_correctly_covered')}")
    print()
    print(f"Output JSON            : {out_path}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="T17.C.3.D harness for the Claude track of A6 (RoB)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser(
        "prepare",
        help="init_run + full-text cache per paper + gold sidecar (NO subagent).",
    )
    p_prep.add_argument(
        "--config", default=None,
        help="Path to eval YAML (default: config/eval/medio.yaml).",
    )
    p_prep.add_argument("--query", default=None, help="Override the clinical query.")
    p_prep.set_defaults(func=cmd_prepare)

    p_eval = sub.add_parser(
        "evaluate",
        help="Score committed A6 outputs (overall recomputed in code via shared C.2 helpers).",
    )
    p_eval.add_argument("run_id")
    p_eval.add_argument("--out", default=None, help="Override output JSON path.")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
