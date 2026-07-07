# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Orchestrator -- run the 4 hero evaluators sequentially on the micro sample.

Sprint:  T16.D.5.G
Schema:  docs/heroes_output_schema.md
Inputs:
  - config/eval/micro.yaml          : single source of truth for paths + heroes
  - experiments/eval_hero1_screening.py
  - experiments/eval_hero2_fulltext.py
  - experiments/eval_hero3_extraction.py
  - experiments/eval_hero4_rob.py

Outputs:
  outputs/eval/micro/hero{N}_results_{ts}.json     (one per hero, written by hero)
  outputs/eval/micro/summary_{ts}.json             (aggregated, schema-stable)
  outputs/eval/micro/summary_{ts}.md               (human-readable, Spanish)

VRAM strategy
-------------
The four LLM agents (A3, A4, A5, A6) all use Qwen2.5-7B 4-bit. Loading the
model takes 2-5 minutes per agent, so reloading 4 times would cost ~10-20 min.
We load the model ONCE via the first agent (A3) and inject the shared model /
tokenizer references into the other three agents BEFORE calling each hero's
``run_evaluation``. Each hero's ``run_evaluation`` accepts an
``agent_instance`` parameter so it skips the internal instantiation.

When ``screen_batch`` / ``screen_fulltext`` / ``extract_all`` / ``assess_all``
call ``self.load_model()`` internally, the call short-circuits because
``_loaded_path`` and ``model`` are already populated (see BaseAgent.load_model).

Failure handling
----------------
If a hero crashes (any exception), the orchestrator logs a truncated stack
trace into ``summary.errors`` and continues with the next hero. The remaining
heroes still run. The summary always emits a valid JSON/MD even if all heroes
fail.

Idle / empty results (a hero processed 0 papers) are treated as warnings, not
errors.

CLI
---
    python experiments/run_heroes_micro.py \\
        --config config/eval/micro.yaml \\
        --output outputs/eval/micro/ \\
        --heroes hero1,hero2,hero3,hero4

All flags are optional with sensible defaults.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# Ensure project root is on sys.path so `agents` and `experiments` packages
# resolve when launching as `python experiments/run_heroes_micro.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._eval_common import (  # noqa: E402
    gpu_info as _gpu_info,
    load_eval_config,
    seed_torch as _seed_torch,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "eval" / "micro.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "micro"
DEFAULT_HEROES = ("hero1", "hero2", "hero3", "hero4")
ALL_HEROES = ("hero1", "hero2", "hero3", "hero4")

# Truncation cap for embedded stack traces in summary.errors (keep summary
# JSON small and readable; full trace also goes to stderr).
TRACEBACK_TRUNCATE = 2000


# ──────────────────────────────────────────────────────────────────────────────
# Shared-agent setup
# ──────────────────────────────────────────────────────────────────────────────

def _build_shared_agents() -> dict:
    """Instantiate A3/A4/A5/A6 sharing a single model in VRAM.

    The model is loaded ONCE via Agent3 (the first hero in the pipeline order).
    A4 / A5 / A6 receive the same ``model`` / ``tokenizer`` references plus the
    bookkeeping attributes (``_loaded_path``, ``_has_chat_template``) so their
    internal ``self.load_model()`` calls short-circuit.

    Returns:
        ``{"hero1": Agent3Screening, "hero2": Agent4FullText,
           "hero3": Agent5Extraction, "hero4": Agent6Quality}``
    """
    from agents.agent3_screening import Agent3Screening
    from agents.agent4_fulltext import Agent4FullText
    from agents.agent5_extraction import Agent5Extraction
    from agents.agent6_quality import Agent6Quality

    print("[bootstrap] Instantiating Agent3 and loading shared model into VRAM...")
    t0 = time.time()
    a3 = Agent3Screening()
    a3.load_model()
    print(f"[bootstrap] Model loaded in {time.time() - t0:.1f}s — sharing across heroes.")

    # ensemble_size=1 matches the standalone hero2 evaluator (single A4 vote)
    a4 = Agent4FullText(ensemble_size=1)
    a5 = Agent5Extraction()
    a6 = Agent6Quality()

    # Inject the shared model + tokenizer + bookkeeping into A4 / A5 / A6 so
    # their internal load_model() short-circuits (`_loaded_path == path`).
    for child in (a4, a5, a6):
        child.model = a3.model
        child.tokenizer = a3.tokenizer
        child._loaded_path = a3._loaded_path
        child._has_chat_template = a3._has_chat_template
        # Align the model path too, so the short-circuit comparison holds even
        # if config.yaml mapped different keys for these agents.
        child.model_path = a3._loaded_path or a3.model_path

    return {"hero1": a3, "hero2": a4, "hero3": a5, "hero4": a6}


def _release_shared_agents(agents: dict) -> None:
    """Release the shared model from VRAM.

    Only A3 owns the underlying tensors — A4/A5/A6 hold the same references.
    We drop the references on the children first and then call ``free_vram``
    on the owner so the GC can reclaim memory.
    """
    a3 = agents.get("hero1")
    for key in ("hero2", "hero3", "hero4"):
        child = agents.get(key)
        if child is not None:
            child.model = None
            child.tokenizer = None
            child._loaded_path = None
            child._has_chat_template = None
    if a3 is not None:
        try:
            a3.free_vram()
        except Exception as e:  # pragma: no cover - defensive
            print(f"[teardown] free_vram raised: {e}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Hero registry
# ──────────────────────────────────────────────────────────────────────────────

def _hero_registry() -> dict[str, dict]:
    """Return a registry mapping hero_id -> {label, run_fn}.

    Imports are deferred so ``--help`` does not require pandas / sklearn.
    """
    from experiments.eval_hero1_screening import run_hero1_evaluation
    from experiments.eval_hero2_fulltext import run_hero2_evaluation
    from experiments.eval_hero3_extraction import run_hero3_evaluation
    from experiments.eval_hero4_rob import run_hero4_evaluation

    return {
        "hero1": {
            "label": "Hero 1 (A3 — Screening Título/Abstract)",
            "agent_tag": "A3",
            "run_fn": run_hero1_evaluation,
        },
        "hero2": {
            "label": "Hero 2 (A4 — Screening Full-Text)",
            "agent_tag": "A4",
            "run_fn": run_hero2_evaluation,
        },
        "hero3": {
            "label": "Hero 3 (A5 — Extracción de Datos)",
            "agent_tag": "A5",
            "run_fn": run_hero3_evaluation,
        },
        "hero4": {
            "label": "Hero 4 (A6 — Risk of Bias)",
            "agent_tag": "A6",
            "run_fn": run_hero4_evaluation,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-hero execution
# ──────────────────────────────────────────────────────────────────────────────

def _run_single_hero(
    hero_id: str,
    label: str,
    run_fn: Callable,
    config_path: Path,
    output_dir: Path,
    agent_instance,
) -> dict:
    """Execute a single hero. Always returns a per-hero record dict.

    Record shape:
        {
            "hero_id": str,
            "label": str,
            "status": "ok" | "failed" | "warning",
            "elapsed_seconds": float,
            "result": dict | None,           # the hero's full result JSON
            "error": dict | None,            # {error_type, message, traceback} on failure
        }
    """
    print()
    print("=" * 72)
    print(f"[{hero_id}] {label}: iniciado")
    print("=" * 72)
    t_start = time.time()
    record: dict = {
        "hero_id": hero_id,
        "label": label,
        "status": "ok",
        "elapsed_seconds": 0.0,
        "result": None,
        "error": None,
    }
    try:
        result = run_fn(config_path, output_dir, agent_instance=agent_instance)
        record["result"] = result
        elapsed = round(time.time() - t_start, 2)
        record["elapsed_seconds"] = elapsed
        # Warning if no papers processed (eval_size == 0 in heroes 2-4).
        eval_n = result.get("eval_size")
        if eval_n is not None and eval_n == 0:
            record["status"] = "warning"
            print(
                f"[{hero_id}] {label}: completado pero eval_size=0 "
                f"({elapsed}s) — sin papers procesables."
            )
        else:
            print(f"[{hero_id}] {label}: completado en {elapsed}s")
    except Exception as exc:
        elapsed = round(time.time() - t_start, 2)
        record["elapsed_seconds"] = elapsed
        record["status"] = "failed"
        tb = traceback.format_exc()
        record["error"] = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb[:TRACEBACK_TRUNCATE],
        }
        print(f"[{hero_id}] {label}: FALLÓ en {elapsed}s — {type(exc).__name__}: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
    return record


# ──────────────────────────────────────────────────────────────────────────────
# Summary aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _extract_top_metrics(record: dict) -> dict:
    """Pluck the few headline metrics each hero advertises.

    Returns an empty dict for failed heroes (caller decides how to render).
    The keys are intentionally hero-specific to avoid leaky abstractions.
    """
    result = record.get("result") or {}
    metrics = result.get("metrics") or {}
    hero_id = record["hero_id"]
    if hero_id == "hero1":
        return {
            "sensitivity_overall": metrics.get("sensitivity_overall"),
            "sensitivity_adjusted": metrics.get("sensitivity_adjusted"),
            "sensitivity_abstract_yes": metrics.get("sensitivity_abstract_yes"),
            "sensitivity_abstract_no": metrics.get("sensitivity_abstract_no"),
            "specificity": metrics.get("specificity"),
            "precision": metrics.get("precision"),
            "f1": metrics.get("f1"),
            "accuracy": metrics.get("accuracy"),
            "cohens_kappa": metrics.get("cohens_kappa"),
            "tp": metrics.get("tp"),
            "fn": metrics.get("fn"),
            "tn": metrics.get("tn"),
            "fp": metrics.get("fp"),
            "uncertain_count": metrics.get("uncertain_count"),
            "n_relevant_gold": metrics.get("n_relevant_gold"),
            "n_irrelevant_gold": metrics.get("n_irrelevant_gold"),
        }
    if hero_id == "hero2":
        return {
            "accuracy": metrics.get("accuracy"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "f1": metrics.get("f1"),
            "cohens_kappa": metrics.get("cohens_kappa"),
            "top_3_confusions": _top_confusions(
                metrics.get("stratified_matrix_8x8") or {},
                top_n=3,
            ),
        }
    if hero_id == "hero3":
        return {
            "pct_fields_correct_overall": metrics.get("pct_fields_correct_overall"),
            "fields_total": metrics.get("fields_total"),
            "fields_correct": metrics.get("fields_correct"),
            "outcome_match_counts": metrics.get("outcome_match_counts"),
            "by_field_type": metrics.get("by_field_type"),
            "by_outcome_category": metrics.get("by_outcome_category"),
        }
    if hero_id == "hero4":
        primary = metrics.get("primary") or {}
        secondary = metrics.get("secondary") or {}
        return {
            "primary_kappa_quadratic": primary.get("cohens_kappa_quadratic"),
            "primary_kappa_linear": primary.get("cohens_kappa_linear"),
            "primary_accuracy_flat": primary.get("accuracy_flat"),
            "primary_n_pairs": primary.get("n_pairs"),
            "secondary_pct_outcomes_correctly_covered": secondary.get(
                "pct_outcomes_correctly_covered"
            ),
            "secondary_n_outcomes": secondary.get("n_outcomes"),
        }
    return {}


def _top_confusions(matrix: dict, top_n: int = 3) -> list[dict]:
    """Return the ``top_n`` off-diagonal cells of an N×N confusion matrix.

    Each entry is ``{"gold": ..., "predicted": ..., "count": int}``. Empty
    list if the matrix is missing or fully diagonal.
    """
    if not isinstance(matrix, dict) or not matrix:
        return []
    flat: list[tuple[str, str, int]] = []
    for gold, row in matrix.items():
        if not isinstance(row, dict):
            continue
        for pred, count in row.items():
            try:
                c = int(count)
            except (TypeError, ValueError):
                continue
            if gold == pred or c <= 0:
                continue
            flat.append((gold, pred, c))
    flat.sort(key=lambda t: t[2], reverse=True)
    return [
        {"gold": g, "predicted": p, "count": c}
        for (g, p, c) in flat[:top_n]
    ]


def _find_result_path(output_dir: Path, hero_id: str, run_started_ts: float) -> Optional[str]:
    """Locate the JSON written by a hero during this run.

    Heroes name their files ``hero{N}_results_{YYYYMMDD_HHMMSS}.json``. We
    pick the most recent file matching the pattern whose mtime is after the
    run start (small skew tolerated).
    """
    pattern = f"{hero_id}_results_*.json"
    candidates = sorted(output_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.stat().st_mtime >= run_started_ts - 5:
            try:
                return str(c.resolve())
            except OSError:
                return str(c)
    return None


def _build_summary_json(
    run_start_iso: str,
    elapsed_total: float,
    config_path: Path,
    output_dir: Path,
    heroes_records: list[dict],
    requested_heroes: list[str],
    run_start_epoch: float,
) -> dict:
    """Build the aggregated summary JSON conforming to T16.D.5.G spec."""
    per_hero: dict = {}
    errors: list = []
    for rec in heroes_records:
        hero_id = rec["hero_id"]
        per_hero[hero_id] = {
            "label": rec["label"],
            "status": rec["status"],
            "elapsed_seconds": rec["elapsed_seconds"],
            "result_path": _find_result_path(output_dir, hero_id, run_start_epoch),
            "top_metrics": _extract_top_metrics(rec),
        }
        if rec["status"] == "failed" and rec.get("error"):
            err = dict(rec["error"])
            err["hero_id"] = hero_id
            errors.append(err)

    return {
        "schema_version": "t16.d.5.g",
        "timestamp_utc": run_start_iso,
        "elapsed_total_seconds": elapsed_total,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "heroes_requested": list(requested_heroes),
        "heroes_executed": [r["hero_id"] for r in heroes_records],
        "environment": {
            "python_version": platform.python_version(),
            "gpu_info": _gpu_info(),
        },
        "per_hero": per_hero,
        "errors": errors,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(value, ndigits: int = 4) -> str:
    """Format scalars for the markdown table: None -> 'n/a', floats rounded."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def _build_summary_md(summary: dict) -> str:
    """Render the human-readable Spanish markdown summary."""
    lines: list[str] = []
    lines.append("# Resumen de evaluación — micro sample (T16.D.5.G)")
    lines.append("")
    lines.append(f"- Timestamp UTC: `{summary['timestamp_utc']}`")
    lines.append(f"- Duración total: `{summary['elapsed_total_seconds']}s`")
    lines.append(f"- Config: `{summary['config_path']}`")
    lines.append(f"- Output dir: `{summary['output_dir']}`")
    env = summary.get("environment", {})
    lines.append(
        f"- Entorno: Python `{env.get('python_version', 'n/a')}`, "
        f"GPU `{env.get('gpu_info', 'n/a')}`"
    )
    lines.append(
        f"- Heroes solicitados: `{', '.join(summary['heroes_requested'])}` — "
        f"ejecutados: `{', '.join(summary['heroes_executed']) or 'ninguno'}`"
    )
    lines.append("")

    per_hero = summary.get("per_hero", {})

    # ── Tabla resumen ──────────────────────────────────────────────────────
    lines.append("## Tabla resumen")
    lines.append("")
    lines.append("| Hero | Status | Tiempo (s) | Output JSON |")
    lines.append("|------|--------|-----------:|-------------|")
    for hero_id in summary["heroes_executed"]:
        info = per_hero.get(hero_id, {})
        status = info.get("status", "?")
        elapsed = info.get("elapsed_seconds", 0.0)
        path = info.get("result_path") or "n/a"
        lines.append(
            f"| {info.get('label', hero_id)} | `{status}` | {elapsed} | `{path}` |"
        )
    lines.append("")

    # ── Hero 1 ─────────────────────────────────────────────────────────────
    info1 = per_hero.get("hero1")
    if info1:
        lines.append("## Hero 1 — A3 Title/Abstract Screening")
        m = info1.get("top_metrics", {})
        lines.append("")
        lines.append(f"- Status: `{info1.get('status')}`")
        lines.append(f"- Sensitivity (overall): `{_fmt(m.get('sensitivity_overall'))}`")
        lines.append(f"- Sensitivity (abstract sí): `{_fmt(m.get('sensitivity_abstract_yes'))}`")
        lines.append(f"- Sensitivity (abstract no): `{_fmt(m.get('sensitivity_abstract_no'))}`")
        lines.append(
            f"- TP / FN / uncertain: `{m.get('tp')} / {m.get('fn')} / {m.get('uncertain_count')}`"
        )
        lines.append("")

    # ── Hero 2 ─────────────────────────────────────────────────────────────
    info2 = per_hero.get("hero2")
    if info2:
        lines.append("## Hero 2 — A4 Full-Text Screening")
        m = info2.get("top_metrics", {})
        lines.append("")
        lines.append(f"- Status: `{info2.get('status')}`")
        lines.append(f"- Accuracy: `{_fmt(m.get('accuracy'))}`")
        lines.append(f"- Sensitivity: `{_fmt(m.get('sensitivity'))}`")
        lines.append(f"- Specificity: `{_fmt(m.get('specificity'))}`")
        lines.append(f"- F1: `{_fmt(m.get('f1'))}`")
        lines.append(f"- Cohen's kappa: `{_fmt(m.get('cohens_kappa'))}`")
        confusions = m.get("top_3_confusions") or []
        if confusions:
            lines.append("- Top 3 confusiones (matriz 8×8):")
            for c in confusions:
                lines.append(
                    f"  - gold=`{c['gold']}` → pred=`{c['predicted']}` "
                    f"(n=`{c['count']}`)"
                )
        else:
            lines.append("- Top 3 confusiones (matriz 8×8): _ninguna confusión registrada_")
        lines.append("")

    # ── Hero 3 ─────────────────────────────────────────────────────────────
    info3 = per_hero.get("hero3")
    if info3:
        lines.append("## Hero 3 — A5 Data Extraction")
        m = info3.get("top_metrics", {})
        lines.append("")
        lines.append(f"- Status: `{info3.get('status')}`")
        lines.append(
            f"- pct_fields_correct_overall: `{_fmt(m.get('pct_fields_correct_overall'))}`"
        )
        lines.append(
            f"- Campos correctos / totales: "
            f"`{m.get('fields_correct')}` / `{m.get('fields_total')}`"
        )
        counts = m.get("outcome_match_counts") or {}
        if counts:
            lines.append(
                f"- Matches por outcome: exact=`{counts.get('exact', 0)}` / "
                f"partial=`{counts.get('partial', 0)}` / "
                f"no_match=`{counts.get('no_match', 0)}`"
            )
        bft = m.get("by_field_type") or {}
        if bft:
            lines.append("- Por tipo de campo:")
            for k in ("N", "X", "SD"):
                stats = bft.get(k, {})
                if stats:
                    lines.append(
                        f"  - {k}: `{stats.get('correct', 0)}/{stats.get('total', 0)}` "
                        f"(`{_fmt(stats.get('pct_correct'))}`)"
                    )
        boc = m.get("by_outcome_category") or {}
        if boc:
            lines.append("- Por categoría de outcome:")
            for cat, stats in boc.items():
                lines.append(
                    f"  - {cat}: `{stats.get('correct', 0)}/{stats.get('total', 0)}` "
                    f"(`{_fmt(stats.get('pct_correct'))}`)"
                )
        lines.append("")

    # ── Hero 4 ─────────────────────────────────────────────────────────────
    info4 = per_hero.get("hero4")
    if info4:
        lines.append("## Hero 4 — A6 Risk of Bias")
        m = info4.get("top_metrics", {})
        lines.append("")
        lines.append(f"- Status: `{info4.get('status')}`")
        lines.append("- **Métrica primaria** (worst-case por estudio):")
        lines.append(
            f"  - κ cuadrática: `{_fmt(m.get('primary_kappa_quadratic'))}`"
        )
        lines.append(
            f"  - κ lineal:     `{_fmt(m.get('primary_kappa_linear'))}`"
        )
        lines.append(
            f"  - Accuracy (flat): `{_fmt(m.get('primary_accuracy_flat'))}`"
        )
        lines.append(f"  - N pares: `{m.get('primary_n_pairs')}`")
        lines.append("- Métrica secundaria (one-to-many, diagnóstica):")
        lines.append(
            f"  - % outcomes correctamente cubiertos: "
            f"`{_fmt(m.get('secondary_pct_outcomes_correctly_covered'))}`"
        )
        lines.append(f"  - N outcomes: `{m.get('secondary_n_outcomes')}`")
        lines.append("")

    # ── Errores ────────────────────────────────────────────────────────────
    errs = summary.get("errors") or []
    if errs:
        lines.append("## Errores")
        lines.append("")
        for err in errs:
            lines.append(
                f"### {err.get('hero_id')}: `{err.get('error_type')}`"
            )
            lines.append("")
            lines.append(f"Mensaje: `{err.get('message')}`")
            tb = err.get("traceback") or ""
            if tb:
                lines.append("")
                lines.append("```")
                lines.append(tb.rstrip())
                lines.append("```")
            lines.append("")
    else:
        lines.append("## Errores")
        lines.append("")
        lines.append("_Sin errores registrados._")
        lines.append("")

    # ── Caveats & next steps ───────────────────────────────────────────────
    lines.append("## Caveats globales")
    lines.append("")
    lines.append(
        "- Tamaño de muestra: `micro` (50 estudios). Resultados son orientativos; "
        "kappa con N pequeño tiene IC amplios."
    )
    lines.append(
        "- Hero 1: la muestra es 100% gold-positive → solo es medible la "
        "sensitivity (sin specificity ni precision)."
    )
    lines.append(
        "- Hero 3: outcome matching es heurístico (substring + token overlap). "
        "Tolerancias: N exacto, X ±5%, SD ±10%."
    )
    lines.append(
        "- Hero 4: A6 emite UN overall por paper vs gold multi-outcome; "
        "métrica primaria = worst-case por estudio (N=10 → poder bajo)."
    )
    lines.append("")
    lines.append("## Siguientes pasos")
    lines.append("")
    lines.append("- Revisar JSON individuales por hero para auditoría detallada.")
    lines.append("- Escalar a sample `small` / `full` cuando los heroes pasen el gate.")
    lines.append(
        "- Inspeccionar `errors[]` y los false_negatives / mismatches para "
        "diagnosticar fallos sistemáticos del LLM."
    )
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_heroes_arg(raw: str) -> list[str]:
    """Parse a comma-separated --heroes string; validate against ALL_HEROES."""
    if not raw:
        return list(DEFAULT_HEROES)
    items = [s.strip().lower() for s in raw.split(",") if s.strip()]
    unknown = [h for h in items if h not in ALL_HEROES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Heroes desconocidos: {unknown}. Válidos: {list(ALL_HEROES)}"
        )
    return items


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Orquestador T16.D.5.G — corre hero1/2/3/4 secuencialmente "
            "sobre el sample micro compartiendo el modelo en VRAM."
        )
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
        help="Output directory for hero JSONs + summary (default: outputs/eval/micro/).",
    )
    p.add_argument(
        "--heroes",
        type=_parse_heroes_arg,
        default=list(DEFAULT_HEROES),
        help=(
            "Comma-separated subset of {hero1,hero2,hero3,hero4} "
            "(default: hero1,hero2,hero3,hero4)."
        ),
    )
    return p.parse_args()


def _resolve_heroes_from_config(eval_cfg: dict, cli_heroes: list[str]) -> list[str]:
    """Intersect CLI-requested heroes with the YAML's ``heroes:`` list (if any).

    The YAML config may restrict which heroes run (e.g. config/eval/micro.yaml
    lists all four). When both CLI and YAML specify heroes, we honour the CLI
    selection but warn if it goes beyond the YAML list.
    """
    yaml_heroes = eval_cfg.get("heroes") or list(ALL_HEROES)
    yaml_heroes = [str(h).strip().lower() for h in yaml_heroes]
    selected = [h for h in cli_heroes if h in yaml_heroes]
    dropped = [h for h in cli_heroes if h not in yaml_heroes]
    if dropped:
        print(
            f"[config] Aviso: heroes solicitados por CLI pero no listados en "
            f"YAML: {dropped} (ignorados).",
            file=sys.stderr,
        )
    if not selected:
        # Defensive fallback: respect the YAML when CLI yielded nothing.
        selected = yaml_heroes
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    output_dir = args.output.resolve()

    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2

    _seed_torch(42)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = load_eval_config(config_path)
    heroes_to_run = _resolve_heroes_from_config(eval_cfg, args.heroes)

    run_start_epoch = time.time()
    run_start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print()
    print("=" * 72)
    print("ORQUESTADOR T16.D.5.G — run_heroes_micro")
    print("=" * 72)
    print(f"Config           : {config_path}")
    print(f"Output dir       : {output_dir}")
    print(f"Heroes a ejecutar: {heroes_to_run}")
    print(f"GPU              : {_gpu_info()}")
    print(f"Timestamp UTC    : {run_start_iso}")

    # ── 1. Bootstrap shared agents (loads model once) ──────────────────────
    agents: dict = {}
    bootstrap_failed = False
    try:
        agents = _build_shared_agents()
    except Exception as exc:
        bootstrap_failed = True
        tb = traceback.format_exc()
        print(f"[bootstrap] FALLÓ: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)

    registry = _hero_registry()
    heroes_records: list[dict] = []

    if bootstrap_failed:
        # Cannot run any hero. Emit a summary documenting the bootstrap failure.
        for hero_id in heroes_to_run:
            heroes_records.append({
                "hero_id": hero_id,
                "label": registry.get(hero_id, {}).get("label", hero_id),
                "status": "failed",
                "elapsed_seconds": 0.0,
                "result": None,
                "error": {
                    "error_type": "BootstrapError",
                    "message": "Shared model bootstrap failed; hero not executed.",
                    "traceback": "",
                },
            })
    else:
        # ── 2. Run each hero ───────────────────────────────────────────────
        for hero_id in heroes_to_run:
            info = registry.get(hero_id)
            if info is None:
                heroes_records.append({
                    "hero_id": hero_id,
                    "label": hero_id,
                    "status": "failed",
                    "elapsed_seconds": 0.0,
                    "result": None,
                    "error": {
                        "error_type": "UnknownHero",
                        "message": f"Hero '{hero_id}' not registered.",
                        "traceback": "",
                    },
                })
                continue
            record = _run_single_hero(
                hero_id=hero_id,
                label=info["label"],
                run_fn=info["run_fn"],
                config_path=config_path,
                output_dir=output_dir,
                agent_instance=agents.get(hero_id),
            )
            heroes_records.append(record)

        # ── 3. Release shared model ────────────────────────────────────────
        print()
        print("[teardown] Liberando VRAM compartida...")
        _release_shared_agents(agents)

    elapsed_total = round(time.time() - run_start_epoch, 2)

    # ── 4. Build + persist summary JSON ────────────────────────────────────
    summary = _build_summary_json(
        run_start_iso=run_start_iso,
        elapsed_total=elapsed_total,
        config_path=config_path,
        output_dir=output_dir,
        heroes_records=heroes_records,
        requested_heroes=heroes_to_run,
        run_start_epoch=run_start_epoch,
    )
    summary_json_path = output_dir / f"summary_{stamp}.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # ── 5. Build + persist markdown report ─────────────────────────────────
    md_text = _build_summary_md(summary)
    summary_md_path = output_dir / f"summary_{stamp}.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # ── 6. Final stdout block ──────────────────────────────────────────────
    print()
    print("=" * 72)
    print("RESUMEN DEL RUN")
    print("=" * 72)
    print(f"Duración total : {elapsed_total}s")
    print(f"Summary JSON   : {summary_json_path}")
    print(f"Summary MD     : {summary_md_path}")
    print()
    for rec in heroes_records:
        print(
            f"  {rec['hero_id']}: status=`{rec['status']}` "
            f"({rec['elapsed_seconds']}s) → "
            f"{summary['per_hero'].get(rec['hero_id'], {}).get('result_path', 'n/a')}"
        )
    if summary["errors"]:
        print()
        print(f"  {len(summary['errors'])} error(es) registrados — ver summary.md")
    print()

    # Non-zero exit when every hero failed.
    if all(rec["status"] == "failed" for rec in heroes_records) and heroes_records:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
