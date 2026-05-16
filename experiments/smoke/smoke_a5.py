"""
Smoke test for Agent5 Data Extraction (T16.D.5.F).

Validates that Agent5.extract_all produces a parseable, sane extraction on a
single hand-picked local PDF from the micro sample_50.csv:

  - #13442 — Bennell K et al. 2020, "Behavior change text messages for home
    exercise adherence in knee osteoarthritis: randomized trial"
    (hero1..hero4 include, PDF with extractable text layer, validated in
    smoke A4).

Pass criteria (all must hold for exit code 0):
  - extraction_quality != 'failed'
  - len(outcomes) > 0
  - At least one outcome has either numeric_complete=True OR has at least
    one numeric field (n_intervention, n_control, mean_intervention,
    mean_control, sd_intervention, sd_control, effect_size_value) that is
    not None.

Output JSON: outputs/eval/smoke/smoke_a5_{YYYYMMDD_HHMMSS}.json

This script does NOT modify agents/, base.py, config.yaml, prompts.py, app.py
or .env.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so `agents` and `experiments` resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._eval_common import (  # noqa: E402
    PAULA_SPORT_PROTOCOL,
    extract_pdf_text as _extract_pdf_text,
    gpu_info as _gpu_info,
    load_agent_params,
    seed_torch as _seed_torch,
    truncate as _truncate,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "smoke"
DEFAULT_PDFS_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_todos"
AGENT_TAG = "A5"
SMOKE_ID = "smoke_a5"

# Hand-picked fixture: a single, relatively simple include from sample_50.
# Re-fixtured 2026-05-16 (T16.D.6.A.bis): #3490 (Bennett 2007) was a scanned
# PDF with no extractable text layer. Replaced with #13442 (Bennell 2020),
# already validated in smoke_a4 with 64395 chars of full text.
SMOKE_FIXTURE = {
    "covidence_id": "#13442",
    "pdf_filename": "Bennell-2020-Behavior-change-text-messages-for-h_14.pdf",
    "title": (
        "Behavior change text messages for home exercise adherence in "
        "knee osteoarthritis: randomized trial"
    ),
    "abstract": "",
    "authors": ["Bennell KL"],
    "year": "2020",
    "journal": "",
}

# A5 numeric fields used by the pass/fail check.
NUMERIC_FIELDS = (
    "n_intervention",
    "n_control",
    "mean_intervention",
    "mean_control",
    "sd_intervention",
    "sd_control",
    "effect_size_value",
)


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def _has_any_numeric(outcome: dict) -> bool:
    """Return True if at least one numeric field on the outcome is not None."""
    for f in NUMERIC_FIELDS:
        v = outcome.get(f)
        if v is None:
            continue
        if isinstance(v, float) and v != v:  # NaN
            continue
        return True
    # Fallback path: n_per_group dict often holds the n_intervention/n_control
    npg = outcome.get("n_per_group")
    if isinstance(npg, dict):
        if npg.get("intervention") is not None or npg.get("control") is not None:
            return True
    return False


def run_smoke_a5(output_dir: Path, pdfs_dir: Path) -> dict:
    """Run the A5 smoke test and write JSON output."""
    _seed_torch(42)

    errors: list = []
    fix = SMOKE_FIXTURE
    pdf_path = pdfs_dir / fix["pdf_filename"]
    text, err = _extract_pdf_text(pdf_path)
    if text is None:
        errors.append({
            "input_id": fix["covidence_id"],
            "error_type": "pdf_extraction_failed",
            "message": f"PDF '{fix['pdf_filename']}' extraction failed: {err}",
        })
        return _emit_empty(
            output_dir=output_dir,
            errors=errors,
            notes=(
                "No se pudo extraer texto del PDF de smoke A5. "
                f"Verifica '{fix['pdf_filename']}' en pdfs_dir."
            ),
        )

    paper = {
        "pmid": fix["covidence_id"],
        "title": fix["title"],
        "abstract": fix["abstract"],
        "authors": fix["authors"],
        "year": fix["year"],
        "journal": fix["journal"],
        "doi": "",
        "full_text": text,
        "full_text_length": len(text),
        "text_source": "local_pdf",
    }

    # ── Agent5 setup ──────────────────────────────────────────────────────
    from agents.agent5_extraction import Agent5Extraction

    a5_params = load_agent_params("agent5_extraction")
    agent = Agent5Extraction()

    llm_config = {
        "temperature": 0.1,
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

    # ── Run extraction ────────────────────────────────────────────────────
    start_ts = time.time()
    try:
        extract_out = agent.extract_all([paper], PAULA_SPORT_PROTOCOL)
    except Exception as e:
        errors.append({
            "input_id": fix["covidence_id"],
            "error_type": "ModelError",
            "message": f"extract_all raised: {type(e).__name__}: {e}",
        })
        extract_out = {"papers": [], "stats": {}}
    elapsed = round(time.time() - start_ts, 2)

    papers_out = extract_out.get("papers", [])
    extracted = papers_out[0] if papers_out else None

    # ── Validate ──────────────────────────────────────────────────────────
    extraction_quality = (
        extracted.get("extraction_quality") if extracted else "missing"
    )
    outcomes = (extracted.get("outcomes") or []) if extracted else []
    n_outcomes = len(outcomes)
    has_any_numeric_outcome = any(_has_any_numeric(o) for o in outcomes)

    check_quality_ok = extraction_quality != "failed" and extraction_quality != "missing"
    check_outcomes_present = n_outcomes > 0
    check_numeric_present = has_any_numeric_outcome

    all_pass = (
        check_quality_ok
        and check_outcomes_present
        and check_numeric_present
        and (not errors or all(
            e.get("error_type") != "ModelError" for e in errors
        ))
    )
    exit_code = 0 if all_pass else 1

    # First outcome snapshot for the report (numeric fields only)
    first_outcome_snapshot = None
    if outcomes:
        o = outcomes[0]
        npg = o.get("n_per_group") if isinstance(o.get("n_per_group"), dict) else {}
        first_outcome_snapshot = {
            "name": o.get("name"),
            "measure_type": o.get("measure_type") or o.get("effect_size_type"),
            "numeric_complete": o.get("numeric_complete"),
            "n_intervention": o.get("n_intervention") or (npg or {}).get("intervention"),
            "n_control": o.get("n_control") or (npg or {}).get("control"),
            "mean_intervention": o.get("mean_intervention"),
            "mean_control": o.get("mean_control"),
            "sd_intervention": o.get("sd_intervention"),
            "sd_control": o.get("sd_control"),
            "effect_size_value": o.get("effect_size_value"),
            "ci_lower": o.get("ci_lower"),
            "ci_upper": o.get("ci_upper"),
            "p_value": o.get("p_value"),
        }

    # ── Envelope ──────────────────────────────────────────────────────────
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "smoke_id": SMOKE_ID,
        "agent": AGENT_TAG,
        "covidence_id": fix["covidence_id"],
        "pdf_filename": fix["pdf_filename"],
        "full_text_length": len(text),
        "model_used": model_display,
        "llm_config": llm_config,
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": elapsed,
        "environment": {
            "python_version": platform.python_version(),
            "gpu_info": _gpu_info(),
        },
        "checks": {
            "quality_ok": check_quality_ok,
            "outcomes_present": check_outcomes_present,
            "numeric_present": check_numeric_present,
        },
        "all_pass": all_pass,
        "extraction_quality": extraction_quality,
        "n_outcomes": n_outcomes,
        "extraction_confidence": (
            extracted.get("extraction_confidence") if extracted else None
        ),
        "sections_used": (
            extracted.get("sections_used", []) if extracted else []
        ),
        "first_outcome": first_outcome_snapshot,
        "outcomes_full": outcomes,
        "errors": errors,
        "notes": (
            "Smoke test A5: 1 PDF de un include conocido (#13442, "
            "Bennell 2020). Validamos quality!='failed', outcomes>=1 y "
            "que al menos un outcome traiga campos numericos no-null."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{SMOKE_ID}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    _print_report(result, out_path)

    try:
        agent.free_vram()
    except Exception:
        pass

    result["_exit_code"] = exit_code
    return result


def _emit_empty(output_dir: Path, errors: list, notes: str) -> dict:
    """Emit a minimal failure envelope when PDF could not be loaded."""
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "smoke_id": SMOKE_ID,
        "agent": AGENT_TAG,
        "covidence_id": SMOKE_FIXTURE["covidence_id"],
        "pdf_filename": SMOKE_FIXTURE["pdf_filename"],
        "full_text_length": 0,
        "model_used": "n/a (no PDF processed)",
        "llm_config": {},
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": 0.0,
        "environment": {
            "python_version": platform.python_version(),
            "gpu_info": _gpu_info(),
        },
        "checks": {
            "quality_ok": False,
            "outcomes_present": False,
            "numeric_present": False,
        },
        "all_pass": False,
        "extraction_quality": "missing",
        "n_outcomes": 0,
        "extraction_confidence": None,
        "sections_used": [],
        "first_outcome": None,
        "outcomes_full": [],
        "errors": errors,
        "notes": notes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{SMOKE_ID}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[WARN] Smoke A5 emitted empty envelope -> {out_path}")
    result["_exit_code"] = 1
    return result


def _print_report(result: dict, out_path: Path) -> None:
    """Print a short summary to stdout (Spanish)."""
    print()
    print("=" * 64)
    print("SMOKE A5 (Extracción de Datos): RESUMEN")
    print("=" * 64)
    print(f"PDF                    : {result['pdf_filename']}")
    print(f"Covidence ID           : {result['covidence_id']}")
    print(f"Modelo                 : {result['model_used']}")
    print(f"Tiempo total           : {result['elapsed_seconds']}s")
    print(f"Texto extraido (chars) : {result['full_text_length']}")
    print()
    print(f"extraction_quality     : {result['extraction_quality']}")
    print(f"extraction_confidence  : {result['extraction_confidence']}")
    print(f"n_outcomes             : {result['n_outcomes']}")
    print(f"sections_used          : {result['sections_used']}")
    print()
    checks = result["checks"]
    print(f"  [{'PASS' if checks['quality_ok'] else 'FAIL'}] quality != failed")
    print(f"  [{'PASS' if checks['outcomes_present'] else 'FAIL'}] outcomes >= 1")
    print(f"  [{'PASS' if checks['numeric_present'] else 'FAIL'}] al menos un outcome con numerico no-null")
    fo = result["first_outcome"]
    if fo:
        print()
        print(f"Primer outcome         : {_truncate(str(fo.get('name')), 80)}")
        print(f"  measure_type         : {fo.get('measure_type')}")
        print(f"  numeric_complete     : {fo.get('numeric_complete')}")
        print(
            f"  n_int/n_ctl          : {fo.get('n_intervention')} / "
            f"{fo.get('n_control')}"
        )
        print(
            f"  mean_int/mean_ctl    : {fo.get('mean_intervention')} / "
            f"{fo.get('mean_control')}"
        )
        print(
            f"  sd_int/sd_ctl        : {fo.get('sd_intervention')} / "
            f"{fo.get('sd_control')}"
        )
        print(f"  effect_size_value    : {fo.get('effect_size_value')}")
    print()
    verdict = "VERDE: todos los checks pasaron." if result["all_pass"] else (
        "ROJO: al menos un check falló — revisar A5 antes de hero3."
    )
    print(verdict)
    print(f"Output JSON guardado en: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke test for Agent5 extraction (1 local PDF)."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: outputs/eval/smoke/).",
    )
    p.add_argument(
        "--pdfs-dir",
        type=Path,
        default=DEFAULT_PDFS_DIR,
        help=(
            "Directory containing the sample_50 PDFs "
            "(default: data/gold_standard/deporte/pdfs_todos/)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output.resolve()
    pdfs_dir = args.pdfs_dir.resolve()
    if not pdfs_dir.is_dir():
        print(f"ERROR: pdfs_dir not found: {pdfs_dir}", file=sys.stderr)
        return 2
    result = run_smoke_a5(output_dir, pdfs_dir)
    return int(result.get("_exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
