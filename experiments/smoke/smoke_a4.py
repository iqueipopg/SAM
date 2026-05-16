"""
Smoke test for Agent4 Full-Text screening (T16.D.5.F).

Validates that Agent4.screen_fulltext (Phase B) produces parseable, sane
decisions on 3 hand-picked local PDFs from the micro sample_50.csv:

  1. Include obvious     : #13442 — Bennell 2020, behaviour-change text
                           messages (hero1..hero4 include)
  2. Wrong-age obvious   : #12037 — JI 2018, smartphone app
                           (strata=Wrong age)
  3. Wrong-outcome obv.  : #10805 — Peimani 2018, peer support glycemic
                           control (strata=Wrong outcome)

The cascade Phase A (HTTP retrieval) is bypassed: we extract the local PDF
text with the shared helper and feed Agent4.screen_fulltext directly with
text_source='local_pdf'.

Pass criteria (all must hold for exit code 0):
  - The include paper produces fulltext_decision == INCLUDE.
  - Both wrong_X papers produce fulltext_decision == EXCLUDE
    (the textual exclusion_reason is NOT checked for exact match — only that
    the decision is EXCLUDE).

Output JSON: outputs/eval/smoke/smoke_a4_{YYYYMMDD_HHMMSS}.json

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
AGENT_TAG = "A4"
SMOKE_ID = "smoke_a4"

# Hand-picked smoke fixtures. Each entry references a local PDF identified
# from data/gold_standard/deporte/samples/micro/sample_50.csv.
SMOKE_FIXTURES: list[dict] = [
    {
        "covidence_id": "#13442",
        "pdf_filename": "Bennell-2020-Behavior-change-text-messages-for-h_14.pdf",
        "title": (
            "Behavior change text messages for home exercise adherence in "
            "knee osteoarthritis: randomized trial"
        ),
        "abstract": "",
        "_expected_decision": "INCLUDE",
        "_strata": "include",
    },
    {
        "covidence_id": "#12037",
        "pdf_filename": "JI-2018-The Effectiveness of a Smartphone Appl.pdf",
        "title": (
            "The Effectiveness of a Smartphone Application: case from "
            "sample_50 (strata=Wrong age)"
        ),
        "abstract": "",
        "_expected_decision": "EXCLUDE",
        "_strata": "Wrong age",
    },
    {
        "covidence_id": "#10805",
        "pdf_filename": (
            "Peimani M et al. - 2018 - A peer support intervention in "
            "improving glycemic _204.pdf"
        ),
        "title": (
            "A peer support intervention in improving glycemic control in "
            "patients with type 2 diabetes (strata=Wrong outcome)"
        ),
        "abstract": "",
        "_expected_decision": "EXCLUDE",
        "_strata": "Wrong outcome",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def _build_papers(pdfs_dir: Path, errors: list) -> tuple[list[dict], list[dict]]:
    """Load each fixture's PDF and build paper dicts for A4.

    Returns ``(papers, skipped)``. Papers that fail PDF extraction are
    appended to ``errors`` and listed in ``skipped``.
    """
    papers: list[dict] = []
    skipped: list[dict] = []

    for fix in SMOKE_FIXTURES:
        cid = fix["covidence_id"]
        pdf_path = pdfs_dir / fix["pdf_filename"]
        text, err = _extract_pdf_text(pdf_path)
        if text is None:
            errors.append({
                "input_id": cid,
                "error_type": "pdf_extraction_failed",
                "message": f"PDF '{fix['pdf_filename']}' extraction failed: {err}",
            })
            skipped.append({"covidence_id": cid, "reason": "pdf_extraction_failed"})
            continue
        papers.append({
            "pmid": cid,
            "title": fix["title"],
            "abstract": fix["abstract"],
            "full_text": text,
            "full_text_length": len(text),
            "text_source": "local_pdf",
            "year": "",
            "journal": "",
            "_expected_decision": fix["_expected_decision"],
            "_strata": fix["_strata"],
            "_pdf_filename": fix["pdf_filename"],
        })
    return papers, skipped


def run_smoke_a4(output_dir: Path, pdfs_dir: Path) -> dict:
    """Run the A4 smoke test and write JSON output."""
    _seed_torch(42)

    errors: list = []
    papers, skipped = _build_papers(pdfs_dir, errors)
    n_inputs = len(SMOKE_FIXTURES)
    n_eval = len(papers)

    if n_eval == 0:
        return _emit_empty(
            output_dir=output_dir,
            errors=errors,
            n_inputs=n_inputs,
            notes=(
                "No se pudo procesar ningún PDF de smoke A4: revisa rutas "
                "y nombres de fichero en `pdfs_dir`."
            ),
        )

    # ── Agent4 setup ──────────────────────────────────────────────────────
    from agents.agent4_fulltext import Agent4FullText

    a4_params = load_agent_params("agent4_validation")
    agent = Agent4FullText(ensemble_size=1)

    llm_config = {
        "temperature": 0.2,
        "max_tokens": 200,
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

    # ── Run Phase B ───────────────────────────────────────────────────────
    # Pass copies so the evaluator-only `_expected_decision` etc. survive even
    # though screen_fulltext mutates papers in place — Agent4 only reads
    # standard fields and adds new ones, so this is safe.
    start_ts = time.time()
    try:
        screen_out = agent.screen_fulltext(papers, PAULA_SPORT_PROTOCOL)
    except Exception as e:
        errors.append({
            "input_id": "<batch>",
            "error_type": "ModelError",
            "message": f"screen_fulltext raised: {type(e).__name__}: {e}",
        })
        screen_out = {"papers": papers, "include": [], "exclude": [], "stats": {}}
    elapsed = round(time.time() - start_ts, 2)

    a4_by_id: dict = {p["pmid"]: p for p in screen_out.get("papers", papers)}

    # ── Validate per-paper ────────────────────────────────────────────────
    per_paper_results: list[dict] = []
    checks_passed = 0
    checks_failed = 0

    for paper in papers:
        pid = paper["pmid"]
        expected = paper["_expected_decision"]
        a4_paper = a4_by_id.get(pid, paper)
        decision = (a4_paper.get("fulltext_decision") or "EXCLUDE").upper()
        reasoning = a4_paper.get("fulltext_reasoning", "")
        excl_reason = a4_paper.get("exclusion_reason") or ""
        screening_conf = a4_paper.get("screening_confidence", 0.0)

        check_pass = (decision == expected)

        per_paper_results.append({
            "covidence_id": pid,
            "strata": paper.get("_strata"),
            "expected_decision": expected,
            "decision": decision,
            "exclusion_reason": _truncate(excl_reason, 200),
            "fulltext_reasoning": _truncate(reasoning, 400),
            "screening_confidence": screening_conf,
            "pdf_filename": paper.get("_pdf_filename"),
            "full_text_length": paper.get("full_text_length", 0),
            "check_pass": check_pass,
        })

        if check_pass:
            checks_passed += 1
        else:
            checks_failed += 1

    all_pass = (checks_failed == 0) and (n_eval == n_inputs) and (
        not errors or all(e.get("error_type") != "ModelError" for e in errors)
    )
    exit_code = 0 if all_pass else 1

    # ── Build envelope ────────────────────────────────────────────────────
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "smoke_id": SMOKE_ID,
        "agent": AGENT_TAG,
        "n_inputs": n_inputs,
        "n_evaluated": n_eval,
        "model_used": model_display,
        "llm_config": llm_config,
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": elapsed,
        "environment": {
            "python_version": platform.python_version(),
            "gpu_info": _gpu_info(),
        },
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "all_pass": all_pass,
        "per_paper_results": per_paper_results,
        "skipped": skipped,
        "errors": errors,
        "notes": (
            "Smoke test A4: 1 obvious-include (#13442) + 2 obvious-exclude "
            "(#12037 Wrong age, #10805 Wrong outcome). Bypassa Phase A: "
            "text_source='local_pdf'. Solo se valida decision binaria; "
            "exclusion_reason libre-text NO se valida."
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


def _emit_empty(
    output_dir: Path,
    errors: list,
    n_inputs: int,
    notes: str,
) -> dict:
    """Emit a minimal failure envelope when no PDF could be loaded."""
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "smoke_id": SMOKE_ID,
        "agent": AGENT_TAG,
        "n_inputs": n_inputs,
        "n_evaluated": 0,
        "model_used": "n/a (no PDFs processed)",
        "llm_config": {},
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": 0.0,
        "environment": {
            "python_version": platform.python_version(),
            "gpu_info": _gpu_info(),
        },
        "checks_passed": 0,
        "checks_failed": n_inputs,
        "all_pass": False,
        "per_paper_results": [],
        "skipped": [],
        "errors": errors,
        "notes": notes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{SMOKE_ID}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[WARN] Smoke A4 emitted empty envelope -> {out_path}")
    result["_exit_code"] = 1
    return result


def _print_report(result: dict, out_path: Path) -> None:
    """Print a short summary to stdout (Spanish)."""
    print()
    print("=" * 64)
    print("SMOKE A4 (Screening Full-Text): RESUMEN")
    print("=" * 64)
    print(f"Inputs                 : {result['n_inputs']}")
    print(f"Evaluados              : {result['n_evaluated']}")
    print(f"Modelo                 : {result['model_used']}")
    print(f"Tiempo total           : {result['elapsed_seconds']}s")
    print(
        f"Checks passed          : {result['checks_passed']}/{result['n_inputs']}"
    )
    print()
    for r in result["per_paper_results"]:
        marker = "PASS" if r["check_pass"] else "FAIL"
        print(
            f"  [{marker}] {r['covidence_id']:<10} {r['strata']:<20} "
            f"esperado={r['expected_decision']:<8} "
            f"obtenido={r['decision']:<8}"
        )
        if not r["check_pass"]:
            print(f"        excl_reason: {r['exclusion_reason']}")
            print(f"        reasoning  : {r['fulltext_reasoning']}")
    print()
    verdict = "VERDE: todos los checks pasaron." if result["all_pass"] else (
        "ROJO: al menos un check falló — revisar A4 antes de hero2."
    )
    print(verdict)
    print(f"Output JSON guardado en: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke test for Agent4 full-text screening (3 local PDFs)."
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
    result = run_smoke_a4(output_dir, pdfs_dir)
    return int(result.get("_exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
