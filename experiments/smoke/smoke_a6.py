"""
Smoke test for Agent6 Quality / Risk-of-Bias (T16.D.5.F).

Validates that Agent6.assess_all produces a parseable, sane RoB assessment on
a single hand-picked local PDF from the micro sample_50.csv:

  - #13442 — Bennell 2020, "Behavior change text messages for home exercise
    adherence in knee osteoarthritis: randomized trial". This is an RCT, so
    Agent6 should pick RoB 2.0.

Pass criteria (all must hold for exit code 0):
  - quality_assessment.tool in {'RoB2', 'ROBINS-I', 'JBI'}
  - quality_assessment.overall is in the documented union of overall labels
    (Low / Some concerns / High / Moderate / Serious / Critical / Include /
     Caution / Exclude / Include with caution / No information)
  - quality_assessment.domains is a non-empty list/dict
  - some textual reasoning is present (overall_justification OR a non-empty
    domains entry with a 'judgement_rationale' / 'reasoning' / similar key)

Output JSON: outputs/eval/smoke/smoke_a6_{YYYYMMDD_HHMMSS}.json

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
AGENT_TAG = "A6"
SMOKE_ID = "smoke_a6"

# Hand-picked fixture: a single hero4-include from sample_50.csv that should
# trigger the RoB 2.0 branch (RCT).
SMOKE_FIXTURE = {
    "covidence_id": "#13442",
    "pdf_filename": "Bennell-2020-Behavior-change-text-messages-for-h_14.pdf",
    "title": (
        "Behavior change text messages for home exercise adherence in knee "
        "osteoarthritis: randomized trial"
    ),
    "abstract": "",
    "authors": ["Bennell K"],
    "year": "2020",
    "journal": "",
}

# Accepted overall labels across all three quality tools.
VALID_OVERALL_LABELS = {
    # RoB 2.0
    "low", "some concerns", "high",
    # ROBINS-I
    "moderate", "serious", "critical", "no information",
    # JBI
    "include", "include with caution", "caution", "exclude",
}

VALID_TOOLS = {"RoB2", "ROBINS-I", "JBI"}

# Keys we look at to find textual reasoning inside a domain entry.
DOMAIN_REASONING_KEYS = (
    "judgement_rationale",
    "judgment_rationale",
    "reasoning",
    "support_for_judgment",
    "support_for_judgement",
    "rationale",
    "justification",
    "comment",
    "comments",
    "notes",
)


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def _has_textual_reasoning(quality: dict) -> bool:
    """Return True if some textual reasoning is present anywhere in the assessment."""
    if not isinstance(quality, dict):
        return False
    overall_just = quality.get("overall_justification") or quality.get("overall_rationale")
    if isinstance(overall_just, str) and overall_just.strip():
        return True
    domains = quality.get("domains")
    items = quality.get("items")
    for container in (domains, items):
        if isinstance(container, dict):
            iterable = container.values()
        elif isinstance(container, list):
            iterable = container
        else:
            iterable = []
        for entry in iterable:
            if isinstance(entry, dict):
                for k in DOMAIN_REASONING_KEYS:
                    v = entry.get(k)
                    if isinstance(v, str) and v.strip():
                        return True
            elif isinstance(entry, str) and entry.strip():
                return True
    return False


def _domains_non_empty(quality: dict) -> bool:
    """Return True if ``domains`` (or ``items`` for JBI) is a non-empty list/dict."""
    if not isinstance(quality, dict):
        return False
    for key in ("domains", "items"):
        v = quality.get(key)
        if isinstance(v, dict) and len(v) > 0:
            return True
        if isinstance(v, list) and len(v) > 0:
            return True
    return False


def run_smoke_a6(output_dir: Path, pdfs_dir: Path) -> dict:
    """Run the A6 smoke test and write JSON output."""
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
                "No se pudo extraer texto del PDF de smoke A6. "
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

    # ── Agent6 setup ──────────────────────────────────────────────────────
    from agents.agent6_quality import Agent6Quality

    a6_params = load_agent_params("agent6_quality")
    agent = Agent6Quality()

    llm_config = {
        "temperature": agent.temperature,
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

    # ── Run assessment ────────────────────────────────────────────────────
    start_ts = time.time()
    try:
        assess_out = agent.assess_all([paper], PAULA_SPORT_PROTOCOL)
    except Exception as e:
        errors.append({
            "input_id": fix["covidence_id"],
            "error_type": "ModelError",
            "message": f"assess_all raised: {type(e).__name__}: {e}",
        })
        assess_out = {"papers": [], "stats": {}}
    elapsed = round(time.time() - start_ts, 2)

    papers_out = assess_out.get("papers", [])
    assessed = papers_out[0] if papers_out else None

    # ── Validate ──────────────────────────────────────────────────────────
    quality = assessed.get("quality_assessment") if assessed else None
    tool = (quality or {}).get("tool", "")
    overall = (quality or {}).get("overall", "")
    overall_lower = str(overall or "").strip().lower()
    design = assessed.get("study_design_detected") if assessed else None
    weight = assessed.get("quality_weight") if assessed else None

    check_tool_ok = tool in VALID_TOOLS
    check_overall_ok = overall_lower in VALID_OVERALL_LABELS
    check_domains_ok = _domains_non_empty(quality or {})
    check_reasoning_ok = _has_textual_reasoning(quality or {})

    all_pass = (
        check_tool_ok
        and check_overall_ok
        and check_domains_ok
        and check_reasoning_ok
        and (not errors or all(
            e.get("error_type") != "ModelError" for e in errors
        ))
    )
    exit_code = 0 if all_pass else 1

    # Snapshot of the domains list / items list for the report
    domains_snapshot: list = []
    if isinstance((quality or {}).get("domains"), dict):
        for k, v in quality["domains"].items():
            if isinstance(v, dict):
                domains_snapshot.append({
                    "domain": k,
                    "judgement": v.get("judgement") or v.get("judgment"),
                })
    elif isinstance((quality or {}).get("domains"), list):
        for d in quality["domains"]:
            if isinstance(d, dict):
                domains_snapshot.append({
                    "domain": d.get("domain") or d.get("name"),
                    "judgement": d.get("judgement") or d.get("judgment"),
                })
    items_snapshot: list = []
    items = (quality or {}).get("items")
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                items_snapshot.append({
                    "item": it.get("item") or it.get("name"),
                    "answer": it.get("answer") or it.get("judgement"),
                })

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
            "tool_ok": check_tool_ok,
            "overall_ok": check_overall_ok,
            "domains_non_empty": check_domains_ok,
            "reasoning_present": check_reasoning_ok,
        },
        "all_pass": all_pass,
        "study_design_detected": design,
        "tool": tool,
        "overall": overall,
        "quality_weight": weight,
        "domains_snapshot": domains_snapshot,
        "items_snapshot": items_snapshot,
        "quality_assessment_full": quality,
        "errors": errors,
        "notes": (
            "Smoke test A6: 1 PDF de un include conocido (#13442 Bennell "
            "2020 RCT). Validamos tool en {RoB2,ROBINS-I,JBI}, overall en "
            "union de labels documentadas, domains/items no vacio y "
            "reasoning textual presente."
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
            "tool_ok": False,
            "overall_ok": False,
            "domains_non_empty": False,
            "reasoning_present": False,
        },
        "all_pass": False,
        "study_design_detected": None,
        "tool": "",
        "overall": "",
        "quality_weight": None,
        "domains_snapshot": [],
        "items_snapshot": [],
        "quality_assessment_full": None,
        "errors": errors,
        "notes": notes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{SMOKE_ID}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[WARN] Smoke A6 emitted empty envelope -> {out_path}")
    result["_exit_code"] = 1
    return result


def _print_report(result: dict, out_path: Path) -> None:
    """Print a short summary to stdout (Spanish)."""
    print()
    print("=" * 64)
    print("SMOKE A6 (Quality / Risk of Bias): RESUMEN")
    print("=" * 64)
    print(f"PDF                    : {result['pdf_filename']}")
    print(f"Covidence ID           : {result['covidence_id']}")
    print(f"Modelo                 : {result['model_used']}")
    print(f"Tiempo total           : {result['elapsed_seconds']}s")
    print(f"Texto extraido (chars) : {result['full_text_length']}")
    print()
    print(f"study_design_detected  : {result['study_design_detected']}")
    print(f"tool                   : {result['tool']}")
    print(f"overall                : {result['overall']}")
    print(f"quality_weight         : {result['quality_weight']}")
    n_dom = len(result.get("domains_snapshot") or [])
    n_items = len(result.get("items_snapshot") or [])
    print(f"domains (n)            : {n_dom}")
    print(f"items   (n)            : {n_items}")
    if result.get("domains_snapshot"):
        for d in result["domains_snapshot"][:8]:
            print(f"   - {_truncate(str(d.get('domain')), 50):<50} -> {d.get('judgement')}")
    print()
    checks = result["checks"]
    print(f"  [{'PASS' if checks['tool_ok'] else 'FAIL'}] tool in {{RoB2, ROBINS-I, JBI}}")
    print(f"  [{'PASS' if checks['overall_ok'] else 'FAIL'}] overall en labels documentadas")
    print(f"  [{'PASS' if checks['domains_non_empty'] else 'FAIL'}] domains/items no vacios")
    print(f"  [{'PASS' if checks['reasoning_present'] else 'FAIL'}] reasoning textual presente")
    print()
    verdict = "VERDE: todos los checks pasaron." if result["all_pass"] else (
        "ROJO: al menos un check falló — revisar A6 antes de hero4."
    )
    print(verdict)
    print(f"Output JSON guardado en: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke test for Agent6 quality / RoB (1 local PDF)."
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
    result = run_smoke_a6(output_dir, pdfs_dir)
    return int(result.get("_exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
