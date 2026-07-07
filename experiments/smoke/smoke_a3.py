# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Smoke test for Agent3 Screening (T16.D.5.F).

Validates that Agent3.screen_batch produces parseable, sane decisions on a tiny
hand-picked input of 3 synthetic title/abstract pairs:

  1. Obvious-relevant 1  : community-based PA RCT in adults 50-70 with BCT
  2. Obvious-relevant 2  : psychological mediators of a walking RCT, age 60-70
  3. Obvious-irrelevant  : polymer composite mechanical properties

Pass criteria (all must hold for exit code 0):
  - Each relevant paper produces decision in {INCLUDE, UNCERTAIN}
    (EXCLUDE on an obvious-relevant case is a fail).
  - The irrelevant paper produces decision == EXCLUDE.

Output JSON: outputs/eval/smoke/smoke_a3_{YYYYMMDD_HHMMSS}.json

This script does NOT modify agents/, base.py, config.yaml, prompts.py, app.py
or .env. It exists purely to gate-check the agent before running the hero
evaluators against the full sample_50.
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
    gpu_info as _gpu_info,
    load_agent_params,
    seed_torch as _seed_torch,
    truncate as _truncate,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "smoke"
AGENT_TAG = "A3"
SMOKE_ID = "smoke_a3"

# Synthetic mini-dataset. Each paper carries an `_expected` field used by the
# pass/fail logic at the end of the run.
SMOKE_PAPERS: list[dict] = [
    {
        "pmid": "smoke_a3_relevant_1",
        "title": (
            "Effects of a community-based exercise program on physical "
            "activity in adults aged 50-70: a randomized controlled trial"
        ),
        "abstract": (
            "We conducted an RCT in 200 community-dwelling adults aged 55-68 "
            "to evaluate behavior change techniques (goal-setting, "
            "self-monitoring, motivational interviewing) for promoting "
            "physical activity. Participants were randomized to the "
            "intervention or to usual care. The primary outcome was moderate-"
            "to-vigorous physical activity measured by accelerometry at "
            "12 weeks. Secondary outcomes included self-reported PA "
            "(IPAQ) and step counts. The intervention group increased "
            "MVPA by 15 min/day compared with controls (p<0.01)."
        ),
        "_expected": "relevant",
        "_expected_decisions": ("INCLUDE", "UNCERTAIN"),
    },
    {
        "pmid": "smoke_a3_relevant_2",
        "title": (
            "Psychological mediators of a walking intervention in older "
            "adults: RCT findings"
        ),
        "abstract": (
            "Self-efficacy and motivation were measured before and after a "
            "12-week walking program in 150 community-dwelling adults aged "
            "60-70 years. The intervention incorporated motivational "
            "interviewing and goal-setting; the comparator was an attention "
            "control. Step counts via pedometer increased significantly in "
            "the intervention arm. Mediation analysis suggested self-"
            "efficacy mediated the effect on PA."
        ),
        "_expected": "relevant",
        "_expected_decisions": ("INCLUDE", "UNCERTAIN"),
    },
    {
        "pmid": "smoke_a3_irrelevant",
        "title": (
            "Mechanical properties of polymer composites under dynamic load"
        ),
        "abstract": (
            "We characterize the tensile strength of glass fiber composites "
            "at strain rates ranging from 10^-3 to 10^3 per second. "
            "Specimens were tested via split Hopkinson pressure bar. "
            "Results indicate strain-rate hardening behavior consistent with "
            "viscoelastic models. No human participants were involved."
        ),
        "_expected": "irrelevant",
        "_expected_decisions": ("EXCLUDE",),
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def run_smoke_a3(output_dir: Path) -> dict:
    """Run the A3 smoke test and write JSON output. Returns the result dict.

    The returned dict carries an ``_exit_code`` key with the value the caller
    should propagate (0 = all checks passed, 1 = at least one fail).
    """
    _seed_torch(42)

    # Strip evaluator-only metadata before passing to A3.
    papers_for_agent = [
        {"pmid": p["pmid"], "title": p["title"], "abstract": p["abstract"]}
        for p in SMOKE_PAPERS
    ]

    # ── 1. Agent3 setup ───────────────────────────────────────────────────
    from agents.agent3_screening import Agent3Screening

    a3_params = load_agent_params("agent3_screening")
    agent = Agent3Screening()

    llm_config = {
        "temperature": 0.2,
        "max_tokens": a3_params.get("max_tokens", 150),
        "repetition_penalty": a3_params.get("repetition_penalty", 1.3),
        "seed": 42,
        "ensemble_size": agent.ensemble_size,
        "voting_threshold": agent.voting_threshold,
        "max_input_chars": a3_params.get("max_input_chars", 800),
    }

    model_used = agent.model_path or "unknown"
    if isinstance(model_used, str) and "qwen" in model_used.lower():
        model_display = "Qwen2.5-7B-Instruct (4-bit BitsAndBytes)"
    else:
        model_display = str(model_used)

    # ── 2. Run screening ──────────────────────────────────────────────────
    start_ts = time.time()
    errors: list = []
    try:
        screen_out = agent.screen_batch(papers_for_agent, PAULA_SPORT_PROTOCOL)
    except Exception as e:
        errors.append({
            "input_id": "<batch>",
            "error_type": "ModelError",
            "message": f"screen_batch raised: {type(e).__name__}: {e}",
        })
        screen_out = {"include": [], "exclude": [], "uncertain": [], "stats": {}}
    elapsed = round(time.time() - start_ts, 2)

    # Index by pmid for fast lookup
    a3_by_id: dict = {}
    for bucket in ("include", "exclude", "uncertain"):
        for r in screen_out.get(bucket, []):
            a3_by_id[r.get("pmid", "")] = r

    # ── 3. Validate per-paper ─────────────────────────────────────────────
    per_paper_results: list[dict] = []
    checks_passed = 0
    checks_failed = 0

    for paper in SMOKE_PAPERS:
        pid = paper["pmid"]
        expected = paper["_expected"]
        expected_decisions = paper["_expected_decisions"]
        a3_result = a3_by_id.get(pid)

        if a3_result is None:
            decision = "MISSING"
            confidence = 0.0
            reasoning = "Agent3 returned no entry for this paper"
            screening_conf = 0.0
            check_pass = False
            errors.append({
                "input_id": pid,
                "error_type": "ValidationError",
                "message": "pmid absent from Agent3 output",
            })
        else:
            decision = a3_result.get("decision", "EXCLUDE")
            confidence = a3_result.get("confidence", 0.0)
            reasoning = a3_result.get("reasoning", "")
            screening_conf = a3_result.get("screening_confidence", 0.0)
            check_pass = decision in expected_decisions

        per_paper_results.append({
            "pmid": pid,
            "expected": expected,
            "expected_decisions": list(expected_decisions),
            "decision": decision,
            "confidence": confidence,
            "screening_confidence": screening_conf,
            "reasoning": _truncate(reasoning, 400),
            "check_pass": check_pass,
        })

        if check_pass:
            checks_passed += 1
        else:
            checks_failed += 1

    all_pass = (checks_failed == 0) and (not errors or all(
        e.get("error_type") != "ModelError" for e in errors
    ))
    exit_code = 0 if all_pass else 1

    # ── 4. Build envelope ─────────────────────────────────────────────────
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = {
        "smoke_id": SMOKE_ID,
        "agent": AGENT_TAG,
        "n_inputs": len(SMOKE_PAPERS),
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
        "errors": errors,
        "notes": (
            "Smoke test A3: 2 obvious-relevant (esperan INCLUDE/UNCERTAIN) + "
            "1 obvious-irrelevant (espera EXCLUDE). Pass total = 0/1."
        ),
    }

    # ── 5. Persist ────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{SMOKE_ID}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    _print_report(result, out_path)

    # Free VRAM before exit (A3 holds the model).
    try:
        agent.free_vram()
    except Exception:
        pass

    result["_exit_code"] = exit_code
    return result


def _print_report(result: dict, out_path: Path) -> None:
    """Print a short summary to stdout (Spanish)."""
    print()
    print("=" * 64)
    print("SMOKE A3 (Screening Título/Abstract): RESUMEN")
    print("=" * 64)
    print(f"Inputs                 : {result['n_inputs']}")
    print(f"Modelo                 : {result['model_used']}")
    print(f"Tiempo total           : {result['elapsed_seconds']}s")
    print(f"Checks passed          : {result['checks_passed']}/{result['n_inputs']}")
    print(f"Checks failed          : {result['checks_failed']}/{result['n_inputs']}")
    print()
    for r in result["per_paper_results"]:
        marker = "PASS" if r["check_pass"] else "FAIL"
        print(
            f"  [{marker}] {r['pmid']:<28} "
            f"esperado={r['expected']:<11} obtenido={r['decision']:<10} "
            f"conf={r['confidence']}"
        )
        if not r["check_pass"]:
            print(f"        razon: {_truncate(r['reasoning'], 120)}")
    print()
    verdict = "VERDE: todos los checks pasaron." if result["all_pass"] else (
        "ROJO: al menos un check falló — revisar A3 antes de hero1."
    )
    print(verdict)
    print(f"Output JSON guardado en: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke test for Agent3 screening (3 synthetic inputs)."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: outputs/eval/smoke/).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output.resolve()
    result = run_smoke_a3(output_dir)
    return int(result.get("_exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
