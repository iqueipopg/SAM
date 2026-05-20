"""
T17.B.2 protocol-ceiling validator (no GPU, no LLM).

Validates the upper bound of A4 split sensitivity over dev N=387 under
PAULA_SPORT_PROTOCOL_V2, before running the GPU experiment. Inputs:

  * `experiments/T17.B.1/baseline_dev.json` — confusion matrix + metrics
    from T16.E.2.B over the 387-paper dev partition (TP=21, FN=24).
  * `docs/iterations/T17.B.1_iteration_report.md` — table §5.2 listing the
    23 INCLUDE-gold papers that V1 mis-excluded (category D).

Method:

  1. Read `tp_baseline` (T16.E.2.B INCLUDE recall) from the baseline JSON.
  2. Parse table §5.2 to obtain the 23 category-D rows.
  3. Apply V2 keyword rules over the "excluding condition" column:
       - RECOVERABLE  — chronic-condition keywords now in-scope
       - STILL_EXCL_PSYCH — missing psychological component
       - STILL_EXCL_AGE   — age out of range or ambiguous
       - STILL_EXCL_PARSE — technical reporter parse error
       - AMBIGUOUS        — multiple categories mixed
  4. Compute three ceilings (point / optimistic / pessimistic).
  5. Emit JSON + verdict (PASS ≥ gate_threshold, else FAIL_TRIGGER_ASKUSER).

Run:
    python -m experiments.build_t17b2_protocol_validation
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keyword rules (case-insensitive substring match on the "excluding condition"
# column of table §5.2). Order matters: parse-error and age-out checks dominate
# chronic-condition matches when both keywords appear.
RECOVERABLE_KEYWORDS = (
    "t2d",
    "diabetes",
    "cvd",
    "cardiac",
    "cancer survivor",
    "osteoarthritis",
    "multimorbidity",
    "hypertension",
    "hta",
    "obesity",
    "missing info on absence of major comorbidities",
    "metabolic syndrome",
)
STILL_EXCL_PSYCH_KEYWORDS = (
    "sin componente psicológico",
    "intervención sin componente",
)
STILL_EXCL_AGE_KEYWORDS = (
    "edad ambigua",
    "edad fuera",
    "aged < 50",
    "criterio poblacional múltiple",
)
PARSE_ERROR_IDS = {"#3123", "#13730", "#1911"}


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _parse_audit_table(audit_md_path: Path) -> list[dict]:
    """Return list of {covidence_id, population, excluding_condition, rationale}
    for each row of table §5.2 (rows starting with `| #<digits> |`)."""
    text = audit_md_path.read_text(encoding="utf-8")
    rows: list[dict] = []
    row_pat = re.compile(r"^\|\s*(#\d+)\s*\|")
    for line in text.splitlines():
        if not row_pat.match(line):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 4:
            continue
        covidence_id, population, excluding_condition, rationale = cells[:4]
        rows.append({
            "covidence_id": covidence_id,
            "population": population,
            "excluding_condition": excluding_condition,
            "rationale": rationale,
        })
    return rows


def _classify_row(row: dict) -> tuple[str, bool, str]:
    """Return (v2_verdict, ambiguous, rationale_text)."""
    cov_id = row["covidence_id"]
    cond = row["excluding_condition"].lower()
    pop = row["population"].lower()

    # 1. Parse-error rows (technical, not protocol-related)
    if cov_id in PARSE_ERROR_IDS or "parse error" in pop:
        return (
            "STILL_EXCL_PARSE",
            False,
            "Reporter parse error — technical failure, not protocol-related",
        )

    # 2. Missing psychological component (V2 still excludes)
    psych_match = any(kw in cond for kw in STILL_EXCL_PSYCH_KEYWORDS)

    # 3. Age out of range (V2 still excludes)
    age_match = any(kw in cond for kw in STILL_EXCL_AGE_KEYWORDS)

    # 4. Chronic-condition recovery under V2
    recoverable_match = any(kw in cond for kw in RECOVERABLE_KEYWORDS)

    # Priority resolution:
    # - psych failure dominates: even if chronic condition, no psych component
    #   means V2 still excludes.
    if psych_match:
        return (
            "STILL_EXCL_PSYCH",
            recoverable_match,
            "No psychological component — V2 still excludes regardless of chronic condition",
        )

    # - age failure also dominates over chronic recovery
    if age_match:
        return (
            "STILL_EXCL_AGE",
            recoverable_match,
            "Age out of range or unspecified — V2 still excludes",
        )

    if recoverable_match:
        return (
            "RECOVERABLE",
            False,
            "Paula 2025 p.3 includes 'diabetes or cardiac conditions' — chronic condition now in scope",
        )

    # No rule matched — flag ambiguous
    return (
        "AMBIGUOUS_NO_MATCH",
        True,
        "No keyword rule matched — manual review required",
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="T17.B.2 protocol-V2 ceiling validator")
    p.add_argument(
        "--baseline-dump",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "T17.B.1" / "baseline_dev.json",
    )
    p.add_argument(
        "--audit-table",
        type=Path,
        default=PROJECT_ROOT / "docs" / "iterations" / "T17.B.1_iteration_report.md",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "T17.B.2" / "protocol_ceiling.json",
    )
    p.add_argument("--gate-threshold", type=float, default=0.80)
    args = p.parse_args(argv)

    # 1. Load baseline confusion matrix
    if not args.baseline_dump.is_file():
        print(f"ERROR: baseline dump not found: {args.baseline_dump}", file=sys.stderr)
        return 2
    baseline = json.loads(args.baseline_dump.read_text(encoding="utf-8"))
    cm = baseline.get("confusion_matrix", {})
    tp_baseline = int(cm.get("TP", 0))
    fn_baseline = int(cm.get("FN", 0))
    n_include_gold_dev = tp_baseline + fn_baseline
    n_dev = int(baseline.get("n_dev", 387))

    if n_include_gold_dev != 45:
        print(
            f"WARNING: n_include_gold_dev={n_include_gold_dev}, expected 45 "
            "(TP+FN del baseline). Continuando con el valor leído del JSON.",
            file=sys.stderr,
        )

    # 2. Parse audit table §5.2
    if not args.audit_table.is_file():
        print(f"ERROR: audit table not found: {args.audit_table}", file=sys.stderr)
        return 2
    rows = _parse_audit_table(args.audit_table)
    if len(rows) != 23:
        print(
            f"WARNING: parsed {len(rows)} rows from audit table, expected 23. "
            "Continuando.",
            file=sys.stderr,
        )

    # 3. Classify
    classified: list[dict] = []
    for row in rows:
        verdict, ambiguous, rationale = _classify_row(row)
        classified.append({
            "covidence_id": row["covidence_id"],
            "excluding_condition": row["excluding_condition"],
            "v2_verdict": verdict,
            "ambiguous": ambiguous,
            "rationale": rationale,
        })

    n_recoverable_v2 = sum(1 for r in classified if r["v2_verdict"] == "RECOVERABLE")
    n_still_excluded_v2 = sum(
        1 for r in classified if r["v2_verdict"].startswith("STILL_EXCL")
    )
    n_ambiguous = sum(1 for r in classified if r["ambiguous"])

    # 4. Compute ceilings
    ceiling_point = (tp_baseline + n_recoverable_v2) / n_include_gold_dev
    ceiling_opt = (tp_baseline + n_recoverable_v2 + n_ambiguous) / n_include_gold_dev
    ceiling_pess = (tp_baseline + n_recoverable_v2) / n_include_gold_dev

    verdict = "PASS" if ceiling_point >= args.gate_threshold else "FAIL_TRIGGER_ASKUSER"

    payload = {
        "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_commit": _git_commit(),
        "tag_target": "pre-T17.B.2",
        "n_dev": n_dev,
        "n_include_gold_dev": n_include_gold_dev,
        "tp_baseline_v1": tp_baseline,
        "papers_d_total": len(classified),
        "papers_d_recoverable_v2": n_recoverable_v2,
        "papers_d_still_excluded_v2": n_still_excluded_v2,
        "papers_d_ambiguous": n_ambiguous,
        "ceiling_sens_v2_point": round(ceiling_point, 6),
        "ceiling_sens_v2_optimistic": round(ceiling_opt, 6),
        "ceiling_sens_v2_pessimistic": round(ceiling_pess, 6),
        "gate_threshold": args.gate_threshold,
        "verdict": verdict,
        "papers_classified": classified,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    msg_pass = (
        f"PASS — techo V2 = {ceiling_point:.3f} ≥ {args.gate_threshold:.2f} "
        "→ tag pre-T17.B.2"
    )
    msg_fail = (
        f"FAIL — techo V2 = {ceiling_point:.3f} < {args.gate_threshold:.2f} "
        "→ AskUser"
    )
    print(msg_pass if verdict == "PASS" else msg_fail)
    print(f"Output: {args.output}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
