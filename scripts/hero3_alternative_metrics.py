"""
T16.D.7.B — Hero 3 alternative metrics.

Reads the A5 cache produced by ``scripts/diagnose_hero3.py``
(``outputs/eval/micro/_a5_full_outputs.json``) and computes diagnostic
metrics that decouple Agent5's intrinsic extraction from the schema
alignment problem identified in T16.D.7.A.

Does NOT re-run A5. Read-only on the cache + gold + sample CSVs.
Writes to ``outputs/eval/micro/hero3_alternative_metrics.{md,json}``.

Constraints honoured:
  - NO modifications to ``agents/`` or ``experiments/eval_hero3_extraction.py``
  - NO modifications to prompts
"""

from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_hero3_extraction import (  # noqa: E402
    _build_gold_subset,
    _filter_hero3_includes,
)
from experiments._eval_common import extract_pdf_text  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

A5_CACHE = PROJECT_ROOT / "outputs" / "eval" / "micro" / "_a5_full_outputs.json"
GOLD_METRICS = PROJECT_ROOT / "data" / "gold_standard_normalized" / "gold_metrics.csv"
ARM_MAPPING = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "unified" / "arm_to_parent_mapping.csv"
SAMPLE_CSV = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "samples" / "micro" / "sample_50.csv"
PDFS_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_todos"
OUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "micro"
OUT_JSON = OUT_DIR / "hero3_alternative_metrics.json"
OUT_MD = OUT_DIR / "hero3_alternative_metrics.md"
A5_PROMPTS_FILE = PROJECT_ROOT / "agents" / "prompts.py"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _coerce_float(v):
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null", "n/a"):
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def _classify_outcome(o: dict) -> tuple[str, dict]:
    """Bucket an A5 outcome by which fields it contains.

    Buckets (priority order — first match wins):
      - empty:        no numeric field is non-null
      - numeric_full: all 6 of (n_int, n_ctl, mean_int, mean_ctl, sd_int, sd_ctl) non-null
      - effect_with_ci: effect_size_value AND at least one of ci_lower/ci_upper non-null,
                        with at least one mean missing (otherwise numeric_full takes priority)
      - n_only:       n_int AND n_ctl non-null, all 4 (mean_int, mean_ctl, sd_int, sd_ctl) null
      - partial:      anything else (mix of partially-populated numeric fields)
    """
    n_int = _coerce_float(o.get("n_intervention"))
    n_ctl = _coerce_float(o.get("n_control"))
    if n_int is None:
        npg = o.get("n_per_group") if isinstance(o.get("n_per_group"), dict) else None
        if npg:
            n_int = _coerce_float(npg.get("intervention"))
    if n_ctl is None:
        npg = o.get("n_per_group") if isinstance(o.get("n_per_group"), dict) else None
        if npg:
            n_ctl = _coerce_float(npg.get("control"))
    m_int = _coerce_float(o.get("mean_intervention"))
    m_ctl = _coerce_float(o.get("mean_control"))
    sd_int = _coerce_float(o.get("sd_intervention"))
    sd_ctl = _coerce_float(o.get("sd_control"))
    eff = _coerce_float(o.get("effect_size_value"))
    if eff is None:
        eff = _coerce_float(o.get("value"))
    cil = _coerce_float(o.get("ci_lower"))
    ciu = _coerce_float(o.get("ci_upper"))
    pval = _coerce_float(o.get("p_value"))
    se = _coerce_float(o.get("se"))

    fields_present = {
        "n_intervention": n_int is not None,
        "n_control": n_ctl is not None,
        "mean_intervention": m_int is not None,
        "mean_control": m_ctl is not None,
        "sd_intervention": sd_int is not None,
        "sd_control": sd_ctl is not None,
        "effect_size_value": eff is not None,
        "ci_lower": cil is not None,
        "ci_upper": ciu is not None,
        "p_value": pval is not None,
        "se": se is not None,
    }
    any_value = any(fields_present.values())
    if not any_value:
        return "empty", fields_present
    numeric_full = (
        n_int is not None and n_ctl is not None
        and m_int is not None and m_ctl is not None
        and sd_int is not None and sd_ctl is not None
    )
    if numeric_full:
        return "numeric_full", fields_present
    n_pair = n_int is not None and n_ctl is not None
    mean_pair_missing = m_int is None or m_ctl is None
    if eff is not None and (cil is not None or ciu is not None) and mean_pair_missing:
        return "effect_with_ci", fields_present
    if n_pair and m_int is None and m_ctl is None and sd_int is None and sd_ctl is None:
        return "n_only", fields_present
    return "partial", fields_present


def _has_effect_and_ci(o: dict) -> bool:
    eff = _coerce_float(o.get("effect_size_value")) if o else None
    if eff is None and o:
        eff = _coerce_float(o.get("value"))
    cil = _coerce_float(o.get("ci_lower")) if o else None
    ciu = _coerce_float(o.get("ci_upper")) if o else None
    return eff is not None and cil is not None and ciu is not None


def _is_numeric_complete(o: dict) -> bool:
    """Trust A5's own numeric_complete flag if present, else recompute."""
    if isinstance(o.get("numeric_complete"), bool):
        return bool(o["numeric_complete"])
    bucket, _ = _classify_outcome(o)
    return bucket == "numeric_full"


# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — declare schema
# ──────────────────────────────────────────────────────────────────────────────

def schema_block(gold_df: pd.DataFrame) -> dict:
    return {
        "columns": list(gold_df.columns),
        "dtypes": {c: str(t) for c, t in gold_df.dtypes.items()},
        "sample_rows": gold_df.head(3).where(pd.notnull(gold_df.head(3)), None).to_dict(orient="records"),
        "rows_total": int(len(gold_df)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 1-4 — metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metric_1(papers: list[dict]) -> dict:
    """Percent of A5 outcomes with numeric_complete=True over all A5 outcomes."""
    total = 0
    complete = 0
    per_study = {}
    for p in papers:
        cid = str(p.get("pmid", ""))
        outs = p.get("outcomes") or []
        n_total = len(outs)
        n_complete = sum(1 for o in outs if _is_numeric_complete(o))
        per_study[cid] = {"total": n_total, "complete": n_complete,
                          "pct": (round(n_complete / n_total, 4) if n_total else None)}
        total += n_total
        complete += n_complete
    return {
        "total_outcomes": total,
        "numeric_complete": complete,
        "pct_numeric_complete": (round(complete / total, 4) if total else 0.0),
        "per_study": per_study,
    }


def compute_metric_2(papers: list[dict]) -> dict:
    """Percent of A5 outcomes with effect_size + CI (both bounds) non-null."""
    total = 0
    with_eff_ci = 0
    per_study = {}
    for p in papers:
        cid = str(p.get("pmid", ""))
        outs = p.get("outcomes") or []
        n_total = len(outs)
        n_eff = sum(1 for o in outs if _has_effect_and_ci(o))
        per_study[cid] = {"total": n_total, "with_effect_and_ci": n_eff,
                          "pct": (round(n_eff / n_total, 4) if n_total else None)}
        total += n_total
        with_eff_ci += n_eff
    return {
        "total_outcomes": total,
        "with_effect_and_ci": with_eff_ci,
        "pct_effect_and_ci": (round(with_eff_ci / total, 4) if total else 0.0),
        "per_study": per_study,
    }


def compute_metric_3(papers: list[dict], gold_subset: pd.DataFrame) -> dict:
    """Ratio N_outcomes_A5 / N_primary_comparisons_gold per study.

    Definition of *primary comparison*:
        Unique value of ``outcome_category`` per ``_effective_parent``.
        Rationale: ``outcome_category`` is Paula's canonical bucket (MVPA,
        Sedentary, Steps, walk, Exercises, Strenght, All, Adherence, ...).
        Collapsing timepoints / arms via this column gives the number of
        distinct outcomes the meta-analysis actually compares, independent
        of how many timepoints or arms the trial reports.
    """
    a5_count: dict[str, int] = {}
    for p in papers:
        cid = str(p.get("pmid", ""))
        a5_count[cid] = len(p.get("outcomes") or [])

    gold_count: dict[str, int] = {}
    gold_breakdown: dict[str, list[str]] = {}
    for parent, sub in gold_subset.groupby("_effective_parent"):
        cats = (
            sub["outcome_category"]
            .dropna()
            .astype(str)
            .map(lambda s: s.strip())
            .replace({"": pd.NA})
            .dropna()
            .unique()
            .tolist()
        )
        gold_count[str(parent)] = len(cats)
        gold_breakdown[str(parent)] = cats

    per_study = {}
    for cid in a5_count:
        a5_n = a5_count[cid]
        g_n = gold_count.get(cid, 0)
        per_study[cid] = {
            "n_outcomes_a5": a5_n,
            "n_primary_comparisons_gold": g_n,
            "gold_categories": gold_breakdown.get(cid, []),
            "ratio_a5_over_gold": (round(a5_n / g_n, 3) if g_n else None),
        }
    return {
        "grouping_rule": (
            "primary_comparison = unique non-empty outcome_category within "
            "_effective_parent group (arm-mapped rows folded in via "
            "high-confidence arm_to_parent_mapping)"
        ),
        "per_study": per_study,
    }


def compute_metric_4(papers: list[dict]) -> dict:
    """Distribution of A5 outcomes by bucket per the rules in _classify_outcome."""
    overall = defaultdict(int)
    per_study: dict[str, dict] = {}
    per_outcome_records: list[dict] = []
    for p in papers:
        cid = str(p.get("pmid", ""))
        s_buckets = defaultdict(int)
        for j, o in enumerate(p.get("outcomes") or []):
            bucket, fields_present = _classify_outcome(o)
            overall[bucket] += 1
            s_buckets[bucket] += 1
            per_outcome_records.append({
                "covidence_id": cid,
                "a5_idx": j,
                "name": o.get("name"),
                "bucket": bucket,
                "fields_present": fields_present,
            })
        per_study[cid] = dict(s_buckets)
    return {
        "bucket_rules": {
            "empty": "no numeric field is non-null",
            "numeric_full": "all 6 of (n_int, n_ctl, mean_int, mean_ctl, sd_int, sd_ctl) non-null",
            "effect_with_ci": "effect_size_value AND (ci_lower or ci_upper) non-null with at least one mean missing",
            "n_only": "n_int and n_ctl non-null, all 4 (mean_int, mean_ctl, sd_int, sd_ctl) null",
            "partial": "anything else (mix of partially-populated numeric fields)",
        },
        "overall_counts": dict(overall),
        "per_study_counts": per_study,
        "per_outcome_records": per_outcome_records,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — diagnose specific partial-extraction papers
# ──────────────────────────────────────────────────────────────────────────────

def _read_a5_prompt_templates() -> dict:
    """Return the names of the A5 prompt templates found in agents/prompts.py."""
    txt = A5_PROMPTS_FILE.read_text(encoding="utf-8") if A5_PROMPTS_FILE.is_file() else ""
    names: list[str] = []
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("AGENT5_") and ("TEMPLATE" in s or "PROMPT" in s) and "=" in s:
            names.append(s.split("=", 1)[0].strip())
    return {"file": str(A5_PROMPTS_FILE.relative_to(PROJECT_ROOT)),
            "agent5_template_names": sorted(set(names))}


def _pdf_excerpt_for(cid: str, sample_df: pd.DataFrame, max_chars: int = 1500) -> dict:
    rows = sample_df[sample_df["covidence_id"] == cid]
    if rows.empty:
        return {"covidence_id": cid, "error": "not_in_sample"}
    pdf_filename = rows.iloc[0].get("pdf_filename")
    if not isinstance(pdf_filename, str) or not pdf_filename:
        return {"covidence_id": cid, "error": "no_pdf_filename"}
    pdf_path = PDFS_DIR / pdf_filename
    text, err = extract_pdf_text(pdf_path)
    if text is None:
        return {"covidence_id": cid, "pdf_filename": pdf_filename, "error": err}
    return {
        "covidence_id": cid,
        "pdf_filename": pdf_filename,
        "full_text_length": len(text),
        "head_excerpt": text[:max_chars],
        "tail_excerpt": text[-max_chars:],
    }


def diagnose_partial(cid: str, papers: list[dict], sample_df: pd.DataFrame) -> dict:
    paper = next((p for p in papers if str(p.get("pmid")) == cid), None)
    if paper is None:
        return {"covidence_id": cid, "error": "not_in_a5_cache"}
    outs = paper.get("outcomes") or []
    bucket_counts = defaultdict(int)
    outcomes_dump = []
    for o in outs:
        bucket, fields_present = _classify_outcome(o)
        bucket_counts[bucket] += 1
        outcomes_dump.append({
            "name": o.get("name"),
            "bucket": bucket,
            "fields_present": fields_present,
            "raw": o,
        })
    return {
        "covidence_id": cid,
        "extraction_quality": paper.get("extraction_quality"),
        "extraction_notes": paper.get("extraction_notes"),
        "extraction_confidence": paper.get("extraction_confidence"),
        "sections_used": paper.get("sections_used"),
        "concordance": paper.get("concordance"),
        "n_outcomes": len(outs),
        "bucket_counts": dict(bucket_counts),
        "outcomes_dump": outcomes_dump,
        "pdf_excerpt": _pdf_excerpt_for(cid, sample_df),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Markdown writer
# ──────────────────────────────────────────────────────────────────────────────

def write_md(envelope: dict) -> None:
    lines: list[str] = []
    lines.append("# Hero 3 — métricas alternativas (T16.D.7.B)")
    lines.append("")
    lines.append(f"- Timestamp: `{envelope['timestamp_utc']}`")
    lines.append(f"- A5 cache: `{A5_CACHE.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- A5 cache elapsed: `{envelope['a5_cache_elapsed_seconds']}s`")
    lines.append(f"- N papers analizados: `{envelope['n_papers']}`")
    lines.append("")
    if envelope.get("blocks"):
        lines.append("## Bloqueos detectados")
        lines.append("")
        for b in envelope["blocks"]:
            lines.append(f"- {b}")
        lines.append("")

    # Step 0 schema
    lines.append("## Paso 0 — schema gold_metrics.csv")
    lines.append("")
    s = envelope["schema"]
    lines.append(f"- Filas totales: `{s['rows_total']}`")
    lines.append(f"- Columnas (n={len(s['columns'])}): {', '.join(f'`{c}`' for c in s['columns'])}")
    lines.append("")
    lines.append("**dtypes**:")
    lines.append("")
    lines.append("| columna | dtype |")
    lines.append("|---|---|")
    for c, t in s["dtypes"].items():
        lines.append(f"| `{c}` | `{t}` |")
    lines.append("")
    lines.append("**3 filas de ejemplo**:")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(s["sample_rows"], indent=2, ensure_ascii=False, default=str))
    lines.append("```")
    lines.append("")
    lines.append("**Columna usada para agrupar outcomes (métrica 3)**: `outcome_category`")
    lines.append("")
    lines.append("Regla aplicada: una **primary_comparison** = valor único no vacío de "
                 "`outcome_category` dentro del grupo `_effective_parent` "
                 "(arm-mapped rows folded vía `arm_to_parent_mapping.csv` "
                 "confidence='high', tal como hace el evaluador hero3).")
    lines.append("")

    # Metric 1
    lines.append("## Métrica 1 — % outcomes A5 con `numeric_complete=True`")
    lines.append("")
    m1 = envelope["metric_1"]
    lines.append(f"- Total outcomes A5: **{m1['total_outcomes']}**")
    lines.append(f"- numeric_complete=True: **{m1['numeric_complete']}**")
    lines.append(f"- **% global: {m1['pct_numeric_complete']*100:.2f}%**")
    lines.append("")
    lines.append("| covidence_id | total | numeric_complete | % |")
    lines.append("|---|---:|---:|---:|")
    for cid, s in m1["per_study"].items():
        pct = "—" if s["pct"] is None else f"{s['pct']*100:.1f}%"
        lines.append(f"| {cid} | {s['total']} | {s['complete']} | {pct} |")
    lines.append("")

    # Metric 2
    lines.append("## Métrica 2 — % outcomes A5 con effect_size + CI ambos no-null")
    lines.append("")
    m2 = envelope["metric_2"]
    lines.append(f"- Total outcomes A5: **{m2['total_outcomes']}**")
    lines.append(f"- effect_size_value + ci_lower + ci_upper todos no-null: **{m2['with_effect_and_ci']}**")
    lines.append(f"- **% global: {m2['pct_effect_and_ci']*100:.2f}%**")
    lines.append("")
    lines.append("| covidence_id | total | con effect+CI | % |")
    lines.append("|---|---:|---:|---:|")
    for cid, s in m2["per_study"].items():
        pct = "—" if s["pct"] is None else f"{s['pct']*100:.1f}%"
        lines.append(f"| {cid} | {s['total']} | {s['with_effect_and_ci']} | {pct} |")
    lines.append("")

    # Metric 3
    lines.append("## Métrica 3 — ratio N_outcomes_A5 / N_primary_comparisons_gold")
    lines.append("")
    m3 = envelope["metric_3"]
    lines.append(f"- Regla de agrupamiento: {m3['grouping_rule']}")
    lines.append("")
    lines.append("| covidence_id | A5 outcomes | gold primary cmps | ratio | gold_categories |")
    lines.append("|---|---:|---:|---:|---|")
    for cid, s in m3["per_study"].items():
        ratio = "—" if s["ratio_a5_over_gold"] is None else f"{s['ratio_a5_over_gold']:.3f}"
        cats = ", ".join(s["gold_categories"]) if s["gold_categories"] else "—"
        lines.append(f"| {cid} | {s['n_outcomes_a5']} | {s['n_primary_comparisons_gold']} | {ratio} | {cats} |")
    lines.append("")

    # Metric 4
    lines.append("## Métrica 4 — distribución A5 outcomes por bucket de fields")
    lines.append("")
    m4 = envelope["metric_4"]
    lines.append("**Reglas (prioridad descendente)**:")
    lines.append("")
    for k, v in m4["bucket_rules"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("**Conteo global**:")
    lines.append("")
    lines.append("| bucket | n |")
    lines.append("|---|---:|")
    for k, n in sorted(m4["overall_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{k}` | {n} |")
    lines.append("")
    lines.append("**Por estudio**:")
    lines.append("")
    bucket_names = sorted({b for s in m4["per_study_counts"].values() for b in s.keys()})
    header = "| covidence_id | " + " | ".join(bucket_names) + " |"
    sep = "|---" + "|---:" * len(bucket_names) + "|"
    lines.append(header)
    lines.append(sep)
    for cid, sb in m4["per_study_counts"].items():
        row = "| " + cid + " | " + " | ".join(str(sb.get(b, 0)) for b in bucket_names) + " |"
        lines.append(row)
    lines.append("")

    # Step 5: per-paper deep dive
    lines.append("## Métrica 5 — Diagnóstico partial-extraction (#13950 y #5298)")
    lines.append("")
    lines.append("**Prompt templates A5** (referencia, no se modifican):")
    lines.append("")
    tpl = envelope.get("a5_prompt_templates", {})
    lines.append(f"- file: `{tpl.get('file', '?')}`")
    lines.append(f"- nombres detectados: {', '.join(tpl.get('agent5_template_names', []) or ['(ninguno)'])}")
    lines.append("")
    for entry in envelope.get("metric_5", []):
        cid = entry["covidence_id"]
        lines.append(f"### {cid}")
        lines.append("")
        if entry.get("error"):
            lines.append(f"- Error: `{entry['error']}`")
            lines.append("")
            continue
        lines.append(f"- extraction_quality: `{entry.get('extraction_quality')}`")
        lines.append(f"- extraction_confidence: `{entry.get('extraction_confidence')}`")
        lines.append(f"- sections_used: `{entry.get('sections_used')}`")
        notes = entry.get("extraction_notes")
        if notes:
            lines.append(f"- extraction_notes:")
            lines.append("")
            lines.append("```")
            lines.append(str(notes)[:2000])
            lines.append("```")
        lines.append(f"- n_outcomes A5: **{entry.get('n_outcomes')}**")
        bucks = entry.get("bucket_counts", {})
        lines.append(f"- buckets A5: {bucks}")
        lines.append("")
        if entry.get("outcomes_dump"):
            lines.append("**Outcomes dump**:")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(entry["outcomes_dump"], indent=2, ensure_ascii=False, default=str))
            lines.append("```")
            lines.append("")
        pdf_ex = entry.get("pdf_excerpt", {})
        if pdf_ex and not pdf_ex.get("error"):
            lines.append(f"- PDF `{pdf_ex.get('pdf_filename')}` length: `{pdf_ex.get('full_text_length')}` chars")
            lines.append("")
            lines.append("**PDF head excerpt (primeros 1500 chars)**:")
            lines.append("")
            lines.append("```")
            lines.append(pdf_ex.get("head_excerpt", "")[:1500])
            lines.append("```")
            lines.append("")
            lines.append("**PDF tail excerpt (últimos 1500 chars)**:")
            lines.append("")
            lines.append("```")
            lines.append(pdf_ex.get("tail_excerpt", "")[:1500])
            lines.append("```")
            lines.append("")
        else:
            lines.append(f"- PDF excerpt error: `{pdf_ex.get('error')}`")
            lines.append("")
        concordance = entry.get("concordance")
        if concordance:
            lines.append(f"- concordance:")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(concordance, indent=2, ensure_ascii=False, default=str))
            lines.append("```")
            lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    blocks: list[str] = []
    print("=" * 72)
    print("T16.D.7.B — Hero 3 alternative metrics")
    print("=" * 72)

    # Load cache
    if not A5_CACHE.is_file():
        msg = f"BLOQUEO: A5 cache no encontrada en {A5_CACHE}"
        print(msg)
        OUT_MD.write_text(f"# T16.D.7.B\n\n{msg}\n", encoding="utf-8")
        return 1
    with open(A5_CACHE, encoding="utf-8") as f:
        a5_data = json.load(f)
    papers = a5_data.get("papers", []) or []
    print(f"[load] A5 cache: {len(papers)} papers, elapsed={a5_data.get('elapsed_seconds')}s")

    # Load gold
    try:
        gold_metrics_df = pd.read_csv(GOLD_METRICS, dtype={"covidence_id": str})
    except Exception as e:
        msg = f"BLOQUEO gold_metrics: {type(e).__name__}: {e}"
        print(msg)
        blocks.append(msg)
        gold_metrics_df = pd.DataFrame()

    try:
        arm_df = pd.read_csv(ARM_MAPPING, dtype=str)
    except Exception as e:
        msg = f"BLOQUEO arm_mapping: {type(e).__name__}: {e}"
        print(msg)
        blocks.append(msg)
        arm_df = pd.DataFrame()

    try:
        sample_df = pd.read_csv(SAMPLE_CSV, dtype={"covidence_id": str})
    except Exception as e:
        msg = f"BLOQUEO sample_50: {type(e).__name__}: {e}"
        print(msg)
        blocks.append(msg)
        sample_df = pd.DataFrame()

    # Build gold subset using the same logic as the hero3 evaluator
    include_ids = [str(p.get("pmid")) for p in papers]
    if gold_metrics_df.empty or arm_df.empty:
        gold_subset = pd.DataFrame()
        gold_debug: dict = {}
    else:
        gold_subset, gold_debug = _build_gold_subset(gold_metrics_df, arm_df, include_ids)
    print(f"[gold] subset rows: {len(gold_subset)}")

    # Step 0 schema
    schema = schema_block(gold_metrics_df) if not gold_metrics_df.empty else {
        "columns": [], "dtypes": {}, "sample_rows": [], "rows_total": 0,
    }

    # Metrics 1-4
    print("[metric_1] numeric_complete ...")
    m1 = compute_metric_1(papers)
    print(f"            pct={m1['pct_numeric_complete']:.4f} ({m1['numeric_complete']}/{m1['total_outcomes']})")
    print("[metric_2] effect+CI ...")
    m2 = compute_metric_2(papers)
    print(f"            pct={m2['pct_effect_and_ci']:.4f} ({m2['with_effect_and_ci']}/{m2['total_outcomes']})")
    print("[metric_3] ratio A5/gold primary comparisons ...")
    try:
        m3 = compute_metric_3(papers, gold_subset) if not gold_subset.empty else {
            "grouping_rule": "n/a (gold_subset empty)", "per_study": {},
        }
    except Exception as e:
        blocks.append(f"BLOQUEO metric_3: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        m3 = {"grouping_rule": "blocked", "per_study": {}}
    print("[metric_4] field-buckets ...")
    m4 = compute_metric_4(papers)
    print(f"            buckets={m4['overall_counts']}")

    # Step 5
    print("[metric_5] diagnose #13950 and #5298 ...")
    a5_tpl = _read_a5_prompt_templates()
    metric_5: list[dict] = []
    for cid in ("#13950", "#5298"):
        try:
            metric_5.append(diagnose_partial(cid, papers, sample_df))
        except Exception as e:
            blocks.append(f"BLOQUEO metric_5 {cid}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            metric_5.append({"covidence_id": cid, "error": f"{type(e).__name__}: {e}"})

    envelope = {
        "task": "T16.D.7.B Hero 3 alternative metrics",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "a5_cache_path": str(A5_CACHE.relative_to(PROJECT_ROOT)),
        "a5_cache_elapsed_seconds": a5_data.get("elapsed_seconds"),
        "n_papers": len(papers),
        "blocks": blocks,
        "schema": schema,
        "metric_1": m1,
        "metric_2": m2,
        "metric_3": m3,
        "metric_4": m4,
        "metric_5": metric_5,
        "a5_prompt_templates": a5_tpl,
        "gold_subset_debug": gold_debug if not gold_subset.empty else {},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False, default=str)
    write_md(envelope)

    print()
    print(f"[done] JSON: {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"       MD  : {OUT_MD.relative_to(PROJECT_ROOT)}")
    print(f"       blocks: {len(blocks)}")
    return 0 if not blocks else 0  # do not fail; report inline


if __name__ == "__main__":
    raise SystemExit(main())
