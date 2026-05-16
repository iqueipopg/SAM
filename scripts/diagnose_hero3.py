"""
T16.D.7.A — Hero 3 deep-dive diagnostic.

The hero3_results JSON only preserves matched A5 outcomes (via
``raw_predictions[i].predicted``) — A5 outcomes that fail to align with any
gold row are lost. To distinguish (a) matching_problem from (b) scope_mismatch
from (c) garbage we need the FULL per-study A5 outcome list, so this script
re-runs Agent5Extraction on the same hero3 inputs and caches the result.

Constraints honoured:
  - NO modifications to ``agents/agent5_extraction.py``
  - NO modifications to ``experiments/eval_hero3_extraction.py``
  - NO modifications to prompts
  - Imports the existing helpers but only as a consumer; does NOT touch them

Inputs:
  - data/gold_standard/deporte/samples/micro/sample_50.csv (hero3 rows)
  - data/gold_standard/deporte/unified/gold_screening_unified.csv
  - data/gold_standard_normalized/gold_metrics.csv
  - data/gold_standard/deporte/unified/arm_to_parent_mapping.csv
  - data/gold_standard/deporte/pdfs_todos/*.pdf
  - latest outputs/eval/micro/hero3_results_*.json (for cross-reference only)

Cache:
  - outputs/eval/micro/_a5_full_outputs.json
    Re-used if newer than CACHE_MAX_AGE_SECONDS. Pass --force to refresh.

Outputs:
  - outputs/eval/micro/hero3_diagnosis.md
  - outputs/eval/micro/hero3_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._eval_common import (  # noqa: E402
    PAULA_SPORT_PROTOCOL,
    extract_pdf_text,
    seed_torch,
)
from experiments.eval_hero3_extraction import (  # noqa: E402
    _filter_hero3_includes,
    _build_gold_subset,
)


# ──────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_CSV = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "samples" / "micro" / "sample_50.csv"
GOLD_SCREENING = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "unified" / "gold_screening_unified.csv"
GOLD_METRICS = PROJECT_ROOT / "data" / "gold_standard_normalized" / "gold_metrics.csv"
ARM_MAPPING = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "unified" / "arm_to_parent_mapping.csv"
EXTRACTED_METADATA = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "unified" / "extracted_metadata.csv"
PDFS_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_todos"
OUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "micro"
A5_CACHE = OUT_DIR / "_a5_full_outputs.json"
CACHE_MAX_AGE_SECONDS = 6 * 3600  # 6 h

# difflib SequenceMatcher.ratio() >= FUZZY_THRESHOLD treated as a usable match.
# 0.70 is the value the briefing requested.
FUZZY_THRESHOLD = 0.70

NUMERIC_FIELDS = (
    "n_intervention",
    "n_control",
    "mean_intervention",
    "mean_control",
    "sd_intervention",
    "sd_control",
    "effect_size_value",
)

# Mapping gold field -> A5 outcome key (mirrors eval_hero3_extraction.FIELD_MAPPING).
FIELD_PAIRS = (
    ("N_GC", "n_control"),
    ("N_GI", "n_intervention"),
    ("X_GC", "mean_control"),
    ("X_GI", "mean_intervention"),
    ("SD_GC", "sd_control"),
    ("SD_GI", "sd_intervention"),
)


_PUNC_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _normalize_name(s: object) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    txt = str(s or "").lower().strip()
    txt = txt.translate(_PUNC_TABLE)
    return re.sub(r"\s+", " ", txt).strip()


def _outcome_is_garbage(o: dict) -> tuple[bool, str]:
    """Return ``(is_garbage, reason)`` for an A5 outcome.

    Heuristics:
      - empty / "nan" / "none" name
      - OR zero informative numeric fields
    """
    name = (o.get("name") or "").strip()
    nl = name.lower()
    if not name or nl in ("nan", "none", "null", "n/a"):
        return True, "empty_name"
    n_with_value = 0
    for f in NUMERIC_FIELDS:
        v = o.get(f)
        if v is None:
            continue
        if isinstance(v, float) and v != v:  # NaN
            continue
        if isinstance(v, str) and v.strip().lower() in ("", "nan", "none", "null", "n/a"):
            continue
        n_with_value += 1
    if n_with_value == 0:
        return True, "no_numeric_data"
    return False, ""


def _best_match(a5_name: str, gold_names: list[str]) -> tuple[Optional[int], float, str]:
    """Return ``(best_gold_idx, score, match_type)`` for ``a5_name`` vs gold list.

    match_type ∈ {exact, substring, fuzzy, weak, no_a5_name, no_gold}.
    """
    a5n = _normalize_name(a5_name)
    if not a5n:
        return None, 0.0, "no_a5_name"
    if not gold_names:
        return None, 0.0, "no_gold"
    best_idx: Optional[int] = None
    best_score = 0.0
    best_type = "weak"
    for i, g in enumerate(gold_names):
        gn = _normalize_name(g)
        if not gn:
            continue
        if a5n == gn:
            return i, 1.0, "exact"
        if a5n in gn or gn in a5n:
            if 0.9 > best_score:
                best_idx, best_score, best_type = i, 0.9, "substring"
            continue
        r = SequenceMatcher(None, a5n, gn).ratio()
        if r > best_score:
            best_idx, best_score = i, r
            best_type = "fuzzy" if r >= FUZZY_THRESHOLD else "weak"
    return best_idx, best_score, best_type


def _coerce_float(v: object) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "n/a", "null"):
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Build paper dicts (mirror hero3 evaluator logic)
# ──────────────────────────────────────────────────────────────────────────────

def build_papers() -> tuple[list[dict], list[dict]]:
    sample = pd.read_csv(SAMPLE_CSV, dtype={"covidence_id": str})
    gold_scr = pd.read_csv(GOLD_SCREENING, dtype={"covidence_id": str})
    includes = _filter_hero3_includes(sample)
    gold_idx = gold_scr.set_index("covidence_id")

    # T16.D.9.C — apply the same canonical-PDF override the hero3 evaluator
    # uses, so the cache regenerated here mirrors the evaluator behaviour.
    canonical_pdf_by_cid: dict[str, str] = {}
    if EXTRACTED_METADATA.exists():
        em_df = pd.read_csv(EXTRACTED_METADATA, dtype={"covidence_id": str})
        if "pdf_used" in em_df.columns:
            for _, em_row in em_df.iterrows():
                cid_em = str(em_row.get("covidence_id", "") or "").strip()
                pdf_used = em_row.get("pdf_used")
                if cid_em and pdf_used is not None and not pd.isna(pdf_used):
                    canonical_pdf_by_cid[cid_em] = str(pdf_used)

    papers: list[dict] = []
    skipped: list[dict] = []

    for _, row in includes.iterrows():
        cid = str(row["covidence_id"])
        pdf_filename_sample = row.get("pdf_filename")
        has_pdf = bool(row.get("has_pdf", False))
        # Canonical override (T16.D.9.C)
        canonical_pdf = canonical_pdf_by_cid.get(cid)
        if canonical_pdf is not None and pd.isna(canonical_pdf):
            canonical_pdf = None
        if canonical_pdf:
            pdf_filename = canonical_pdf
            has_pdf = True
        else:
            pdf_filename = pdf_filename_sample
        if not has_pdf or pdf_filename is None or pd.isna(pdf_filename):
            skipped.append({"covidence_id": cid, "reason": "no_pdf_available"})
            continue
        pdf_path = PDFS_DIR / str(pdf_filename)
        text, err = extract_pdf_text(pdf_path)
        if text is None:
            skipped.append({
                "covidence_id": cid,
                "reason": "pdf_extraction_failed",
                "err": err,
            })
            continue
        title = ""
        if cid in gold_idx.index:
            gr = gold_idx.loc[cid]
            if isinstance(gr, pd.DataFrame):
                gr = gr.iloc[0]
            title = str(gr.get("title", "") or "").strip()
            if title.lower() in ("nan", "none"):
                title = ""
        papers.append({
            "pmid": cid,
            "title": title,
            "abstract": "",
            "authors": [],
            "year": "",
            "journal": "",
            "doi": "",
            "full_text": text,
            "full_text_length": len(text),
            "text_source": "local_pdf",
            "_pdf_filename": str(pdf_filename),
        })
    return papers, skipped


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Cached A5 invocation
# ──────────────────────────────────────────────────────────────────────────────

def load_or_run_a5(papers: list[dict], force: bool) -> dict:
    if not force and A5_CACHE.is_file():
        age = time.time() - A5_CACHE.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            print(f"[cache] Reusing A5 outputs ({A5_CACHE}, age={int(age)}s)")
            with open(A5_CACHE) as f:
                return json.load(f)
        print(f"[cache] Cache stale (age={int(age)}s > {CACHE_MAX_AGE_SECONDS}s)")
    print(f"[run] Loading Agent5 and running extract_all on {len(papers)} papers...")
    seed_torch(42)
    from agents.agent5_extraction import Agent5Extraction
    agent = Agent5Extraction()
    t0 = time.time()
    out = agent.extract_all(papers, PAULA_SPORT_PROTOCOL)
    elapsed = round(time.time() - t0, 2)
    print(f"[run] A5 finished in {elapsed}s")
    try:
        agent.free_vram()
    except Exception:
        pass
    out_persistable = {
        "elapsed_seconds": elapsed,
        "n_papers_in": len(papers),
        "papers": out.get("papers", []),
        "stats": out.get("stats", {}),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(A5_CACHE, "w", encoding="utf-8") as f:
        json.dump(out_persistable, f, indent=2, ensure_ascii=False, default=str)
    return out_persistable


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Per-study diagnostic matching
# ──────────────────────────────────────────────────────────────────────────────

def diagnose_study(cid: str, a5_outcomes: list[dict], gold_rows: list[dict]) -> dict:
    gold_names = [str(r.get("outcome_name") or "") for r in gold_rows]

    a5_records: list[dict] = []
    for j, ao in enumerate(a5_outcomes):
        name = ao.get("name", "")
        gi, score, mtype = _best_match(name, gold_names)
        is_garbage, garbage_reason = _outcome_is_garbage(ao)
        a5_records.append({
            "a5_idx": j,
            "a5_name": name,
            "best_gold_idx": gi,
            "best_gold_name": (gold_names[gi] if gi is not None else None),
            "score": round(score, 3),
            "match_type": mtype,
            "is_garbage": is_garbage,
            "garbage_reason": garbage_reason,
            "a5_outcome": {
                "name": ao.get("name"),
                "measure_type": ao.get("measure_type") or ao.get("effect_size_type"),
                "n_intervention": ao.get("n_intervention"),
                "n_control": ao.get("n_control"),
                "mean_intervention": ao.get("mean_intervention"),
                "mean_control": ao.get("mean_control"),
                "sd_intervention": ao.get("sd_intervention"),
                "sd_control": ao.get("sd_control"),
                "effect_size_value": ao.get("effect_size_value"),
                "numeric_complete": ao.get("numeric_complete"),
            },
        })

    matched_gold_idx = {
        r["best_gold_idx"]
        for r in a5_records
        if r["best_gold_idx"] is not None and r["score"] >= FUZZY_THRESHOLD
    }
    unmatched_gold = [
        {"idx": i, "row": r}
        for i, r in enumerate(gold_rows)
        if i not in matched_gold_idx
    ]

    n_a5 = len(a5_outcomes)
    n_gold = len(gold_rows)

    if not gold_rows:
        classification = "no_gold"
    elif n_a5 == 0:
        classification = "garbage"
    else:
        n_garbage = sum(1 for r in a5_records if r["is_garbage"])
        n_fuzzy = sum(1 for r in a5_records if r["score"] >= FUZZY_THRESHOLD)
        n_exact = sum(1 for r in a5_records if r["match_type"] == "exact")
        n_substring = sum(1 for r in a5_records if r["match_type"] == "substring")
        n_distinct = sum(
            1 for r in a5_records
            if r["score"] < FUZZY_THRESHOLD and not r["is_garbage"]
        )
        pct_garbage = n_garbage / n_a5
        pct_fuzzy = n_fuzzy / n_a5
        pct_distinct = n_distinct / n_a5

        if pct_garbage >= 0.5:
            classification = "garbage"
        elif pct_fuzzy >= 0.7 and (n_exact + n_substring) < n_fuzzy:
            classification = "matching_problem"
        elif pct_distinct >= 0.7:
            classification = "scope_mismatch"
        else:
            classification = "mixed"

    return {
        "covidence_id": cid,
        "classification": classification,
        "n_outcomes_a5": n_a5,
        "n_outcomes_gold": n_gold,
        "a5_records": a5_records,
        "unmatched_gold_count": len(unmatched_gold),
        "unmatched_gold_names": [
            str(g["row"].get("outcome_name") or "") for g in unmatched_gold
        ],
        "gold_rows_snapshot": [
            {
                "outcome_name": str(g.get("outcome_name") or ""),
                "timepoint": str(g.get("timepoint") or ""),
                "instrument": str(g.get("instrument") or ""),
                "N_GC": _coerce_float(g.get("N_GC")),
                "N_GI": _coerce_float(g.get("N_GI")),
                "X_GC": _coerce_float(g.get("X_GC")),
                "X_GI": _coerce_float(g.get("X_GI")),
                "SD_GC": _coerce_float(g.get("SD_GC")),
                "SD_GI": _coerce_float(g.get("SD_GI")),
            }
            for g in gold_rows
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Markdown report
# ──────────────────────────────────────────────────────────────────────────────

def write_markdown(path: Path, summary: dict, per_study: list[dict], gold_debug: dict, a5_meta: dict, skipped: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Hero 3 deep-dive diagnóstico — T16.D.7.A")
    lines.append("")
    lines.append(f"- Timestamp: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    lines.append(f"- Fuzzy threshold: `{FUZZY_THRESHOLD}`")
    lines.append(f"- A5 cache reused: `{a5_meta.get('reused')}`")
    lines.append(f"- A5 elapsed: `{a5_meta.get('elapsed_seconds')}s`")
    lines.append(f"- A5 papers: `{a5_meta.get('n_papers')}`")
    lines.append(f"- Skipped (pre-A5): `{len(skipped)}`")
    for s in skipped:
        lines.append(f"  - {s.get('covidence_id')}: {s.get('reason')}")
    lines.append("")
    lines.append("## Resumen ejecutivo")
    lines.append("")
    lines.append("| Categoría | N estudios |")
    lines.append("|---|---:|")
    for k in ("matching_problem", "scope_mismatch", "garbage", "mixed", "no_gold"):
        lines.append(f"| {k} | {summary.get(k, 0)} |")
    lines.append("")
    total = sum(summary.values())
    lines.append(f"Total estudios diagnosticados: **{total}**")
    lines.append("")
    lines.append("**Reglas de clasificación**:")
    lines.append("- `garbage` si A5 tiene 0 outcomes O ≥50% de outcomes con name vacío / sin datos numéricos")
    lines.append("- `matching_problem` si ≥70% A5 outcomes fuzzy-match (score≥0.7) con gold pero exact/substring < fuzzy")
    lines.append("- `scope_mismatch` si ≥70% A5 outcomes son distintos (score<0.7) y no son garbage")
    lines.append("- `mixed` si combinación")
    lines.append("- `no_gold` si gold_subset vacío para el estudio")
    lines.append("")
    lines.append("## Por estudio")
    lines.append("")
    for st in per_study:
        cid = st["covidence_id"]
        lines.append(f"### {cid} — `{st['classification']}`")
        lines.append("")
        lines.append(f"- A5 outcomes: **{st['n_outcomes_a5']}** | Gold outcomes: **{st['n_outcomes_gold']}**")
        lines.append(f"- Gold sin match A5 (score<{FUZZY_THRESHOLD}): **{st['unmatched_gold_count']}**")
        if st["unmatched_gold_names"]:
            sample = ", ".join(repr(n)[:50] for n in st["unmatched_gold_names"][:6])
            lines.append(f"  - Names (primeros 6): {sample}")
        lines.append("")
        if st["a5_records"]:
            lines.append("**Tabla A5 → mejor gold match**")
            lines.append("")
            lines.append("| # | A5 name | mejor gold | score | type | garbage? |")
            lines.append("|---:|---|---|---:|---|---|")
            for r in st["a5_records"]:
                lines.append(
                    f"| {r['a5_idx']} | {(r['a5_name'] or '')[:60]} | "
                    f"{(r['best_gold_name'] or '')[:50]} | {r['score']:.3f} | "
                    f"{r['match_type']} | "
                    f"{'YES (' + r['garbage_reason'] + ')' if r['is_garbage'] else 'no'} |"
                )
            lines.append("")

            # Side-by-side numeric for matches above threshold
            usable = [r for r in st["a5_records"] if r["score"] >= FUZZY_THRESHOLD and r["best_gold_idx"] is not None]
            if usable:
                lines.append("**Comparación numérica (score ≥ fuzzy_threshold):**")
                lines.append("")
                lines.append("| A5 | gold | N_GC | N_GI | X_GC | X_GI | SD_GC | SD_GI |")
                lines.append("|---|---|---|---|---|---|---|---|")
                for r in usable:
                    gi = r["best_gold_idx"]
                    g = st["gold_rows_snapshot"][gi]
                    ao = r["a5_outcome"]
                    def fmt(v):
                        return "—" if v is None else (f"{v:.3g}" if isinstance(v, (int, float)) else str(v))
                    lines.append(
                        f"| A5#{r['a5_idx']} '{(r['a5_name'] or '')[:30]}' | "
                        f"'{(g['outcome_name'] or '')[:30]}' | "
                        f"g={fmt(g['N_GC'])} / a={fmt(ao.get('n_control'))} | "
                        f"g={fmt(g['N_GI'])} / a={fmt(ao.get('n_intervention'))} | "
                        f"g={fmt(g['X_GC'])} / a={fmt(ao.get('mean_control'))} | "
                        f"g={fmt(g['X_GI'])} / a={fmt(ao.get('mean_intervention'))} | "
                        f"g={fmt(g['SD_GC'])} / a={fmt(ao.get('sd_control'))} | "
                        f"g={fmt(g['SD_GI'])} / a={fmt(ao.get('sd_intervention'))} |"
                    )
                lines.append("")
        else:
            lines.append("**A5 sin outcomes extraídos.**")
            lines.append("")

    # Raw A5 examples
    lines.append("## Ejemplos crudos de outputs A5 (3-5 primeros)")
    lines.append("")
    shown = 0
    for st in per_study:
        for r in st["a5_records"]:
            if shown >= 5:
                break
            lines.append(f"### {st['covidence_id']} — A5 outcome #{r['a5_idx']}")
            lines.append("```json")
            lines.append(json.dumps(r["a5_outcome"], indent=2, ensure_ascii=False, default=str))
            lines.append("```")
            lines.append("")
            shown += 1
        if shown >= 5:
            break

    lines.append("## Gold subset debug")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(gold_debug, indent=2, ensure_ascii=False, default=str))
    lines.append("```")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T16.D.7.A — Hero 3 deep-dive diagnostic.")
    p.add_argument("--force", action="store_true", help="Ignore A5 cache and re-run extraction.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print("=" * 72)
    print("T16.D.7.A — Hero 3 deep-dive diagnostic")
    print("=" * 72)

    print("\n[1/5] Building papers from sample_50 hero3 includes...")
    papers, skipped = build_papers()
    print(f"      Papers built: {len(papers)} | Skipped: {len(skipped)}")
    for s in skipped:
        print(f"        - {s}")

    print("\n[2/5] Loading or running A5 extraction...")
    cache_existed = A5_CACHE.is_file() and not args.force
    a5_out = load_or_run_a5(papers, force=args.force)
    a5_by_id = {str(p.get("pmid", "")): p for p in a5_out.get("papers", [])}
    print(f"      A5 papers: {len(a5_by_id)}")
    a5_meta = {
        "reused": cache_existed and (time.time() - A5_CACHE.stat().st_mtime) < CACHE_MAX_AGE_SECONDS,
        "elapsed_seconds": a5_out.get("elapsed_seconds"),
        "n_papers": len(a5_by_id),
    }

    print("\n[3/5] Loading gold_metrics + arm_to_parent_mapping...")
    gold_metrics = pd.read_csv(GOLD_METRICS, dtype={"covidence_id": str})
    arm_df = pd.read_csv(ARM_MAPPING, dtype=str)
    include_ids = [p["pmid"] for p in papers]
    gold_subset, gold_debug = _build_gold_subset(gold_metrics, arm_df, include_ids)
    print(f"      Gold subset rows: {len(gold_subset)}")
    print(f"      Gold subset by parent: {gold_debug.get('subset_by_parent')}")

    print("\n[4/5] Running per-study diagnostic matching...")
    gold_by_cid: dict[str, list[dict]] = defaultdict(list)
    for _, row in gold_subset.iterrows():
        parent = str(row["_effective_parent"])
        gold_by_cid[parent].append(row.to_dict())

    summary = {"matching_problem": 0, "scope_mismatch": 0, "garbage": 0, "mixed": 0, "no_gold": 0}
    per_study: list[dict] = []
    for cid in include_ids:
        a5_paper = a5_by_id.get(cid) or {}
        a5_outcomes = a5_paper.get("outcomes") or []
        gold_rows = gold_by_cid.get(cid, [])
        study = diagnose_study(cid, a5_outcomes, gold_rows)
        summary[study["classification"]] += 1
        per_study.append(study)
        print(
            f"      {cid:8} classification={study['classification']:18} "
            f"A5={study['n_outcomes_a5']:3} gold={study['n_outcomes_gold']:3} "
            f"unmatched_gold={study['unmatched_gold_count']:3}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "hero3_diagnosis.json"
    md_path = OUT_DIR / "hero3_diagnosis.md"
    envelope = {
        "task": "T16.D.7.A Hero 3 deep-dive diagnostic",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fuzzy_threshold": FUZZY_THRESHOLD,
        "a5_meta": a5_meta,
        "skipped_pre_a5": skipped,
        "summary": summary,
        "per_study": per_study,
        "gold_subset_debug": gold_debug,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False, default=str)
    write_markdown(md_path, summary, per_study, gold_debug, a5_meta, skipped)

    print(f"\n[5/5] Output JSON : {json_path.relative_to(PROJECT_ROOT)}")
    print(f"           MD   : {md_path.relative_to(PROJECT_ROOT)}")
    print(f"\nSUMMARY: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
