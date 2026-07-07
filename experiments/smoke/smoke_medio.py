# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Unified smoke test for the four agents over the medio sample (T16.E.1.fix).

Differences vs the per-agent smokes (smoke_a3/a4/a5/a6)
-------------------------------------------------------
Existing smokes use hand-picked synthetic / micro fixtures. This unified smoke
runs each agent on 2-3 rows drawn from the medio sample CSVs (built in
T16.E.1) so we validate that the evaluator + agent path does NOT break when
fed real medio data. Validation is intentionally relaxed: we check that the
agent returns a well-formed output (no exception, expected keys present),
not that it matches gold (which would be a full evaluation, not a smoke).

The Hero 1 path is the most important here: T16.E.1.fix added mixed-sample
support to eval_hero1_screening.py. The smoke runs A3 over 1 relevant + 2
irrelevant rows from sample_medio_a3.csv to exercise the new metric branches.

VRAM strategy: model loaded once via Agent3Screening, refs shared into A4/A5/A6
(same pattern as run_heroes_micro.py).

Output: outputs/eval/smoke/smoke_medio_{YYYYMMDD_HHMMSS}.json

This script does NOT modify agents/, prompts, config.yaml, .env.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

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

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "smoke"
MEDIO_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "samples" / "medio"
PDFS_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_todos"
GOLD_PATH = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "unified" / "gold_screening_unified.csv"

SMOKE_ID = "smoke_medio"
SEED = 20260518

# How many rows to draw per agent.
N_HERO1 = 3   # 1 relevant + 2 irrelevant
N_HERO2 = 3   # 1 include + 2 excludes
N_HERO3 = 2   # 2 includes (PDFs ~150s each)
N_HERO4 = 3   # 3 includes


def _pick_rows(
    df: pd.DataFrame,
    column: str,
    value: str,
    n: int,
    rng: random.Random,
) -> List[dict]:
    """Pick deterministic ``n`` rows where ``column == value``."""
    subset = df[df[column] == value].copy()
    subset = subset.sort_values(by="covidence_id", kind="mergesort").reset_index(drop=True)
    indices = list(range(len(subset)))
    rng.shuffle(indices)
    return subset.iloc[indices[:n]].to_dict("records")


def _build_hero1_papers(rng: random.Random, gold_idx) -> List[dict]:
    """Pick 1 relevant + 2 irrelevant rows from sample_medio_a3.csv."""
    df = pd.read_csv(MEDIO_DIR / "sample_medio_a3.csv", dtype={"covidence_id": str})
    rel_rows = _pick_rows(df, "human_label_ta", "relevant", 1, rng)
    irr_rows = _pick_rows(df, "human_label_ta", "irrelevant", N_HERO1 - 1, rng)
    rows = rel_rows + irr_rows
    papers = []
    for row in rows:
        cid = str(row["covidence_id"])
        title = ""
        abstract = ""
        if cid in gold_idx.index:
            gold_row = gold_idx.loc[cid]
            if isinstance(gold_row, pd.DataFrame):
                gold_row = gold_row.iloc[0]
            title = str(gold_row.get("title") or "").strip()
            abstract = str(gold_row.get("abstract") or "").strip()
            if abstract.lower() in ("nan", "none"):
                abstract = ""
            if title.lower() in ("nan", "none"):
                title = ""
        papers.append({
            "pmid": cid,
            "title": title,
            "abstract": abstract,
            "_gold_label": row["human_label_ta"],
        })
    return papers


def _build_hero2_papers(rng: random.Random, gold_idx) -> List[dict]:
    """Pick 1 include + 2 Wrong-age excludes (both common reasons)."""
    df = pd.read_csv(MEDIO_DIR / "sample_medio_a4.csv", dtype={"covidence_id": str})
    inc_rows = _pick_rows(df, "human_label_ft", "include", 1, rng)
    exc_subset = df[(df["human_label_ft"] == "exclude") & (df["exclusion_reason_ft"] == "Wrong age")]
    exc_subset = exc_subset.sort_values(by="covidence_id", kind="mergesort").reset_index(drop=True)
    idx = list(range(len(exc_subset)))
    rng.shuffle(idx)
    exc_rows = exc_subset.iloc[idx[:N_HERO2 - 1]].to_dict("records")
    rows = inc_rows + exc_rows
    papers = []
    for row in rows:
        cid = str(row["covidence_id"])
        pdf = row.get("pdf_filename") or ""
        pdf_path = PDFS_DIR / pdf
        text, err = _extract_pdf_text(pdf_path) if pdf else (None, "no_pdf")
        if text is None:
            papers.append({
                "pmid": cid,
                "_skip": True,
                "_skip_reason": err or "no_pdf",
                "_gold_decision": row.get("human_label_ft"),
                "_pdf_filename": pdf,
            })
            continue
        title = ""
        abstract = ""
        if cid in gold_idx.index:
            gold_row = gold_idx.loc[cid]
            if isinstance(gold_row, pd.DataFrame):
                gold_row = gold_row.iloc[0]
            title = str(gold_row.get("title") or "").strip()
            abstract = str(gold_row.get("abstract") or "").strip()
            if abstract.lower() in ("nan", "none"):
                abstract = ""
        papers.append({
            "pmid": cid,
            "title": title,
            "abstract": abstract,
            "full_text": text,
            "full_text_length": len(text),
            "text_source": "local_pdf",
            "year": "",
            "journal": "",
            "_gold_decision": row.get("human_label_ft"),
            "_pdf_filename": pdf,
        })
    return papers


def _build_pdf_papers(sample_csv: str, n: int, rng: random.Random, gold_idx) -> List[dict]:
    """Generic PDF-based loader for hero3/hero4 medio samples (all-include rows)."""
    df = pd.read_csv(MEDIO_DIR / sample_csv, dtype={"covidence_id": str})
    df = df.sort_values(by="covidence_id", kind="mergesort").reset_index(drop=True)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    rows = df.iloc[idx[:n]].to_dict("records")
    papers = []
    for row in rows:
        cid = str(row["covidence_id"])
        pdf = row.get("pdf_filename") or ""
        pdf_path = PDFS_DIR / pdf
        text, err = _extract_pdf_text(pdf_path) if pdf else (None, "no_pdf")
        if text is None:
            papers.append({
                "pmid": cid,
                "_skip": True,
                "_skip_reason": err or "no_pdf",
                "_pdf_filename": pdf,
            })
            continue
        title = ""
        abstract = ""
        authors = ""
        if cid in gold_idx.index:
            gold_row = gold_idx.loc[cid]
            if isinstance(gold_row, pd.DataFrame):
                gold_row = gold_row.iloc[0]
            title = str(gold_row.get("title") or "").strip()
            abstract = str(gold_row.get("abstract") or "").strip()
            authors = str(gold_row.get("authors") or "").strip()
            if abstract.lower() in ("nan", "none"):
                abstract = ""
        papers.append({
            "pmid": cid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": "",
            "journal": "",
            "doi": "",
            "full_text": text,
            "full_text_length": len(text),
            "text_source": "local_pdf",
            "_pdf_filename": pdf,
        })
    return papers


# ──────────────────────────────────────────────────────────────────────────────
# Per-agent runs
# ──────────────────────────────────────────────────────────────────────────────

def _run_hero1(papers: List[dict], agent) -> dict:
    """Run A3.screen_batch; validate well-formed output for each pmid."""
    t0 = time.time()
    try:
        out = agent.screen_batch(papers, PAULA_SPORT_PROTOCOL)
        ok = True
        err_msg = ""
    except Exception as exc:
        out = {"include": [], "exclude": [], "uncertain": [], "stats": {}}
        ok = False
        err_msg = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.time() - t0, 2)

    a3_by_id: dict = {}
    for bucket in ("include", "exclude", "uncertain"):
        for r in out.get(bucket, []):
            a3_by_id[r.get("pmid", "")] = r

    per_paper = []
    well_formed = 0
    for p in papers:
        a3 = a3_by_id.get(p["pmid"])
        decision = (a3 or {}).get("decision", "")
        # Well-formed means: A3 returned something with a recognised decision.
        is_well_formed = decision in ("INCLUDE", "EXCLUDE", "UNCERTAIN")
        if is_well_formed:
            well_formed += 1
        per_paper.append({
            "pmid": p["pmid"],
            "gold_label": p.get("_gold_label"),
            "title": _truncate(p.get("title", ""), 80),
            "decision": decision,
            "well_formed": is_well_formed,
        })

    return {
        "agent": "A3",
        "n_inputs": len(papers),
        "n_well_formed": well_formed,
        "all_well_formed": ok and well_formed == len(papers),
        "elapsed_seconds": elapsed,
        "error": err_msg,
        "per_paper": per_paper,
    }


def _run_hero2(papers: List[dict], agent) -> dict:
    """Run A4.screen_fulltext on the non-skipped papers."""
    runnable = [p for p in papers if not p.get("_skip")]
    skipped = [p for p in papers if p.get("_skip")]
    t0 = time.time()
    if not runnable:
        return {
            "agent": "A4",
            "n_inputs": len(papers),
            "n_skipped": len(skipped),
            "n_well_formed": 0,
            "all_well_formed": False,
            "elapsed_seconds": 0.0,
            "error": "no runnable papers",
            "per_paper": [{"pmid": p["pmid"], "skipped": True} for p in papers],
        }
    try:
        out = agent.screen_fulltext(runnable, PAULA_SPORT_PROTOCOL)
        ok = True
        err_msg = ""
    except Exception as exc:
        out = {"papers": runnable, "include": [], "exclude": [], "stats": {}}
        ok = False
        err_msg = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.time() - t0, 2)

    a4_by_id = {p["pmid"]: p for p in out.get("papers", runnable)}
    per_paper = []
    well_formed = 0
    for p in papers:
        if p.get("_skip"):
            per_paper.append({"pmid": p["pmid"], "skipped": True,
                              "reason": p.get("_skip_reason")})
            continue
        a4p = a4_by_id.get(p["pmid"], p)
        decision = (a4p.get("fulltext_decision") or "").upper()
        is_well_formed = decision in ("INCLUDE", "EXCLUDE", "UNCERTAIN")
        if is_well_formed:
            well_formed += 1
        per_paper.append({
            "pmid": p["pmid"],
            "gold_decision": p.get("_gold_decision"),
            "decision": decision,
            "well_formed": is_well_formed,
        })
    return {
        "agent": "A4",
        "n_inputs": len(papers),
        "n_runnable": len(runnable),
        "n_skipped": len(skipped),
        "n_well_formed": well_formed,
        "all_well_formed": ok and well_formed == len(runnable),
        "elapsed_seconds": elapsed,
        "error": err_msg,
        "per_paper": per_paper,
    }


def _run_hero3(papers: List[dict], agent) -> dict:
    """Run A5.extract_all on includes; validate at least one outcome per paper."""
    runnable = [p for p in papers if not p.get("_skip")]
    skipped = [p for p in papers if p.get("_skip")]
    t0 = time.time()
    if not runnable:
        return {
            "agent": "A5",
            "n_inputs": len(papers),
            "n_skipped": len(skipped),
            "n_well_formed": 0,
            "all_well_formed": False,
            "elapsed_seconds": 0.0,
            "error": "no runnable papers",
            "per_paper": [{"pmid": p["pmid"], "skipped": True} for p in papers],
        }
    try:
        out = agent.extract_all(runnable, PAULA_SPORT_PROTOCOL)
        ok = True
        err_msg = ""
    except Exception as exc:
        out = {"papers": []}
        ok = False
        err_msg = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.time() - t0, 2)

    extracted_by_id = {p.get("pmid"): p for p in out.get("papers", [])}
    per_paper = []
    well_formed = 0
    for p in papers:
        if p.get("_skip"):
            per_paper.append({"pmid": p["pmid"], "skipped": True,
                              "reason": p.get("_skip_reason")})
            continue
        ep = extracted_by_id.get(p["pmid"], {})
        outcomes = ep.get("outcomes") or []
        quality = ep.get("extraction_quality", "")
        is_well_formed = quality != "failed" and isinstance(outcomes, list)
        if is_well_formed:
            well_formed += 1
        per_paper.append({
            "pmid": p["pmid"],
            "extraction_quality": quality,
            "n_outcomes": len(outcomes),
            "well_formed": is_well_formed,
        })
    return {
        "agent": "A5",
        "n_inputs": len(papers),
        "n_runnable": len(runnable),
        "n_skipped": len(skipped),
        "n_well_formed": well_formed,
        "all_well_formed": ok and well_formed == len(runnable),
        "elapsed_seconds": elapsed,
        "error": err_msg,
        "per_paper": per_paper,
    }


def _run_hero4(papers: List[dict], agent) -> dict:
    """Run A6.assess_all; validate that overall + domains are present."""
    runnable = [p for p in papers if not p.get("_skip")]
    skipped = [p for p in papers if p.get("_skip")]
    t0 = time.time()
    if not runnable:
        return {
            "agent": "A6",
            "n_inputs": len(papers),
            "n_skipped": len(skipped),
            "n_well_formed": 0,
            "all_well_formed": False,
            "elapsed_seconds": 0.0,
            "error": "no runnable papers",
            "per_paper": [{"pmid": p["pmid"], "skipped": True} for p in papers],
        }
    try:
        out = agent.assess_all(runnable, PAULA_SPORT_PROTOCOL)
        ok = True
        err_msg = ""
    except Exception as exc:
        out = {"papers": []}
        ok = False
        err_msg = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.time() - t0, 2)

    assessed_by_id = {p.get("pmid"): p for p in out.get("papers", [])}
    per_paper = []
    well_formed = 0
    for p in papers:
        if p.get("_skip"):
            per_paper.append({"pmid": p["pmid"], "skipped": True,
                              "reason": p.get("_skip_reason")})
            continue
        ap = assessed_by_id.get(p["pmid"], {})
        quality = ap.get("quality_assessment") or {}
        tool = quality.get("tool", "")
        overall = quality.get("overall", "")
        domains = quality.get("domains") or quality.get("items") or []
        has_domains = (isinstance(domains, dict) and len(domains) > 0) or (
            isinstance(domains, list) and len(domains) > 0
        )
        is_well_formed = tool in ("RoB2", "ROBINS-I", "JBI") and bool(overall) and has_domains
        if is_well_formed:
            well_formed += 1
        per_paper.append({
            "pmid": p["pmid"],
            "tool": tool,
            "overall": overall,
            "n_domains": len(domains) if hasattr(domains, "__len__") else 0,
            "well_formed": is_well_formed,
        })
    return {
        "agent": "A6",
        "n_inputs": len(papers),
        "n_runnable": len(runnable),
        "n_skipped": len(skipped),
        "n_well_formed": well_formed,
        "all_well_formed": ok and well_formed == len(runnable),
        "elapsed_seconds": elapsed,
        "error": err_msg,
        "per_paper": per_paper,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _build_shared_agents():
    """Same VRAM-sharing pattern as run_heroes_micro / run_heroes_medio."""
    from agents.agent3_screening import Agent3Screening
    from agents.agent4_fulltext import Agent4FullText
    from agents.agent5_extraction import Agent5Extraction
    from agents.agent6_quality import Agent6Quality

    print("[bootstrap] Loading shared model via Agent3...")
    a3 = Agent3Screening()
    a3.load_model()
    a4 = Agent4FullText(ensemble_size=1)
    a5 = Agent5Extraction()
    a6 = Agent6Quality()
    for child in (a4, a5, a6):
        child.model = a3.model
        child.tokenizer = a3.tokenizer
        child._loaded_path = a3._loaded_path
        child._has_chat_template = a3._has_chat_template
        child.model_path = a3._loaded_path or a3.model_path
    return a3, a4, a5, a6


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified smoke over medio sample.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _seed_torch(42)
    rng = random.Random(SEED)
    gold_df = pd.read_csv(GOLD_PATH, dtype={"covidence_id": str})
    gold_idx = gold_df.set_index("covidence_id")

    print("=" * 72)
    print("SMOKE MEDIO — 4 agentes sobre 2-3 filas del sample medio cada uno")
    print("=" * 72)
    print(f"Seed: {SEED}; GPU: {_gpu_info()}")

    # Build inputs first (CPU-only, fast, no GPU needed).
    print("\n[prep] Cargando filas y extrayendo PDFs...")
    h1_papers = _build_hero1_papers(rng, gold_idx)
    h2_papers = _build_hero2_papers(rng, gold_idx)
    h3_papers = _build_pdf_papers("sample_medio_a5.csv", N_HERO3, rng, gold_idx)
    h4_papers = _build_pdf_papers("sample_medio_a6.csv", N_HERO4, rng, gold_idx)

    print(f"  Hero1 papers (A3): {len(h1_papers)} (1 rel + {N_HERO1-1} irr)")
    print(f"  Hero2 papers (A4): {len(h2_papers)} (1 inc + {N_HERO2-1} exc)")
    print(f"  Hero3 papers (A5): {len(h3_papers)} (todos includes)")
    print(f"  Hero4 papers (A6): {len(h4_papers)} (todos includes)")

    # Bootstrap shared model.
    try:
        a3, a4, a5, a6 = _build_shared_agents()
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n[bootstrap] FALLO: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
        return 2

    # Run each hero.
    print("\n[hero1] Ejecutando A3.screen_batch...")
    r1 = _run_hero1(h1_papers, a3)
    print(f"  -> {r1['n_well_formed']}/{r1['n_inputs']} well-formed ({r1['elapsed_seconds']}s)")

    print("\n[hero2] Ejecutando A4.screen_fulltext...")
    r2 = _run_hero2(h2_papers, a4)
    print(f"  -> {r2['n_well_formed']}/{r2.get('n_runnable', 0)} well-formed "
          f"({r2['elapsed_seconds']}s, {r2.get('n_skipped', 0)} skipped)")

    print("\n[hero3] Ejecutando A5.extract_all...")
    r3 = _run_hero3(h3_papers, a5)
    print(f"  -> {r3['n_well_formed']}/{r3.get('n_runnable', 0)} well-formed "
          f"({r3['elapsed_seconds']}s, {r3.get('n_skipped', 0)} skipped)")

    print("\n[hero4] Ejecutando A6.assess_all...")
    r4 = _run_hero4(h4_papers, a6)
    print(f"  -> {r4['n_well_formed']}/{r4.get('n_runnable', 0)} well-formed "
          f"({r4['elapsed_seconds']}s, {r4.get('n_skipped', 0)} skipped)")

    # Free VRAM.
    try:
        a3.free_vram()
    except Exception:
        pass

    all_ok = all(r["all_well_formed"] for r in (r1, r2, r3, r4))
    envelope = {
        "smoke_id": SMOKE_ID,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
        "all_ok": all_ok,
        "environment": {
            "python_version": platform.python_version(),
            "gpu_info": _gpu_info(),
        },
        "hero1": r1,
        "hero2": r2,
        "hero3": r3,
        "hero4": r4,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{SMOKE_ID}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False, default=str)

    print()
    print("=" * 72)
    print("RESUMEN SMOKE MEDIO")
    print("=" * 72)
    for hero_label, rec in (("Hero1 A3", r1), ("Hero2 A4", r2),
                            ("Hero3 A5", r3), ("Hero4 A6", r4)):
        flag = "OK" if rec["all_well_formed"] else "FAIL"
        print(f"  {hero_label}: [{flag}] {rec['n_well_formed']}/{rec['n_inputs']} "
              f"well-formed ({rec['elapsed_seconds']}s)")
    print()
    print(f"Output: {out_path}")
    print("[VERDE]" if all_ok else "[ROJO] revisar arriba")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
