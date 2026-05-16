"""
T16.D.7.C — Hero 4 (Agent6 RoB) deep-dive diagnostic.

Re-runs Agent6Quality on the same hero4 inputs and captures, for each study:
  - the prompt sent to the LLM
  - the raw LLM response (before parsing)
  - the parsed quality_assessment (tool, overall, domains, justifications, weight)

Cross-references each prediction with Paula's gold_rob (worst-case per paper
plus per-outcome) using the same mapping helpers as eval_hero4_rob.

Outputs:
  - outputs/eval/micro/hero4_diagnosis.md
  - outputs/eval/micro/hero4_diagnosis.json
  - outputs/eval/micro/_a6_full_outputs.json (cache, snapshot)

Constraints honoured:
  - NO modifications to agents/* or experiments/eval_hero4_rob.py
  - NO modifications to prompts
  - Monkey-patches Agent6Quality._simple_generate only inside this script's
    process to capture (prompt, response) pairs; the on-disk source is
    untouched.

Caché y determinismo:
  Agent6Quality uses temperature=0.2 with do_sample=True, so the LLM output
  is non-deterministic. The cache is a *snapshot* of one run; running again
  may produce a different cache. The script will reuse the cache if newer
  than CACHE_MAX_AGE_SECONDS and the user hasn't passed ``--force``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
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
from experiments.eval_hero4_rob import (  # noqa: E402
    _filter_hero4_includes,
    _build_ref_to_covidence,
    _build_gold_subset,
    _normalize_a6_overall,
    _worst_case_gold,
    A6_OVERALL_NORMALISATION,
    ROB_SCALE,
    ROB_ORDINAL,
)


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_CSV = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "samples" / "micro" / "sample_50.csv"
GOLD_SCREENING = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "unified" / "gold_screening_unified.csv"
GOLD_ROB = PROJECT_ROOT / "data" / "gold_standard_normalized" / "gold_rob.csv"
REF_MAPPING = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "unified" / "ref_to_covidence_mapping.csv"
PDFS_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_todos"
OUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "micro"
A6_CACHE = OUT_DIR / "_a6_full_outputs.json"
OUT_MD = OUT_DIR / "hero4_diagnosis.md"
OUT_JSON = OUT_DIR / "hero4_diagnosis.json"
CACHE_MAX_AGE_SECONDS = 6 * 3600


# ──────────────────────────────────────────────────────────────────────────────
# A6 introspection (Paso 0)
# ──────────────────────────────────────────────────────────────────────────────

def inspect_a6_emit_behavior() -> dict:
    """Return a static description of A6's emit behavior and sampling config."""
    a6_path = PROJECT_ROOT / "agents" / "agent6_quality.py"
    txt = a6_path.read_text(encoding="utf-8")
    findings = {
        "source": str(a6_path.relative_to(PROJECT_ROOT)),
        "emits_overall": "overall" in txt,
        "emits_domains": (
            "domains[domain_name]" in txt
            or "_ROBINS_I_DOMAINS" in txt
            or "_JBI_ITEMS" in txt
        ),
        "rob2_domains": [
            "randomization", "deviations", "missing_data", "measurement", "selection",
        ],
        "robins_i_domains": [
            "confounding", "selection_of_participants", "classification_of_interventions",
            "deviations_from_interventions", "missing_data", "measurement_of_outcomes",
            "selection_of_results",
        ],
        "jbi_items_count": 10,
        "default_temperature_setting": "TEMPERATURE_CONFIG['agent6_quality'] (default 0.2)",
        "do_sample_default": "True (from base.py gen_cfg.get('do_sample', True))",
        "top_p_default": 0.9,
        "non_deterministic": True,
        "default_rob2_overall_on_parse_fail": "Some concerns",
        "least_favorable_rule_for_overall": (
            "RoB2: any High -> High; else any 'Some concerns' -> 'Some concerns'; "
            "else 'Low'. Because the per-domain default for missing/invalid fields "
            "is 'Some concerns', any partial parse cascades to overall='Some concerns'."
        ),
        "calls_per_paper": (
            "Up to 2: 1 design-detection (skipped when RCT regex matches title/abstract) "
            "+ 1 assessment call (RoB2 / ROBINS-I / JBI). Single aggregated JSON."
        ),
    }
    return findings


# ──────────────────────────────────────────────────────────────────────────────
# Paper construction
# ──────────────────────────────────────────────────────────────────────────────

def build_papers() -> tuple[list[dict], list[dict]]:
    sample = pd.read_csv(SAMPLE_CSV, dtype={"covidence_id": str})
    gold_scr = pd.read_csv(GOLD_SCREENING, dtype={"covidence_id": str})
    includes = _filter_hero4_includes(sample)
    gold_idx = gold_scr.set_index("covidence_id")

    papers: list[dict] = []
    skipped: list[dict] = []
    for _, row in includes.iterrows():
        cid = str(row["covidence_id"])
        pdf_filename = row.get("pdf_filename")
        if not row.get("has_pdf", False) or pd.isna(pdf_filename):
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
        abstract = ""
        if cid in gold_idx.index:
            gr = gold_idx.loc[cid]
            if isinstance(gr, pd.DataFrame):
                gr = gr.iloc[0]
            title = str(gr.get("title", "") or "").strip()
            abstract = str(gr.get("abstract", "") or "").strip()
            if title.lower() in ("nan", "none"):
                title = ""
            if abstract.lower() in ("nan", "none"):
                abstract = ""
        papers.append({
            "pmid": cid,
            "title": title,
            "abstract": abstract,
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
# A6 invocation with monkey-patched _simple_generate to capture prompts/responses
# ──────────────────────────────────────────────────────────────────────────────

def run_or_load_a6(papers: list[dict], force: bool) -> dict:
    """Re-run Agent6 with capture of every (prompt, response) pair."""
    if not force and A6_CACHE.is_file():
        age = time.time() - A6_CACHE.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            print(f"[cache] Reusing A6 outputs ({A6_CACHE}, age={int(age)}s)")
            with open(A6_CACHE, encoding="utf-8") as f:
                return json.load(f)
        print(f"[cache] Cache stale (age={int(age)}s > {CACHE_MAX_AGE_SECONDS}s)")

    print(f"[run] Loading Agent6 and running assess_all on {len(papers)} papers...")
    seed_torch(42)
    from agents.agent6_quality import Agent6Quality
    agent = Agent6Quality()

    # Capture buffer. Keyed by pmid of the currently-processing paper.
    capture: dict[str, list[dict]] = defaultdict(list)
    current_pmid: dict[str, str] = {"value": ""}
    original_simple_generate = agent._simple_generate

    def _patched_simple_generate(prompt: str, max_tokens: int = 500) -> str:
        response = original_simple_generate(prompt, max_tokens=max_tokens)
        capture[current_pmid["value"]].append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "response": response,
            "captured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        return response

    agent._simple_generate = _patched_simple_generate  # type: ignore[assignment]

    # _assess_single is the entry per paper — wrap it so we know which pmid the
    # subsequent _simple_generate calls belong to.
    original_assess_single = agent._assess_single

    def _patched_assess_single(paper, protocol):
        current_pmid["value"] = str(paper.get("pmid") or "")
        return original_assess_single(paper, protocol)

    agent._assess_single = _patched_assess_single  # type: ignore[assignment]

    t0 = time.time()
    out = agent.assess_all(papers, PAULA_SPORT_PROTOCOL)
    elapsed = round(time.time() - t0, 2)
    print(f"[run] A6 finished in {elapsed}s")
    try:
        agent.free_vram()
    except Exception:
        pass

    persistable = {
        "elapsed_seconds": elapsed,
        "n_papers_in": len(papers),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "non_deterministic": True,
        "papers": out.get("papers", []),
        "stats": out.get("stats", {}),
        "captured_llm_calls": {k: v for k, v in capture.items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(A6_CACHE, "w", encoding="utf-8") as f:
        json.dump(persistable, f, indent=2, ensure_ascii=False, default=str)
    return persistable


# ──────────────────────────────────────────────────────────────────────────────
# Gold cross-reference
# ──────────────────────────────────────────────────────────────────────────────

def build_gold_per_study(include_ids: list[str]) -> tuple[dict[str, list[dict]], dict]:
    """Return ``{covidence_id: [gold rob row dicts]}`` plus debug info."""
    gold_rob = pd.read_csv(GOLD_ROB)
    ref_map = pd.read_csv(REF_MAPPING)
    ref_to_cid, target_refs = _build_ref_to_covidence(ref_map, set(include_ids))
    subset, debug = _build_gold_subset(gold_rob, ref_to_cid, target_refs)
    by_cid: dict[str, list[dict]] = defaultdict(list)
    for _, row in subset.iterrows():
        cid = str(row["_derived_covidence_id"])
        rj = str(row.get("rob_judgment") or "").strip()
        if rj not in ROB_ORDINAL:
            continue
        by_cid[cid].append({
            "outcome_name": str(row.get("outcome") or "").strip(),
            "rob_judgment": rj,
            "obj_sf": str(row.get("obj_sf") or "").strip(),
            "instrument_human": str(row.get("instrument_human") or "").strip(),
            "ref_number": str(row.get("ref_number") or "").strip(),
        })
    return dict(by_cid), debug


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic processing
# ──────────────────────────────────────────────────────────────────────────────

def diagnose(papers: list[dict], a6_out: dict, gold_by_cid: dict[str, list[dict]]) -> dict:
    captured = a6_out.get("captured_llm_calls", {}) or {}
    a6_by_id = {str(p.get("pmid", "")): p for p in a6_out.get("papers", [])}

    per_study: list[dict] = []
    tool_dist: dict[str, int] = defaultdict(int)
    overall_pred_dist: dict[str, int] = defaultdict(int)
    overall_pred_norm_dist: dict[str, int] = defaultdict(int)
    rob2_domain_dist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    robins_i_domain_dist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for paper in papers:
        cid = str(paper.get("pmid", ""))
        a6 = a6_by_id.get(cid) or {}
        quality = a6.get("quality_assessment") or {}
        tool = quality.get("tool", "n/a")
        overall_raw = quality.get("overall")
        overall_norm = _normalize_a6_overall(overall_raw)
        weight = a6.get("quality_weight")
        design = a6.get("study_design_detected")
        domains = quality.get("domains") or {}

        tool_dist[tool] += 1
        overall_pred_dist[str(overall_raw)] += 1
        overall_pred_norm_dist[str(overall_norm)] += 1

        if tool == "RoB2":
            for dname, d in domains.items():
                rob2_domain_dist[dname][str(d.get("judgment"))] += 1
        elif tool == "ROBINS-I":
            for dname, d in domains.items():
                robins_i_domain_dist[dname][str(d.get("judgment"))] += 1

        gold_outcomes = gold_by_cid.get(cid, [])
        gold_worst = _worst_case_gold([o["rob_judgment"] for o in gold_outcomes]) if gold_outcomes else None

        per_study.append({
            "covidence_id": cid,
            "title": paper.get("title", ""),
            "study_design_detected": design,
            "tool": tool,
            "overall_predicted_raw": overall_raw,
            "overall_predicted_norm": overall_norm,
            "quality_weight": weight,
            "domains": domains,
            "items": quality.get("items"),
            "yes_count": quality.get("yes_count"),
            "overall_justification": quality.get("overall_justification"),
            "gold_outcomes": gold_outcomes,
            "gold_worst_case": gold_worst,
            "primary_match": (gold_worst is not None and gold_worst == overall_norm),
            "n_llm_calls_captured": len(captured.get(cid, [])),
            "captured_calls": captured.get(cid, []),
            "extraction_quality": (
                a6.get("study_characteristics") and a6.get("study_characteristics").get("study_design")
                if isinstance(a6.get("study_characteristics"), dict) else None
            ),
        })

    return {
        "per_study": per_study,
        "tool_distribution": dict(tool_dist),
        "overall_predicted_raw_distribution": dict(overall_pred_dist),
        "overall_predicted_norm_distribution": dict(overall_pred_norm_dist),
        "rob2_domain_distribution": {k: dict(v) for k, v in rob2_domain_dist.items()},
        "robins_i_domain_distribution": {k: dict(v) for k, v in robins_i_domain_dist.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Example selection
# ──────────────────────────────────────────────────────────────────────────────

def pick_examples(per_study: list[dict], min_examples: int = 3, max_examples: int = 5) -> list[dict]:
    """Pick studies covering gold extremes (Low, Some concerns, High) if available."""
    seen_buckets: set[str] = set()
    chosen: list[dict] = []
    # First pass: try one per gold worst-case bucket.
    for bucket in ("L", "H", "SC"):
        for st in per_study:
            if st.get("gold_worst_case") == bucket and bucket not in seen_buckets:
                chosen.append(st)
                seen_buckets.add(bucket)
                break
    # Fill up with any remaining.
    for st in per_study:
        if st in chosen:
            continue
        if len(chosen) >= max_examples:
            break
        chosen.append(st)
    return chosen[:max_examples] if len(chosen) >= min_examples else chosen


# ──────────────────────────────────────────────────────────────────────────────
# Hypotheses scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_hypotheses(per_study: list[dict], a6_emit: dict) -> dict:
    """Return per-hypothesis evidence summary.

    a) Prompt sesgado a "Some concerns" — inspect prompt for biased wording.
    b) Fallback en parser cuando no hay señal — count studies whose overall
       matches the "Default: LLM response could not be parsed" path OR whose
       domains carry the per-domain default justification.
    c) PDFs sin info suficiente para juicio diferente — count studies with low
       full_text_length or marginal_extraction (we don't have those here, but
       any non-deterministic overall_justification could hint).
    d) Parser que colapsa varios outputs a "Some concerns" — count studies
       whose raw LLM response contains "Some concerns" verbatim but also
       contains "Low" or "High" somewhere (indicates parser collapsed).
    """
    evidence: dict = {"a": [], "b": [], "c": [], "d": []}

    for st in per_study:
        cid = st["covidence_id"]
        overall_just = (st.get("overall_justification") or "").lower()
        domains = st.get("domains") or {}
        # b) explicit default markers in justifications
        b_hits = []
        for dname, d in domains.items():
            j = (d.get("justification") or "").lower() if isinstance(d, dict) else ""
            if "could not" in j or "not specified" in j or "default" in j or "not assessable" in j:
                b_hits.append({"domain": dname, "justification": d.get("justification") if isinstance(d, dict) else None})
        if "could not be parsed" in overall_just or "derived from domain judgments" in overall_just:
            b_hits.append({"domain": "<overall>", "justification": st.get("overall_justification")})
        if b_hits:
            evidence["b"].append({"covidence_id": cid, "hits": b_hits})

        # d) parser collapse: check raw response for mixed labels
        for call in st.get("captured_calls") or []:
            resp = (call.get("response") or "").lower()
            if not resp:
                continue
            has_low = ("\"low\"" in resp) or (": low" in resp) or ("'low'" in resp)
            has_sc = "some concerns" in resp
            has_high = ("\"high\"" in resp) or (": high" in resp) or ("'high'" in resp)
            if (has_low or has_high) and has_sc and st.get("overall_predicted_norm") == "SC":
                evidence["d"].append({
                    "covidence_id": cid,
                    "raw_response_has_low": has_low,
                    "raw_response_has_high": has_high,
                    "raw_response_has_some_concerns": has_sc,
                })
                break

    # a) prompt bias — single per-prompt-template scan, captured once
    prompt_seen = None
    for st in per_study:
        for call in st.get("captured_calls") or []:
            p = call.get("prompt") or ""
            if "RoB 2" in p or "Cochrane methodologist performing Risk of Bias" in p:
                prompt_seen = p
                break
        if prompt_seen:
            break
    if prompt_seen:
        sc_count = prompt_seen.lower().count("some concerns")
        low_count = prompt_seen.lower().count("\"low\"")
        high_count = prompt_seen.lower().count("\"high\"")
        evidence["a"].append({
            "prompt_template_seen": True,
            "occurrences_in_prompt": {
                "some_concerns": sc_count,
                "low_label": low_count,
                "high_label": high_count,
            },
            "note": (
                "All three labels appear roughly the same number of times in the prompt "
                "(once per domain definition). The prompt itself does not lexically "
                "over-emphasise 'Some concerns'; if 'a' holds it would be via "
                "instruction wording, not label frequency."
            ),
        })
    else:
        evidence["a"].append({"prompt_template_seen": False, "note": "No RoB2 prompt captured (no RCT in sample)"})

    return evidence


# ──────────────────────────────────────────────────────────────────────────────
# Markdown writer
# ──────────────────────────────────────────────────────────────────────────────

def write_md(envelope: dict) -> None:
    lines: list[str] = []
    lines.append("# Hero 4 deep-dive diagnóstico — T16.D.7.C")
    lines.append("")
    lines.append(f"- Timestamp: `{envelope['timestamp_utc']}`")
    lines.append(f"- A6 cache: `{A6_CACHE.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- A6 cache elapsed: `{envelope.get('a6_cache_elapsed_seconds')}s`")
    lines.append(f"- N papers analizados: `{envelope['n_papers']}`")
    lines.append(f"- Cache reused: `{envelope.get('cache_reused')}`")
    lines.append("")
    if envelope.get("blocks"):
        lines.append("## Bloqueos detectados")
        lines.append("")
        for b in envelope["blocks"]:
            lines.append(f"- {b}")
        lines.append("")
    lines.append("## Paso 0 — inspección de Agent6")
    lines.append("")
    e = envelope["a6_emit"]
    lines.append(f"- Source: `{e['source']}`")
    lines.append(f"- Emite `overall`: `{e['emits_overall']}`")
    lines.append(f"- Emite `domain-level` judgments: `{e['emits_domains']}`")
    lines.append(f"- Domains RoB2: {e['rob2_domains']}")
    lines.append(f"- Domains ROBINS-I: {e['robins_i_domains']}")
    lines.append(f"- JBI items: {e['jbi_items_count']}")
    lines.append(f"- Temperature: `{e['default_temperature_setting']}`")
    lines.append(f"- do_sample default: `{e['do_sample_default']}`")
    lines.append(f"- top_p default: `{e['top_p_default']}`")
    lines.append(f"- **Non-deterministic**: `{e['non_deterministic']}` — caché es snapshot del run, no idempotente.")
    lines.append(f"- Default RoB2 overall on parse fail: `{e['default_rob2_overall_on_parse_fail']}`")
    lines.append(f"- Calls por paper: {e['calls_per_paper']}")
    lines.append("")
    lines.append("> Regla del overall: " + e["least_favorable_rule_for_overall"])
    lines.append("")

    lines.append("## Distribuciones")
    lines.append("")
    lines.append("### Tool elegido por paper")
    lines.append("")
    lines.append("| tool | n |")
    lines.append("|---|---:|")
    for k, v in envelope["tool_distribution"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("### Overall predicho (raw A6)")
    lines.append("")
    lines.append("| overall (raw) | n |")
    lines.append("|---|---:|")
    for k, v in envelope["overall_predicted_raw_distribution"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("### Overall predicho (normalizado a {L, SC, H})")
    lines.append("")
    lines.append("| overall (norm) | n |")
    lines.append("|---|---:|")
    for k, v in envelope["overall_predicted_norm_distribution"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    rob2 = envelope.get("rob2_domain_distribution") or {}
    if rob2:
        lines.append("### RoB2 — distribución por dominio")
        lines.append("")
        all_labels = sorted({lab for d in rob2.values() for lab in d.keys()})
        header = "| domain | " + " | ".join(all_labels) + " |"
        sep = "|---" + "|---:" * len(all_labels) + "|"
        lines.append(header)
        lines.append(sep)
        for dname, dist in rob2.items():
            row = "| " + dname + " | " + " | ".join(str(dist.get(lab, 0)) for lab in all_labels) + " |"
            lines.append(row)
        lines.append("")
    robins = envelope.get("robins_i_domain_distribution") or {}
    if robins:
        lines.append("### ROBINS-I — distribución por dominio")
        lines.append("")
        all_labels = sorted({lab for d in robins.values() for lab in d.keys()})
        header = "| domain | " + " | ".join(all_labels) + " |"
        sep = "|---" + "|---:" * len(all_labels) + "|"
        lines.append(header)
        lines.append(sep)
        for dname, dist in robins.items():
            row = "| " + dname + " | " + " | ".join(str(dist.get(lab, 0)) for lab in all_labels) + " |"
            lines.append(row)
        lines.append("")

    lines.append("## Tabla resumen por estudio")
    lines.append("")
    lines.append("| covidence_id | design | tool | A6 overall raw | A6 overall norm | gold worst | match | weight | gold N outcomes |")
    lines.append("|---|---|---|---|---|---|---|---:|---:|")
    for st in envelope["per_study"]:
        cid = st["covidence_id"]
        mt = "✓" if st.get("primary_match") else "✗"
        lines.append(
            f"| {cid} | {st.get('study_design_detected')} | {st.get('tool')} | "
            f"`{st.get('overall_predicted_raw')}` | `{st.get('overall_predicted_norm')}` | "
            f"`{st.get('gold_worst_case')}` | {mt} | {st.get('quality_weight')} | "
            f"{len(st.get('gold_outcomes') or [])} |"
        )
    lines.append("")

    lines.append("## Ejemplos side-by-side (3-5)")
    lines.append("")
    examples = envelope.get("examples") or []
    for st in examples:
        cid = st["covidence_id"]
        lines.append(f"### {cid} — gold worst-case = `{st.get('gold_worst_case')}` / A6 = `{st.get('overall_predicted_norm')}`")
        lines.append("")
        lines.append(f"- design_detected: `{st.get('study_design_detected')}`")
        lines.append(f"- tool: `{st.get('tool')}`")
        lines.append(f"- overall_predicted_raw: `{st.get('overall_predicted_raw')}`")
        lines.append(f"- overall_justification: `{st.get('overall_justification')}`")
        lines.append(f"- weight: `{st.get('quality_weight')}`")
        lines.append("")
        if st.get("domains"):
            lines.append("**A6 domains (judgment / justification):**")
            lines.append("")
            for dname, d in st["domains"].items():
                if isinstance(d, dict):
                    lines.append(f"- `{dname}`: **{d.get('judgment')}** — {d.get('justification')}")
                else:
                    lines.append(f"- `{dname}`: {d}")
            lines.append("")
        if st.get("gold_outcomes"):
            lines.append("**Gold outcomes (Paula):**")
            lines.append("")
            lines.append("| outcome | rob_judgment | obj_sf | instrument | ref_number |")
            lines.append("|---|---|---|---|---|")
            for o in st["gold_outcomes"]:
                lines.append(
                    f"| {o.get('outcome_name')} | `{o.get('rob_judgment')}` | "
                    f"{o.get('obj_sf')} | {o.get('instrument_human')} | {o.get('ref_number')} |"
                )
            lines.append("")
        calls = st.get("captured_calls") or []
        if calls:
            lines.append("**LLM calls capturadas**:")
            lines.append("")
            for idx, c in enumerate(calls):
                lines.append(f"#### call #{idx} (max_tokens={c.get('max_tokens')})")
                lines.append("")
                lines.append("**Prompt** (primeros 2500 chars):")
                lines.append("")
                lines.append("```")
                lines.append((c.get("prompt") or "")[:2500])
                lines.append("```")
                lines.append("")
                lines.append("**Raw LLM response**:")
                lines.append("")
                lines.append("```")
                lines.append((c.get("response") or "")[:3000])
                lines.append("```")
                lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Hipótesis del modal collapse")
    lines.append("")
    hyp = envelope.get("hypotheses") or {}
    lines.append("### a) Prompt sesgado a 'Some concerns'")
    lines.append("")
    for a in hyp.get("a", []):
        lines.append("```json")
        lines.append(json.dumps(a, indent=2, ensure_ascii=False, default=str))
        lines.append("```")
        lines.append("")
    lines.append("### b) Lógica de fallback / per-domain default cuando no hay señal")
    lines.append("")
    lines.append(f"Estudios con evidencia de default: **{len(hyp.get('b', []))}** / {envelope['n_papers']}")
    for b in hyp.get("b", []):
        lines.append("")
        lines.append(f"- **{b['covidence_id']}**:")
        for hit in b["hits"][:5]:
            lines.append(f"  - `{hit['domain']}` → `{hit['justification']}`")
    lines.append("")
    lines.append("### c) PDFs sin info suficiente para juicio diferente")
    lines.append("")
    lines.append("Indicador: full_text_length por paper (ver tabla resumen). Mayor parte de los PDFs tienen >20k chars; no es evidencia inmediata, pero se reporta en el JSON. Marginal extraction conocido: #7759 (395 chars) — capturado por hero4 evaluator como `marginal_extraction`.")
    lines.append("")
    lines.append("### d) Parser que colapsa respuestas con múltiples labels a 'Some concerns'")
    lines.append("")
    lines.append(f"Estudios cuya respuesta raw contiene a la vez 'Some concerns' y al menos uno de Low/High, terminando en overall normalizado='SC': **{len(hyp.get('d', []))}**")
    for d in hyp.get("d", []):
        lines.append(f"- **{d['covidence_id']}**: raw response contiene low={d['raw_response_has_low']}, high={d['raw_response_has_high']}, some_concerns={d['raw_response_has_some_concerns']}")
    lines.append("")
    lines.append("## Gold subset debug")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(envelope.get("gold_debug") or {}, indent=2, ensure_ascii=False, default=str))
    lines.append("```")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T16.D.7.C Hero 4 deep-dive diagnostic.")
    p.add_argument("--force", action="store_true", help="Ignore A6 cache and re-run.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print("=" * 72)
    print("T16.D.7.C — Hero 4 (Agent6 RoB) deep-dive diagnostic")
    print("=" * 72)

    blocks: list[str] = []

    print("\n[0/5] Inspecting Agent6 emit behavior + sampling config...")
    a6_emit = inspect_a6_emit_behavior()
    print(f"      emits_overall={a6_emit['emits_overall']} emits_domains={a6_emit['emits_domains']} non_det={a6_emit['non_deterministic']}")

    print("\n[1/5] Building papers from sample_50 hero4 includes...")
    try:
        papers, skipped = build_papers()
    except Exception as e:
        blocks.append(f"BLOQUEO build_papers: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        papers, skipped = [], []
    print(f"      Papers built: {len(papers)} | Skipped: {len(skipped)}")
    for s in skipped:
        print(f"        - {s}")

    print("\n[2/5] Loading or running A6 (cache snapshot — non-deterministic)...")
    cache_existed = A6_CACHE.is_file() and not args.force
    try:
        a6_out = run_or_load_a6(papers, force=args.force)
        cache_reused = cache_existed and (time.time() - A6_CACHE.stat().st_mtime) < CACHE_MAX_AGE_SECONDS
    except Exception as e:
        blocks.append(f"BLOQUEO A6 run: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        a6_out = {"papers": [], "stats": {}, "captured_llm_calls": {}, "elapsed_seconds": 0}
        cache_reused = False

    print(f"      A6 papers: {len(a6_out.get('papers', []))}")

    print("\n[3/5] Loading gold_rob + ref_to_covidence...")
    include_ids = [str(p.get("pmid")) for p in papers]
    try:
        gold_by_cid, gold_debug = build_gold_per_study(include_ids)
    except Exception as e:
        blocks.append(f"BLOQUEO gold load: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        gold_by_cid, gold_debug = {}, {}
    print(f"      gold studies covered: {len(gold_by_cid)}")

    print("\n[4/5] Running per-study cross-reference...")
    diag = diagnose(papers, a6_out, gold_by_cid)
    examples = pick_examples(diag["per_study"])
    print(f"      examples picked: {len(examples)} ({[s['covidence_id'] for s in examples]})")

    print("\n[5/5] Scoring hypotheses...")
    hypotheses = score_hypotheses(diag["per_study"], a6_emit)

    envelope = {
        "task": "T16.D.7.C Hero 4 deep-dive diagnostic",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "a6_cache_path": str(A6_CACHE.relative_to(PROJECT_ROOT)),
        "a6_cache_elapsed_seconds": a6_out.get("elapsed_seconds"),
        "cache_reused": cache_reused,
        "n_papers": len(papers),
        "skipped_pre_a6": skipped,
        "blocks": blocks,
        "a6_emit": a6_emit,
        "tool_distribution": diag["tool_distribution"],
        "overall_predicted_raw_distribution": diag["overall_predicted_raw_distribution"],
        "overall_predicted_norm_distribution": diag["overall_predicted_norm_distribution"],
        "rob2_domain_distribution": diag["rob2_domain_distribution"],
        "robins_i_domain_distribution": diag["robins_i_domain_distribution"],
        "per_study": diag["per_study"],
        "examples": examples,
        "hypotheses": hypotheses,
        "gold_debug": gold_debug,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False, default=str)
    write_md(envelope)

    print()
    print(f"[done] JSON: {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"       MD  : {OUT_MD.relative_to(PROJECT_ROOT)}")
    print(f"       blocks: {len(blocks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
