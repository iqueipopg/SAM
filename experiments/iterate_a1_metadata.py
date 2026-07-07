# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""T17.A3 prompt-iteration orchestrator for A1._generate_metadata.

Overnight greedy local search over a hand-curated space of prompt variants
for ``Agent1Protocol._generate_metadata`` (the framework + PICO extractor).
Each variant is a string template using the placeholder ``{query}`` and is
injected into A1 via the new ``metadata_prompt_template`` kwarg on
``generate_protocol`` (see T17.A3 Fase I.1 refactor of agent1_protocol.py).

Metric: per-element semantic cosine similarity (P/I/C/O) computed by
``experiments.eval_a1_pico.run_eval`` against the LEADSInstruct ``pico.*``
ground truth. The orchestrator drives that evaluator and aggregates its
``aggregates.macro_mean`` as the iteration score (not Recall@1000).

The orchestrator:
  1. Loads (or seeds) variants V0..V5 from ``experiments/T17.A3/prompts/``.
  2. Evaluates each on DEV (default ``data/groundtruth/leads_full20.json``)
     by calling ``experiments.eval_a1_pico.run_eval`` with a single shared
     ``a1_instance`` and a single shared embedder (one model load total).
  3. After the initial sweep, proposes new variants by applying marginal
     "operators" (string transforms) to the current best non-broken variant
     until budget exhausted, target sustained, or no new variant can be
     generated.
  4. Persists ``iteration_log.json``, ``best_variant.json`` and the running
     ``iteration.log`` plus every fresh template to ``prompts/``.

Design rules followed:
  - Spanish for user-facing messages; English for code.
  - Single A1 instance + single embedder kept alive across all variants.
  - Per-variant Entrez-free A1 output cache (different prompts -> different
    metadata; cache key = sample_id::template_hash inside eval_a1_pico).
  - Sanity gate per variant: if ``n_with_similarity == 0`` or ``macro is None``
    or any exception leaked, mark ``broken=True`` and skip.

Usage:
    venv/bin/python experiments/iterate_a1_metadata.py \\
        --dev-file data/groundtruth/leads_full20.json \\
        --output-dir experiments/T17.A3/ \\
        --target-macro 0.55 \\
        --ambitious-macro 0.60 \\
        --max-hours 4 \\
        --max-no-improvement 20
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Initial variant seeds (fallback if prompts/V{N}.txt is missing on disk)
# ──────────────────────────────────────────────────────────────────────────────

# V0 must equal the literal current hardcoded prompt in
# agent1_protocol.py._generate_metadata (after T17.A3 refactor) so it
# reproduces the post-T17.A2 baseline. The file bundled in
# experiments/T17.A3/prompts/V0.txt is the canonical source; the dict
# entry here only exists as a self-healing fallback for accidental deletion.
INITIAL_VARIANTS: dict[str, str] = {
    "V0": (
        "Cochrane expert. Identify framework. ALL output must be in ENGLISH.\n\n"
        "KEYWORDS GUIDE:\n"
        '- PICO: "effective", "reduce", "improve", "versus", "compared to", "treatment"\n'
        '- SPIDER: "experience", "barriers", "perceptions", "feelings", "challenges", "how do", "what do"\n'
        '- PECO: "risk", "cause", "exposure", "factor", "associated with"\n'
        '- PCC: "prevalence", "frequency", "how many", "proportion"\n\n'
        'Query: "{query}"\n\n'
        "Step 1 - Detect keywords:\n"
        "Keywords found: [list keywords from query]\n\n"
        "Step 2 - Choose framework:\n"
        "If comparing intervention/treatment -> PICO\n"
        "If exploring experiences/barriers/perceptions -> SPIDER\n"
        "If studying risk/exposure causes -> PECO\n"
        "If describing prevalence -> PCC\n\n"
        "Framework: [your choice]\n\n"
        "Step 3 - Extract components (ALL values in English):\n"
        "JSON (no arrays):\n"
        "{{\n"
        '  "framework": "PICO/SPIDER/PECO/PCC",\n'
        '  "research_question": "English research question",\n'
        '  "population": "English description",\n'
        '  "intervention": "English full name (no abbreviations)",\n'
        '  "comparator": "English description",\n'
        '  "outcome": "English description"\n'
        "}}"
    ),
    # V1-V5 fallback seeds intentionally empty — canonical source is V{N}.txt.
    "V1": "",
    "V2": "",
    "V3": "",
    "V4": "",
    "V5": "",
}

VARIANT_NAMES_INITIAL = ["V0", "V1", "V2", "V3", "V4", "V5"]


# ──────────────────────────────────────────────────────────────────────────────
# Operators (string transforms) for the greedy proposer
# ──────────────────────────────────────────────────────────────────────────────

ATOMICITY_BLOCK = (
    "\nATOMICITY REQUIREMENT:\n"
    "Each component (population, intervention, comparator, outcome) must be a "
    "SINGLE specific concept — not a list, not a description with multiple "
    "items combined by 'and' / 'or' / commas. Examples:\n"
    "  GOOD intervention: \"Cognitive Behavioral Therapy\"\n"
    "  BAD intervention:  \"CBT, mindfulness and pharmacotherapy\"\n"
    "  GOOD outcome: \"Depression remission\"\n"
    "  BAD outcome:  \"Depression, anxiety, and quality of life\"\n"
    "If the question implies multiple interventions or outcomes, pick the most "
    "specific / primary one and put the rest aside.\n"
)

FEW_SHOTS_BLOCK = (
    "\nEXAMPLES (reference patterns across domains):\n\n"
    "EXAMPLE 1 (psychotherapy / PICO):\n"
    "Query: \"Is cognitive behavioral therapy more effective than sertraline "
    "for major depression in adults?\"\n"
    "{{\n"
    '  "framework": "PICO",\n'
    '  "research_question": "Is CBT more effective than sertraline for major depression in adults?",\n'
    '  "population": "Adults with major depressive disorder",\n'
    '  "intervention": "Cognitive Behavioral Therapy",\n'
    '  "comparator": "Sertraline",\n'
    '  "outcome": "Depression remission"\n'
    "}}\n\n"
    "EXAMPLE 2 (surgery / PICO):\n"
    "Query: \"Does laparoscopic cholecystectomy reduce postoperative pain "
    "compared with open cholecystectomy?\"\n"
    "{{\n"
    '  "framework": "PICO",\n'
    '  "research_question": "Does laparoscopic cholecystectomy reduce postoperative pain vs open cholecystectomy?",\n'
    '  "population": "Adults undergoing cholecystectomy",\n'
    '  "intervention": "Laparoscopic cholecystectomy",\n'
    '  "comparator": "Open cholecystectomy",\n'
    '  "outcome": "Postoperative pain"\n'
    "}}\n\n"
    "EXAMPLE 3 (screening / PICO):\n"
    "Query: \"Does mammography screening reduce breast cancer mortality in "
    "women aged 50-69?\"\n"
    "{{\n"
    '  "framework": "PICO",\n'
    '  "research_question": "Does mammography screening reduce breast cancer mortality in women 50-69?",\n'
    '  "population": "Women aged 50-69",\n'
    '  "intervention": "Mammography screening",\n'
    '  "comparator": "No screening",\n'
    '  "outcome": "Breast cancer mortality"\n'
    "}}\n\n"
    "EXAMPLE 4 (prevention / PICO):\n"
    "Query: \"Does HPV vaccination prevent cervical cancer in adolescent "
    "girls?\"\n"
    "{{\n"
    '  "framework": "PICO",\n'
    '  "research_question": "Does HPV vaccination prevent cervical cancer in adolescent girls?",\n'
    '  "population": "Adolescent girls",\n'
    '  "intervention": "HPV vaccination",\n'
    '  "comparator": "No vaccination",\n'
    '  "outcome": "Cervical cancer incidence"\n'
    "}}\n"
)

COT_BLOCK = (
    "\nREASONING STEPS (think internally before producing JSON):\n"
    "CoT-Step 1: Identify the framework using the KEYWORDS GUIDE above.\n"
    "CoT-Step 2: Extract atomic components — one concept per field, no lists.\n"
    "CoT-Step 3: Validate completeness — every required field present and "
    "non-empty; framework matches the components shape.\n"
)

VALIDATION_BLOCK = (
    "\nVALIDATION STEP (mandatory before final output):\n"
    "After producing the JSON, re-read each field and verify:\n"
    "  (a) Is it a SINGLE concept (not a list, not 'X and Y')?\n"
    "  (b) Is it specific and not generic ('treatment', 'patients', "
    "'outcome')?\n"
    "  (c) Does it match the framework chosen?\n"
    "If any field fails one of (a)/(b)/(c), refine it before final output.\n"
)


def _insert_before_final_brace(template: str, block: str) -> str:
    """Insert ``block`` right before the final closing ``}}`` of the JSON schema.

    The V0 template ends with a JSON schema; new instructional blocks should
    appear AFTER the schema (so they read as additional guidance) but BEFORE
    any trailing whitespace. We find the last ``}}`` and append ``block``
    immediately after it.
    """
    marker = "}}"
    idx = template.rfind(marker)
    if idx < 0:
        return template + block
    insert_at = idx + len(marker)
    return template[:insert_at] + block + template[insert_at:]


def _op_add_atomicity(template: str) -> str | None:
    """Inject the atomicity requirement block. Idempotent."""
    if "SINGLE specific concept" in template:
        return None
    return _insert_before_final_brace(template, ATOMICITY_BLOCK)


def _op_add_few_shots(template: str) -> str | None:
    """Inject 4 multi-domain few-shot examples. Idempotent.

    The idempotency check uses ``>= 4`` occurrences of ``EXAMPLE`` headers as
    the proxy: once a template already contains 4+ such markers, we treat
    the operator as a no-op.
    """
    if template.count("EXAMPLE ") >= 4:
        return None
    return _insert_before_final_brace(template, FEW_SHOTS_BLOCK)


def _op_add_cot(template: str) -> str | None:
    """Inject a 3-step chain-of-thought instruction block. Idempotent.

    Idempotency uses the unique marker ``CoT-Step 1`` rather than ``Step 1``
    so the operator does NOT collide with the baseline V0 prompt, which
    already contains ``Step 1 - Detect keywords:`` as part of its scaffolding.
    """
    if "CoT-Step 1" in template:
        return None
    return _insert_before_final_brace(template, COT_BLOCK)


def _op_add_validation_step(template: str) -> str | None:
    """Inject a post-output validation step block. Idempotent."""
    if "re-read each field" in template:
        return None
    return _insert_before_final_brace(template, VALIDATION_BLOCK)


# Registry of single-step operators. The orchestrator applies them one at a
# time on top of the current best non-broken variant.
OPERATORS: dict[str, Callable[[str], str | None]] = {
    "add_atomicity": _op_add_atomicity,
    "add_few_shots": _op_add_few_shots,
    "add_cot": _op_add_cot,
    "add_validation_step": _op_add_validation_step,
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str, log_file: Path | None = None) -> None:
    """Print structured stdout line and (optionally) append to log_file."""
    line = f"[T17.A3] {msg}"
    print(line, flush=True)
    if log_file is not None:
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            traceback.print_exc()


def _build_initial_variants_from_v0(prompts_dir: Path, v0_template: str) -> dict[str, str]:
    """Construct V1-V5 by applying operators in the canonical sequence on V0.

    V1 = V0 + add_atomicity
    V2 = V0 + add_few_shots
    V3 = V0 + add_atomicity + add_few_shots
    V4 = V0 + add_cot
    V5 = V3 + add_cot   (i.e. atomicity + few_shots + CoT)
    """
    v1 = _op_add_atomicity(v0_template) or v0_template
    v2 = _op_add_few_shots(v0_template) or v0_template
    v3_step = _op_add_atomicity(v0_template) or v0_template
    v3 = _op_add_few_shots(v3_step) or v3_step
    v4 = _op_add_cot(v0_template) or v0_template
    v5 = _op_add_cot(v3) or v3
    return {
        "V0": v0_template,
        "V1": v1,
        "V2": v2,
        "V3": v3,
        "V4": v4,
        "V5": v5,
    }


def load_initial_variants(prompts_dir: Path) -> dict[str, str]:
    """Read V0..V5 from disk; if missing, seed deterministically from V0.

    The V{N}.txt files are the canonical source. If any of V1..V5 is missing,
    we regenerate it by applying the canonical operator sequence to V0.
    If V0 itself is missing we fall back to the in-module seed.
    """
    prompts_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    # 1) Load V0 (canonical source: prompts/V0.txt; fallback to seed dict).
    v0_path = prompts_dir / "V0.txt"
    if v0_path.exists():
        v0 = v0_path.read_text(encoding="utf-8")
    else:
        v0 = INITIAL_VARIANTS["V0"]
        if not v0:
            raise FileNotFoundError(
                f"V0 missing on disk ({v0_path}) and no seed available."
            )
        v0_path.write_text(v0, encoding="utf-8")

    # 2) Load or regenerate V1..V5.
    regenerated = _build_initial_variants_from_v0(prompts_dir, v0)
    out["V0"] = v0
    for name in ("V1", "V2", "V3", "V4", "V5"):
        path = prompts_dir / f"{name}.txt"
        if path.exists():
            out[name] = path.read_text(encoding="utf-8")
        else:
            out[name] = regenerated[name]
            path.write_text(out[name], encoding="utf-8")
    return out


def _write_template(prompts_dir: Path, name: str, template: str) -> Path:
    """Persist a freshly proposed template to ``prompts/{name}.txt``."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize file name (the '+' operator is the natural separator).
    safe_name = name.replace("/", "_")
    path = prompts_dir / f"{safe_name}.txt"
    path.write_text(template, encoding="utf-8")
    return path


def run_variant(
    name: str,
    template: str,
    dev_file: Path,
    a1_instance: Any,
    embedder: Any,
    output_dir: Path,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one variant against DEV. Returns the metrics dict."""
    # Local import so the orchestrator can be imported in tests without the
    # heavy sentence-transformers stack.
    from experiments.eval_a1_pico import run_eval  # noqa: WPS433

    variant_dir = output_dir / "cache" / name.replace("/", "_")
    variant_dir.mkdir(parents=True, exist_ok=True)
    a1_output_cache = variant_dir / "a1_outputs.json"
    output_path = variant_dir / "eval.json"

    template_path = output_dir / "prompts" / f"{name.replace('/', '_')}.txt"

    result: dict[str, Any] = {
        "name": name,
        "macro": None,
        "median": None,
        "pct_3_of_4": None,
        "per_element": None,
        "runtime_s": None,
        "n_with_similarity": None,
        "predictions_sample": [],
        "template_path": str(template_path),
        "broken": False,
        "error": None,
        "timestamp": _utc_now_iso(),
    }

    if output_path.exists():
        # Previous attempt cached — load and return without re-running.
        try:
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            result.update(_extract_metrics(name, cached, template_path))
            _log(
                f"variant {name}: cached result macro={result['macro']} "
                f"pct_3_of_4={result['pct_3_of_4']}",
                log_file,
            )
            return result
        except Exception:
            output_path.unlink(missing_ok=True)

    t0 = time.time()
    try:
        eval_result = run_eval(
            sample_file=dev_file,
            output=output_path,
            a1_instance=a1_instance,
            metadata_prompt_template=template,
            embedder=embedder,
            a1_output_cache=a1_output_cache,
        )
    except Exception as exc:  # noqa: BLE001
        result["broken"] = True
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["runtime_s"] = round(time.time() - t0, 2)
        _log(f"variant {name}: BROKEN — {result['error']}", log_file)
        return result

    result["runtime_s"] = round(time.time() - t0, 2)
    result.update(_extract_metrics(name, eval_result, template_path))
    _log(
        f"variant {name}: macro={result['macro']} pct_3_of_4={result['pct_3_of_4']} "
        f"n={result['n_with_similarity']} runtime={result['runtime_s']}s",
        log_file,
    )
    return result


def _extract_metrics(
    name: str,
    eval_result: dict[str, Any],
    template_path: Path,
) -> dict[str, Any]:
    """Pull the metrics we care about out of an eval_a1_pico result dict."""
    per_sample = eval_result.get("per_sample", []) or []
    aggregates = eval_result.get("aggregates", {}) or {}
    macro = aggregates.get("macro_mean")
    median = aggregates.get("median")
    pct_3_of_4 = aggregates.get("pct_samples_3_of_4_above_0.5")
    per_element = aggregates.get("per_element")
    n_with_sim = aggregates.get("n_with_similarity") or 0

    # Snapshot the first 2 non-empty predictions for human inspection.
    preds: list[dict[str, Any]] = []
    for r in per_sample:
        pico = r.get("predicted_pico_normalized")
        if pico and len(preds) < 2:
            preds.append({
                "id": r.get("id"),
                "framework": r.get("framework_chosen"),
                "predicted": pico,
                "gold": r.get("gold_pico"),
                "similarity": r.get("pico_similarity"),
            })

    broken = (n_with_sim == 0) or (macro is None)

    return {
        "name": name,
        "macro": macro,
        "median": median,
        "pct_3_of_4": pct_3_of_4,
        "per_element": per_element,
        "n_with_similarity": n_with_sim,
        "predictions_sample": preds,
        "template_path": str(template_path),
        "broken": broken,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Greedy proposer
# ──────────────────────────────────────────────────────────────────────────────

def propose_next_variant(
    history: list[dict[str, Any]],
    prompts_dir: Path,
) -> tuple[str, str] | None:
    """Greedy proposer: apply a marginal operator on top of the best variant.

    Returns ``(name, template)`` or ``None`` when nothing new can be proposed.

    Strategy:
      - Identify the best non-broken variant by macro similarity.
      - Iterate over OPERATORS in registry order; for each operator name,
        check whether ``f"{best}+{op}"`` already exists in history; if not,
        apply the operator. If the operator returns ``None`` or its output
        is identical to the input (idempotent), treat it as a no-op.
      - When all single-op derivations from the best are exhausted, try
        pairwise combinations of the top two operators seen so far in history
        (by mean macro delta over their parent).
      - Returns ``None`` when truly nothing new can be made.
    """
    non_broken = [h for h in history if not h.get("broken")]
    if not non_broken:
        return None

    def _macro_key(h: dict[str, Any]) -> float:
        m = h.get("macro")
        return float(m) if isinstance(m, (int, float)) else float("-inf")

    sorted_hist = sorted(non_broken, key=_macro_key, reverse=True)
    best = sorted_hist[0]
    best_template = _read_template(best["template_path"], prompts_dir)
    seen_names = {h["name"] for h in history}

    # Pass 1: single-op derivations from best.
    for op_name, op_fn in OPERATORS.items():
        candidate_name = f"{best['name']}+{op_name}"
        if candidate_name in seen_names:
            continue
        try:
            new_template = op_fn(best_template)
        except Exception:
            traceback.print_exc()
            continue
        if not new_template or new_template == best_template:
            # operator is a no-op on this template (already applied)
            seen_names.add(candidate_name)
            continue
        return candidate_name, new_template

    # Pass 2: pairwise combos of the top two operators-from-history by macro.
    op_deltas = _collect_operator_deltas(history)
    sorted_ops = sorted(op_deltas.items(), key=lambda kv: kv[1], reverse=True)
    top_ops = [op for op, _delta in sorted_ops[:2] if op in OPERATORS]
    if len(top_ops) >= 2:
        a, b = top_ops[0], top_ops[1]
        for first, second in ((a, b), (b, a)):
            combo_name = f"{best['name']}+{first}+{second}"
            if combo_name in seen_names:
                continue
            try:
                intermediate = OPERATORS[first](best_template)
                if not intermediate:
                    seen_names.add(combo_name)
                    continue
                new_template = OPERATORS[second](intermediate)
            except Exception:
                traceback.print_exc()
                continue
            if not new_template or new_template == best_template:
                seen_names.add(combo_name)
                continue
            return combo_name, new_template

    return None


def _collect_operator_deltas(history: list[dict[str, Any]]) -> dict[str, float]:
    """For each operator suffix observed in history, mean macro delta vs parent."""
    deltas: dict[str, list[float]] = {}
    by_name = {h["name"]: h for h in history}
    for h in history:
        if h.get("broken"):
            continue
        name = h["name"]
        if "+" not in name:
            continue
        parts = name.split("+")
        parent_name = "+".join(parts[:-1])
        op = parts[-1]
        parent = by_name.get(parent_name)
        if not parent or parent.get("broken"):
            continue
        try:
            delta = float(h["macro"]) - float(parent["macro"])
        except (TypeError, ValueError):
            continue
        deltas.setdefault(op, []).append(delta)
    return {op: sum(vs) / len(vs) for op, vs in deltas.items() if vs}


def _read_template(template_path_str: str, prompts_dir: Path) -> str:
    """Load a previously-persisted template from disk."""
    path = Path(template_path_str)
    if not path.is_absolute():
        path = prompts_dir / path.name
    return path.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def _build_a1_instance() -> Any:
    """Build a single Agent1Protocol kept alive across the whole run."""
    from agents.agent1_protocol import Agent1Protocol  # noqa: WPS433

    return Agent1Protocol()


def _build_embedder() -> Any:
    """Build a single sentence-transformers embedder kept alive across all variants."""
    from experiments.eval_a1_pico import _get_embedder  # noqa: WPS433

    return _get_embedder()


def _persist_history(history: list[dict[str, Any]], path: Path, summary: dict[str, Any]) -> None:
    """Atomic write of the iteration_log.json."""
    payload = {
        "summary": summary,
        "history": history,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _persist_best(best: dict[str, Any], path: Path, prompts_dir: Path) -> None:
    """Write ``best_variant.json`` with the full prompt body inline."""
    template = ""
    try:
        template = _read_template(best["template_path"], prompts_dir)
    except Exception:
        traceback.print_exc()
    payload = {
        "name": best["name"],
        "macro": best.get("macro"),
        "median": best.get("median"),
        "pct_3_of_4": best.get("pct_3_of_4"),
        "per_element": best.get("per_element"),
        "runtime_s": best.get("runtime_s"),
        "n_with_similarity": best.get("n_with_similarity"),
        "predictions_sample": best.get("predictions_sample", []),
        "template_path": best.get("template_path"),
        "template": template,
        "timestamp": _utc_now_iso(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def iterate(
    dev_file: Path,
    output_dir: Path,
    target_macro: float,
    ambitious_macro: float,
    max_hours: float,
    max_no_improvement: int,
    start_from: str | None = None,
    a1_instance: Any = None,
    embedder: Any = None,
) -> dict[str, Any]:
    """Run the full iteration loop. Returns the final summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts"
    log_file = output_dir / "iteration.log"
    log_path = output_dir / "iteration_log.json"
    best_path = output_dir / "best_variant.json"

    _log(
        f"starting iteration dev_file={dev_file} target={target_macro} "
        f"ambitious={ambitious_macro} max_hours={max_hours} "
        f"max_no_improvement={max_no_improvement}",
        log_file,
    )

    variants_initial = load_initial_variants(prompts_dir)
    if start_from is not None and start_from not in variants_initial:
        raise ValueError(f"--start-from='{start_from}' not in initial variants")

    own_a1 = False
    if a1_instance is None:
        try:
            a1_instance = _build_a1_instance()
            own_a1 = True
        except Exception as exc:  # noqa: BLE001
            _log(f"FATAL: could not build A1 instance: {exc}", log_file)
            raise

    if embedder is None:
        try:
            embedder = _build_embedder()
        except Exception as exc:  # noqa: BLE001
            _log(f"FATAL: could not build embedder: {exc}", log_file)
            raise

    history: list[dict[str, Any]] = []
    deadline = time.time() + max_hours * 3600
    no_improvement = 0
    best: dict[str, Any] | None = None
    termination_reason = "completed"

    try:
        # Phase 1: initial sweep V0..V5 (or from --start-from onward).
        initial_names = VARIANT_NAMES_INITIAL
        if start_from is not None:
            idx = initial_names.index(start_from)
            initial_names = initial_names[idx:]

        for name in initial_names:
            if time.time() >= deadline:
                termination_reason = "time_budget_exceeded"
                break
            template = variants_initial[name]
            record = run_variant(
                name=name,
                template=template,
                dev_file=dev_file,
                a1_instance=a1_instance,
                embedder=embedder,
                output_dir=output_dir,
                log_file=log_file,
            )
            history.append(record)
            best, no_improvement = _update_best(best, record, no_improvement)
            _persist_history(history, log_path, _summary(history, best, termination_reason, deadline))
            if best:
                _persist_best(best, best_path, prompts_dir)
            if _target_sustained(history, ambitious_macro):
                termination_reason = "ambitious_target_sustained"
                _log("ambitious target sustained — stopping early", log_file)
                break

        # Phase 2: greedy proposer loop.
        while termination_reason == "completed":
            if time.time() >= deadline:
                termination_reason = "time_budget_exceeded"
                break
            if no_improvement >= max_no_improvement:
                termination_reason = "no_improvement_cap_reached"
                break
            proposal = propose_next_variant(history, prompts_dir)
            if proposal is None:
                termination_reason = "no_new_variant_proposable"
                break
            name, template = proposal
            _write_template(prompts_dir, name, template)
            record = run_variant(
                name=name,
                template=template,
                dev_file=dev_file,
                a1_instance=a1_instance,
                embedder=embedder,
                output_dir=output_dir,
                log_file=log_file,
            )
            history.append(record)
            best, no_improvement = _update_best(best, record, no_improvement)
            _persist_history(history, log_path, _summary(history, best, termination_reason, deadline))
            if best:
                _persist_best(best, best_path, prompts_dir)
            if _target_sustained(history, ambitious_macro):
                termination_reason = "ambitious_target_sustained"
                _log("ambitious target sustained — stopping", log_file)
                break

    finally:
        if own_a1:
            free_vram = getattr(a1_instance, "free_vram", None)
            if callable(free_vram):
                try:
                    free_vram()
                except Exception:
                    traceback.print_exc()

    summary = _summary(history, best, termination_reason, deadline)
    _persist_history(history, log_path, summary)
    if best:
        _persist_best(best, best_path, prompts_dir)
    _log(
        f"iteration finished. termination={termination_reason} "
        f"best={best['name'] if best else 'None'} "
        f"macro={best.get('macro') if best else 'None'}",
        log_file,
    )
    return summary


def _update_best(
    best: dict[str, Any] | None,
    record: dict[str, Any],
    no_improvement: int,
) -> tuple[dict[str, Any] | None, int]:
    """Update best tracker and no_improvement counter."""
    if record.get("broken"):
        return best, no_improvement
    macro = record.get("macro")
    if not isinstance(macro, (int, float)):
        return best, no_improvement
    if best is None or macro > (best.get("macro") or float("-inf")):
        return copy.deepcopy(record), 0
    return best, no_improvement + 1


def _target_sustained(history: list[dict[str, Any]], threshold: float) -> bool:
    """True iff the last two non-broken variants both meet ``threshold``."""
    non_broken = [h for h in history if not h.get("broken")]
    if len(non_broken) < 2:
        return False
    last_two = non_broken[-2:]
    return all(
        isinstance(h.get("macro"), (int, float)) and h["macro"] >= threshold
        for h in last_two
    )


def _summary(
    history: list[dict[str, Any]],
    best: dict[str, Any] | None,
    termination_reason: str,
    deadline: float,
) -> dict[str, Any]:
    """Build the summary block written to iteration_log.json."""
    total_runtime = sum(
        h.get("runtime_s") or 0.0 for h in history if isinstance(h.get("runtime_s"), (int, float))
    )
    return {
        "termination_reason": termination_reason,
        "n_variants_evaluated": len(history),
        "n_broken": sum(1 for h in history if h.get("broken")),
        "total_runtime_s": round(total_runtime, 2),
        "best": {
            "name": best["name"] if best else None,
            "macro": best.get("macro") if best else None,
            "pct_3_of_4": best.get("pct_3_of_4") if best else None,
        },
        "seconds_remaining": max(0.0, round(deadline - time.time(), 2)),
        "timestamp": _utc_now_iso(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "T17.A3 prompt-iteration orchestrator. Greedy local search over "
            "A1._generate_metadata prompt variants against a DEV sample, "
            "scored by per-element P/I/C/O cosine similarity."
        )
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "groundtruth" / "leads_full20.json",
        help="DEV sample JSON (default data/groundtruth/leads_full20.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "T17.A3",
        help="Directory for prompts/, cache/, iteration_log.json (default experiments/T17.A3/).",
    )
    parser.add_argument(
        "--target-macro",
        type=float,
        default=0.55,
        help="Floor target macro similarity (default 0.55).",
    )
    parser.add_argument(
        "--ambitious-macro",
        type=float,
        default=0.60,
        help="Ambitious target — triggers early stop when sustained (default 0.60).",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=4.0,
        help="Hard time budget in hours (default 4.0).",
    )
    parser.add_argument(
        "--max-no-improvement",
        type=int,
        default=20,
        help="Consecutive non-improving variants before giving up (default 20).",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Skip earlier initial variants and start from this one (e.g. V3).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        iterate(
            dev_file=args.dev_file.resolve(),
            output_dir=args.output_dir.resolve(),
            target_macro=float(args.target_macro),
            ambitious_macro=float(args.ambitious_macro),
            max_hours=float(args.max_hours),
            max_no_improvement=int(args.max_no_improvement),
            start_from=args.start_from,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
