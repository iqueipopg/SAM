"""T17.A2 prompt-iteration orchestrator for A1.generate_pubmed_query.

Overnight greedy local search over a hand-curated space of prompt variants
for ``Agent1Protocol.generate_pubmed_query``. Each variant is a string
template using the placeholders ``{user_query}``, ``{intv}``, ``{comp}``,
``{outcome}``, ``{design_filter}`` and is injected into A1 via the new
``prompt_template`` kwarg (see commit feat(T17.A2): inject prompt_template
kwarg into A1.generate_pubmed_query).

The orchestrator:
  1. Loads (or seeds) variants V0..V5 from ``experiments/T17.A2/prompts/``.
  2. Evaluates each on DEV (default ``data/groundtruth/leads_full20.json``)
     by calling ``experiments.eval_hero_a1.run_eval`` with a single shared
     ``a1_instance`` (one model load total).
  3. After the initial sweep, proposes new variants by applying marginal
     "operators" (string transforms) to the current best non-broken variant
     until budget exhausted, target sustained, or no new variant can be
     generated.
  4. Persists ``iteration_log.json``, ``best_variant.json`` and the running
     ``iteration.log`` plus every fresh template to ``prompts/``.

Design rules followed:
  - Spanish for user-facing messages; English for code.
  - Single A1 instance kept alive across all variants (project rule #1
    only applies on full pipeline exit; here we own and finally free the
    instance ourselves).
  - Per-variant Entrez cache directory and ``a1_outputs.json``: different
    prompts yield different queries, so cache sharing would silently mix
    runs.
  - Sanity gate per variant: if ``n_nonempty == 0`` or ``macro is None``
    or any exception leaked, mark ``broken=True`` and skip.

Usage:
    venv/bin/python experiments/iterate_a1_prompt.py \\
        --dev-file data/groundtruth/leads_full20.json \\
        --output-dir experiments/T17.A2/ \\
        --target-macro 0.12 \\
        --ambitious-macro 0.15 \\
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

INITIAL_VARIANTS: dict[str, str] = {
    # V0 must equal the literal current hardcoded prompt in agent1_protocol.py
    # (lines 120-141) so it reproduces the post-T17.A1.1 baseline. The file
    # bundled in experiments/T17.A2/prompts/V0.txt is the canonical source; the
    # dict entry here only exists as a self-healing fallback for accidental
    # deletion.
    "V0": (
        "You are a PubMed search expert. Build a structured Boolean search query.\n\n"
        'Research question: "{user_query}"\n'
        "Intervention: {intv}\n"
        "Comparator: {comp}\n"
        "Outcome/Condition: {outcome}\n"
        "Design: {design_filter}\n\n"
        "Rules:\n"
        "- Use MAXIMUM 3 AND clauses (intervention AND condition AND comparator)\n"
        "- Do NOT add population (adults/children) as a separate AND clause — it over-restricts\n"
        "- Use Boolean: AND between concepts, OR between synonyms\n"
        '- Include VALID MeSH terms with [MeSH] tag (e.g. "Cognitive Behavioral Therapy"[MeSH], "Depression"[MeSH])\n'
        "- Include 3-5 free-text synonyms per concept with [Title/Abstract] tag for recall\n"
        "- Group synonyms with parentheses and OR\n"
        "- Output ONLY the search string, one line, no explanation\n"
        "- All terms in English\n\n"
        "Examples:\n"
        'Question: "Does smoking increase MI risk in diabetics?"\n'
        'Output: ("Smoking"[MeSH] OR "Tobacco Use"[MeSH] OR smoking[Title/Abstract] OR cigarette[Title/Abstract] OR tobacco[Title/Abstract]) AND ("Myocardial Infarction"[MeSH] OR "myocardial infarction"[Title/Abstract] OR "heart attack"[Title/Abstract] OR "acute coronary syndrome"[Title/Abstract]) AND ("Diabetes Mellitus"[MeSH] OR diabetes[Title/Abstract] OR diabetic[Title/Abstract])\n\n'
        'Question: "Is CBT more effective than medication for depression in adults?"\n'
        'Output: ("Cognitive Behavioral Therapy"[MeSH] OR "cognitive behavioral therapy"[Title/Abstract] OR CBT[Title/Abstract] OR "cognitive therapy"[Title/Abstract] OR "behavioral therapy"[Title/Abstract]) AND ("Depression"[MeSH] OR "Depressive Disorder"[MeSH] OR depression[Title/Abstract] OR depressive[Title/Abstract] OR MDD[Title/Abstract]) AND ("Antidepressive Agents"[MeSH] OR "Drug Therapy"[MeSH] OR antidepressant[Title/Abstract] OR pharmacotherapy[Title/Abstract] OR medication[Title/Abstract] OR SSRI[Title/Abstract])\n\n'
        "Output:"
    ),
    # V1-V5 fallback seeds are intentionally minimal — the canonical source is
    # the V{N}.txt files. Whenever they exist, those win.
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

ATOMICITY_RULE = (
    "- ATOMICITY: every [MeSH] tag must hold a single canonical NCBI MeSH "
    "descriptor (e.g. \"Depression\"[MeSH], \"Dental Implants\"[MeSH]). NEVER "
    "use multi-word descriptive phrases as MeSH terms (forbidden: \"Different "
    "types of dental implants\"[MeSH], \"Effectiveness of CBT\"[MeSH]). If you "
    "cannot find an atomic MeSH heading, drop the [MeSH] tag and keep the term "
    "as [Title/Abstract] only.\n"
)

FEW_SHOT_PSYCHOLOGY = (
    'Question: "Does mindfulness-based stress reduction reduce anxiety in '
    'college students?"\n'
    'Output: ("Mindfulness"[MeSH] OR mindfulness[Title/Abstract] OR '
    'MBSR[Title/Abstract] OR "mindfulness-based stress reduction"[Title/Abstract]'
    ' OR meditation[Title/Abstract]) AND ("Anxiety"[MeSH] OR "Anxiety '
    'Disorders"[MeSH] OR anxiety[Title/Abstract] OR anxious[Title/Abstract] OR '
    '"anxiety symptoms"[Title/Abstract]) AND ("Students"[MeSH] OR '
    'student[Title/Abstract] OR undergraduate[Title/Abstract] OR '
    'university[Title/Abstract] OR college[Title/Abstract])\n\n'
)

FEW_SHOT_SURGERY = (
    'Question: "Does laparoscopic versus open cholecystectomy reduce '
    'postoperative pain?"\n'
    'Output: ("Cholecystectomy, Laparoscopic"[MeSH] OR '
    'laparoscopic[Title/Abstract] OR "minimally invasive"[Title/Abstract] OR '
    'keyhole[Title/Abstract]) AND ("Cholecystectomy"[MeSH] OR '
    'cholecystectomy[Title/Abstract] OR "gallbladder removal"[Title/Abstract] '
    'OR "open surgery"[Title/Abstract]) AND ("Pain, Postoperative"[MeSH] OR '
    '"postoperative pain"[Title/Abstract] OR "post-operative pain"[Title/Abstract]'
    ' OR "surgical pain"[Title/Abstract])\n\n'
)

FEW_SHOT_SCREENING = (
    'Question: "Does mammography screening reduce breast cancer mortality?"\n'
    'Output: ("Mammography"[MeSH] OR mammography[Title/Abstract] OR '
    'mammogram[Title/Abstract] OR "breast screening"[Title/Abstract] OR '
    '"screening mammography"[Title/Abstract]) AND ("Breast Neoplasms"[MeSH] OR '
    '"breast cancer"[Title/Abstract] OR "breast carcinoma"[Title/Abstract] OR '
    '"breast tumor"[Title/Abstract]) AND ("Mortality"[MeSH] OR '
    'mortality[Title/Abstract] OR death[Title/Abstract] OR '
    'survival[Title/Abstract])\n\n'
)

FEW_SHOT_PREVENTION = (
    'Question: "Does HPV vaccination prevent cervical cancer in adolescents?"\n'
    'Output: ("Papillomavirus Vaccines"[MeSH] OR "HPV vaccine"[Title/Abstract] '
    'OR "HPV vaccination"[Title/Abstract] OR "papillomavirus vaccine"[Title/Abstract]'
    ' OR Gardasil[Title/Abstract]) AND ("Uterine Cervical Neoplasms"[MeSH] OR '
    '"cervical cancer"[Title/Abstract] OR "cervical neoplasm"[Title/Abstract] '
    'OR "cervical carcinoma"[Title/Abstract]) AND ("Primary Prevention"[MeSH] '
    'OR prevention[Title/Abstract] OR prophylaxis[Title/Abstract] OR '
    '"preventive"[Title/Abstract])\n\n'
)

COT_BLOCK = (
    "Reasoning (think briefly, do not print steps 1-2):\n"
    "Step 1: list the 2-3 atomic concepts in the question (intervention, "
    "condition, comparator).\n"
    "Step 2: for each concept, list its canonical MeSH descriptor plus 3-5 "
    "free-text synonyms.\n"
    "Step 3: assemble each concept block as (MeSH OR synonyms) and join blocks "
    "with AND.\n\n"
)

MESH_HINT = (
    "- HINT: prefer broader MeSH descriptors (e.g. \"Neoplasms\"[MeSH] over "
    "\"Triple Negative Breast Neoplasms\"[MeSH]) — over-specific MeSH terms "
    "drastically reduce recall.\n"
)

TIGHT_OUTPUT_FORMAT_INSTRUCTION = (
    "- STRICT OUTPUT: respond with ONE single line consisting of the Boolean "
    "string only. No prose, no markdown, no leading whitespace, no quotes "
    "wrapping the entire string.\n"
)


def _op_add_atomicity(template: str) -> str:
    """Insert the atomicity rule into the Rules block."""
    if "ATOMICITY:" in template:
        return template
    return template.replace(
        "- Use Boolean: AND between concepts, OR between synonyms\n",
        "- Use Boolean: AND between concepts, OR between synonyms\n"
        + ATOMICITY_RULE,
        1,
    )


def _op_add_few_shot_psychology(template: str) -> str:
    if "mindfulness-based stress reduction" in template:
        return template
    return _insert_before_output(template, FEW_SHOT_PSYCHOLOGY)


def _op_add_few_shot_surgery(template: str) -> str:
    if "Cholecystectomy" in template:
        return template
    return _insert_before_output(template, FEW_SHOT_SURGERY)


def _op_add_few_shot_screening(template: str) -> str:
    if "Mammography" in template:
        return template
    return _insert_before_output(template, FEW_SHOT_SCREENING)


def _op_add_few_shot_prevention(template: str) -> str:
    if "Papillomavirus Vaccines" in template:
        return template
    return _insert_before_output(template, FEW_SHOT_PREVENTION)


def _op_add_cot(template: str) -> str:
    if "Step 1:" in template:
        return template
    # Insert CoT block right before the Examples section.
    if "Examples:\n" in template:
        return template.replace("Examples:\n", COT_BLOCK + "Examples:\n", 1)
    # Fallback: insert before final "Output:"
    return _insert_before_output(template, COT_BLOCK)


def _op_remove_cot(template: str) -> str:
    if "Step 1:" not in template:
        return template
    return template.replace(COT_BLOCK, "", 1)


def _op_add_mesh_hint(template: str) -> str:
    if "HINT: prefer broader MeSH" in template:
        return template
    return template.replace(
        "- All terms in English\n",
        "- All terms in English\n" + MESH_HINT,
        1,
    )


def _op_tighten_output_format(template: str) -> str:
    if "STRICT OUTPUT" in template:
        return template
    return template.replace(
        "- Output ONLY the search string, one line, no explanation\n",
        TIGHT_OUTPUT_FORMAT_INSTRUCTION,
        1,
    )


def _op_loosen_and_count_to_4(template: str) -> str:
    if "MAXIMUM 4 AND" in template:
        return template
    return template.replace(
        "- Use MAXIMUM 3 AND clauses (intervention AND condition AND comparator)",
        "- Use MAXIMUM 4 AND clauses (intervention AND condition AND comparator AND population_or_design)",
        1,
    )


def _insert_before_output(template: str, block: str) -> str:
    """Insert ``block`` immediately before the final ``Output:`` line."""
    marker = "Output:"
    # Find the LAST occurrence of "Output:" — earlier ones may be inside few-shots.
    idx = template.rfind(marker)
    if idx < 0:
        return template + "\n" + block
    return template[:idx] + block + template[idx:]


# Registry of single-step operators. The orchestrator applies them one at a
# time on top of the current best non-broken variant.
OPERATORS: dict[str, Callable[[str], str]] = {
    "add_atomicity": _op_add_atomicity,
    "add_few_shot_psychology": _op_add_few_shot_psychology,
    "add_few_shot_surgery": _op_add_few_shot_surgery,
    "add_few_shot_screening": _op_add_few_shot_screening,
    "add_few_shot_prevention": _op_add_few_shot_prevention,
    "add_cot": _op_add_cot,
    "remove_cot": _op_remove_cot,
    "add_mesh_hint": _op_add_mesh_hint,
    "tighten_output_format": _op_tighten_output_format,
    "loosen_and_count_to_4": _op_loosen_and_count_to_4,
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str, log_file: Path | None = None) -> None:
    """Print structured stdout line and (optionally) append to log_file."""
    line = f"[T17.A2] {msg}"
    print(line, flush=True)
    if log_file is not None:
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # logging failures must not interrupt the iteration loop
            traceback.print_exc()


def load_initial_variants(prompts_dir: Path) -> dict[str, str]:
    """Read V0..V5 from disk; if any missing seed from INITIAL_VARIANTS.

    The V{N}.txt files are the canonical source. Seeds are a safety net for
    accidental deletion of V0; V1-V5 seeds are intentionally empty and the
    orchestrator will abort if any of them is missing on disk AND empty in
    the seed dict (the user must supply the prompts).
    """
    prompts_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for name in VARIANT_NAMES_INITIAL:
        path = prompts_dir / f"{name}.txt"
        if path.exists():
            out[name] = path.read_text(encoding="utf-8")
        else:
            seed = INITIAL_VARIANTS.get(name, "")
            if not seed:
                raise FileNotFoundError(
                    f"Variant {name} missing on disk ({path}) and no seed "
                    f"available. Provide the template manually."
                )
            path.write_text(seed, encoding="utf-8")
            out[name] = seed
    return out


def _write_template(prompts_dir: Path, name: str, template: str) -> Path:
    """Persist a freshly proposed template to ``prompts/{name}.txt``."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    path = prompts_dir / f"{name}.txt"
    path.write_text(template, encoding="utf-8")
    return path


def run_variant(
    name: str,
    template: str,
    dev_file: Path,
    a1_instance: Any,
    output_dir: Path,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one variant against DEV. Returns the metrics dict."""
    from experiments.eval_hero_a1 import run_eval  # local import to avoid Bio at module import

    variant_dir = output_dir / "cache" / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = variant_dir  # Entrez esearch cache (per-variant)
    a1_output_cache = variant_dir / "a1_outputs.json"
    output_path = variant_dir / "eval.json"

    template_path = output_dir / "prompts" / f"{name}.txt"

    result: dict[str, Any] = {
        "name": name,
        "macro": None,
        "nonzero": None,
        "runtime_s": None,
        "n_nonempty": None,
        "queries_sample": [],
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
                f"nonzero={result['nonzero']}",
                log_file,
            )
            return result
        except Exception:
            # corrupted cache — re-run
            output_path.unlink(missing_ok=True)

    t0 = time.time()
    try:
        eval_result = run_eval(
            sample_file=dev_file,
            output=output_path,
            cache_dir=cache_dir,
            a1_output_cache=a1_output_cache,
            intervention=True,  # always-on mesh_validator (T17.A1.1 baseline)
            a1_instance=a1_instance,
            prompt_template=template,
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
        f"variant {name}: macro={result['macro']} nonzero={result['nonzero']} "
        f"n_nonempty={result['n_nonempty']} runtime={result['runtime_s']}s",
        log_file,
    )
    return result


def _extract_metrics(
    name: str,
    eval_result: dict[str, Any],
    template_path: Path,
) -> dict[str, Any]:
    """Pull the metrics we care about out of an eval_hero_a1 result dict."""
    per_sample = eval_result.get("per_sample", []) or []
    aggregates = eval_result.get("aggregates", {}) or {}
    recall_block = aggregates.get("recall_at_1000") or {}
    macro = recall_block.get("mean")

    queries: list[str] = []
    n_nonempty = 0
    for r in per_sample:
        q = r.get("a1_query") or ""
        if q.strip():
            n_nonempty += 1
        if len(queries) < 2 and q.strip():
            queries.append(q)

    # nonzero rate = fraction of samples with recall@1000 > 0
    recall_values = [
        r.get("recall_at_1000")
        for r in per_sample
        if isinstance(r.get("recall_at_1000"), (int, float))
    ]
    nonzero = (
        round(sum(1 for v in recall_values if v and v > 0) / len(recall_values), 4)
        if recall_values
        else None
    )

    broken = (n_nonempty == 0) or (macro is None)

    out: dict[str, Any] = {
        "name": name,
        "macro": macro,
        "nonzero": nonzero,
        "n_nonempty": n_nonempty,
        "queries_sample": queries,
        "template_path": str(template_path),
        "broken": broken,
    }
    return out


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
      - Identify the best non-broken variant by macro recall.
      - Iterate over OPERATORS in registry order; for each operator name,
        check whether ``f"{best}+{op}"`` already exists in history; if not,
        apply the operator and (if the result is materially different from
        the best template) return it.
      - When all single-op derivations from the best are exhausted, try
        pairwise combinations of the top two operators seen so far in history
        (by macro delta over their parent).
      - Returns ``None`` when truly nothing new can be made.
    """
    non_broken = [h for h in history if not h.get("broken")]
    if not non_broken:
        return None

    # Sort by macro desc (None last). Stable sort preserves discovery order
    # within ties so the "best so far" is consistent.
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
        if new_template == best_template:
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
                new_template = OPERATORS[second](intermediate)
            except Exception:
                traceback.print_exc()
                continue
            if new_template == best_template:
                seen_names.add(combo_name)
                continue
            return combo_name, new_template

    return None


def _collect_operator_deltas(history: list[dict[str, Any]]) -> dict[str, float]:
    """For each operator suffix observed in history, compute mean macro delta vs parent."""
    deltas: dict[str, list[float]] = {}
    by_name = {h["name"]: h for h in history}
    for h in history:
        if h.get("broken"):
            continue
        name = h["name"]
        # Names look like "V0+op_a" or "V0+op_a+op_b" or "V0".
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
    # Local import so the orchestrator can be imported in tests without GPU/Bio.
    from agents.agent1_protocol import Agent1Protocol  # noqa: WPS433

    return Agent1Protocol()


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
        "nonzero": best.get("nonzero"),
        "runtime_s": best.get("runtime_s"),
        "n_nonempty": best.get("n_nonempty"),
        "queries_sample": best.get("queries_sample", []),
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
) -> dict[str, Any]:
    """Run the full iteration loop. Returns the final summary dict.

    Parameters mirror the CLI flags. ``a1_instance`` defaults to a freshly
    built ``Agent1Protocol`` (one model load); tests can inject a mock.
    """
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
        # broken variants don't move the counter
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
            "nonzero": best.get("nonzero") if best else None,
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
            "T17.A2 prompt-iteration orchestrator. Greedy local search over "
            "A1.generate_pubmed_query prompt variants against a DEV sample."
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
        default=PROJECT_ROOT / "experiments" / "T17.A2",
        help="Directory for prompts/, cache/, iteration_log.json (default experiments/T17.A2/).",
    )
    parser.add_argument(
        "--target-macro",
        type=float,
        default=0.12,
        help="Floor target macro recall (default 0.12).",
    )
    parser.add_argument(
        "--ambitious-macro",
        type=float,
        default=0.15,
        help="Ambitious target macro recall — triggers early stop when sustained (default 0.15).",
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
        help="Skip earlier initial variants and start from this one (e.g. V3). Useful for resume.",
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
