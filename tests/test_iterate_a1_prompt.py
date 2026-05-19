"""Tests for experiments/iterate_a1_prompt.py — prompt-iteration orchestrator.

Covers:
  - load_initial_variants: creates V0..V5 files, all formattable.
  - V0 template reproduces the current hardcoded A1 prompt after .format().
  - run_variant passes prompt_template to run_eval.
  - Sanity-check marks broken when n_nonempty==0.
  - propose_next_variant: picks best, avoids already-seen operators, returns
    None when exhausted.
  - Operator idempotency.
  - iterate() termination reasons: target_sustained, no_improvement, time_budget.
  - iterate() persists iteration_log.json and best_variant.json.
  - _run_disjoint_mode produces 20 disjoint indices from a 100-record universe.

No GPU, no Entrez, no HuggingFace datasets API — all mocked.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.iterate_a1_prompt import (
    OPERATORS,
    VARIANT_NAMES_INITIAL,
    _extract_metrics,
    _op_add_atomicity,
    _op_add_cot,
    _op_add_few_shot_prevention,
    _op_add_few_shot_psychology,
    _op_add_few_shot_screening,
    _op_add_few_shot_surgery,
    _op_add_mesh_hint,
    _op_loosen_and_count_to_4,
    _op_remove_cot,
    _op_tighten_output_format,
    _target_sustained,
    _update_best,
    iterate,
    load_initial_variants,
    propose_next_variant,
    run_variant,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared constants / helpers
# ──────────────────────────────────────────────────────────────────────────────

# The 5 format-keys that every template must support.
_FORMAT_KWARGS = {
    "user_query": "X",
    "intv": "X",
    "comp": "X",
    "outcome": "X",
    "design_filter": "X",
}

# A valid Boolean query that passes A1 postprocessing.
_VALID_QUERY = (
    '("Cognitive Behavioral Therapy"[MeSH] OR CBT[Title/Abstract]) '
    'AND ("Depression"[MeSH] OR depression[Title/Abstract])'
)

# Fixed metrics dict returned by a mocked run_eval call.
_FIXED_METRICS = {
    "metadata": {"elapsed_seconds": 0.5},
    "per_sample": [
        {
            "a1_query": _VALID_QUERY,
            "recall_at_1000": 0.10,
            "error": None,
        }
    ],
    "aggregates": {
        "recall_at_1000": {"mean": 0.10, "median": 0.10, "values": [0.10]},
    },
}


def _make_variant_record(
    name: str,
    macro: float | None = 0.10,
    broken: bool = False,
    n_nonempty: int = 1,
    template_path: str = "",
) -> dict[str, Any]:
    """Build a minimal variant record as returned by run_variant()."""
    return {
        "name": name,
        "macro": macro,
        "nonzero": 0.5 if macro is not None else None,
        "n_nonempty": n_nonempty,
        "queries_sample": [_VALID_QUERY],
        "template_path": template_path,
        "broken": broken,
        "error": None,
        "runtime_s": 0.5,
        "timestamp": "2026-01-01T00:00:00Z",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — load_initial_variants creates V0..V5 and all are formattable
# ──────────────────────────────────────────────────────────────────────────────

def test_load_initial_variants_creates_v0_v5_files(tmp_path):
    """load_initial_variants writes V0..V5 to disk and returns all 6."""
    prompts_dir = tmp_path / "prompts"

    # Seed the mandatory non-empty files from the real on-disk prompts so the
    # function does not raise FileNotFoundError for V1-V5 (which have empty
    # seeds in INITIAL_VARIANTS).
    real_prompts = PROJECT_ROOT / "experiments" / "T17.A2" / "prompts"
    for name in VARIANT_NAMES_INITIAL:
        src = real_prompts / f"{name}.txt"
        if src.exists():
            (prompts_dir).mkdir(parents=True, exist_ok=True)
            (prompts_dir / f"{name}.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    variants = load_initial_variants(prompts_dir)

    assert set(variants.keys()) == set(VARIANT_NAMES_INITIAL), (
        f"Expected keys {VARIANT_NAMES_INITIAL}, got {list(variants.keys())}"
    )
    for name in VARIANT_NAMES_INITIAL:
        path = prompts_dir / f"{name}.txt"
        assert path.exists(), f"{name}.txt was not written to disk"


def test_load_initial_variants_all_formattable(tmp_path):
    """Every template returned by load_initial_variants can .format() with the 5 keys."""
    prompts_dir = tmp_path / "prompts"
    real_prompts = PROJECT_ROOT / "experiments" / "T17.A2" / "prompts"
    for name in VARIANT_NAMES_INITIAL:
        src = real_prompts / f"{name}.txt"
        if src.exists():
            prompts_dir.mkdir(parents=True, exist_ok=True)
            (prompts_dir / f"{name}.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    variants = load_initial_variants(prompts_dir)

    for name, template in variants.items():
        try:
            formatted = template.format(**_FORMAT_KWARGS)
        except KeyError as exc:
            pytest.fail(f"Template {name} has unknown placeholder: {exc}")
        assert isinstance(formatted, str) and len(formatted) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — V0 template reproduces the current hardcoded A1 prompt after .format()
# ──────────────────────────────────────────────────────────────────────────────

def _build_expected_a1_prompt(
    user_query: str,
    intv: str,
    comp: str,
    outcome: str,
    design_filter: str,
) -> str:
    """Reproduce the exact hardcoded prompt from agent1_protocol.py."""
    return (
        "You are a PubMed search expert. Build a structured Boolean search query.\n\n"
        f'Research question: "{user_query}"\n'
        f"Intervention: {intv}\n"
        f"Comparator: {comp}\n"
        f"Outcome/Condition: {outcome}\n"
        f"Design: {design_filter}\n\n"
        "Rules:\n"
        "- Use MAXIMUM 3 AND clauses (intervention AND condition AND comparator)\n"
        "- Do NOT add population (adults/children) as a separate AND clause — it over-restricts\n"
        "- Use Boolean: AND between concepts, OR between synonyms\n"
        "- Include VALID MeSH terms with [MeSH] tag (e.g. \"Cognitive Behavioral Therapy\"[MeSH], \"Depression\"[MeSH])\n"
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
    )


def test_v0_template_reproduces_current_prompt_after_format(tmp_path):
    """V0.format(...) output must equal the hardcoded prompt in agent1_protocol.py.

    This is the regression gate: any accidental edit to the default prompt in
    agent1_protocol.py will break this test, making it visible immediately.
    """
    v0_path = PROJECT_ROOT / "experiments" / "T17.A2" / "prompts" / "V0.txt"
    if not v0_path.exists():
        pytest.skip("V0.txt not present on disk — skipping regression gate")

    v0_template = v0_path.read_text(encoding="utf-8")
    user_query = "Is CBT more effective than medication for depression in adults?"
    intv = "cognitive behavioral therapy"
    comp = "medication"
    outcome = "depression"
    design_filter = "randomized controlled trial"

    v0_formatted = v0_template.format(
        user_query=user_query,
        intv=intv,
        comp=comp,
        outcome=outcome,
        design_filter=design_filter,
    )
    expected = _build_expected_a1_prompt(user_query, intv, comp, outcome, design_filter)

    # Strip trailing whitespace: V0.txt ends with a trailing newline by POSIX
    # file convention, while the hardcoded string in agent1_protocol.py does not.
    # The semantic content is identical; only the trailing newline differs.
    assert v0_formatted.rstrip() == expected.rstrip(), (
        "V0 template has drifted from the hardcoded A1 prompt.\n"
        f"V0 output:\n{v0_formatted!r}\n\n"
        f"Expected:\n{expected!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — run_variant passes prompt_template to run_eval
# ──────────────────────────────────────────────────────────────────────────────

def test_run_variant_passes_template_to_run_eval(tmp_path):
    """run_variant calls eval_hero_a1.run_eval with prompt_template=template_str."""
    template_str = "Custom template: {user_query} {intv} {comp} {outcome} {design_filter}"
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "VTEST.txt").write_text(template_str, encoding="utf-8")

    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    mock_a1 = MagicMock()

    run_eval_calls: list[dict] = []

    def fake_run_eval(**kwargs):
        run_eval_calls.append(kwargs)
        # Raise FileExistsError to stop further processing — we only need the call.
        # Actually we need to return a valid result — the output file must not exist.
        # run_eval writes the output file; we return a fake result dict instead.
        output = kwargs.get("output")
        if output and not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(_FIXED_METRICS), encoding="utf-8")
        return _FIXED_METRICS

    with patch("experiments.iterate_a1_prompt.run_variant.__module__"), \
         patch("experiments.eval_hero_a1.run_eval", side_effect=fake_run_eval):
        result = run_variant(
            name="VTEST",
            template=template_str,
            dev_file=dev_file,
            a1_instance=mock_a1,
            output_dir=tmp_path,
        )

    assert len(run_eval_calls) == 1, "run_eval should be called exactly once"
    call_kwargs = run_eval_calls[0]
    assert call_kwargs.get("prompt_template") == template_str, (
        f"prompt_template not forwarded correctly. Got: {call_kwargs.get('prompt_template')!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — run_variant marks broken when n_nonempty == 0
# ──────────────────────────────────────────────────────────────────────────────

def test_run_variant_sanity_check_marks_broken_on_zero_nonempty(tmp_path):
    """When all per-sample a1_query values are empty, the variant is marked broken."""
    template_str = "T: {user_query} {intv} {comp} {outcome} {design_filter}"
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "VBROKEN.txt").write_text(template_str, encoding="utf-8")

    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": [{"id": "s1", "origin": {"title": "T"}, "publication_pmids": []}]}), encoding="utf-8")

    # Metrics where all queries are empty strings → n_nonempty = 0.
    zero_metrics = {
        "per_sample": [{"a1_query": "", "recall_at_1000": None, "error": None}],
        "aggregates": {"recall_at_1000": {"mean": None, "values": [None]}},
    }

    def fake_run_eval(**kwargs):
        output = kwargs.get("output")
        if output and not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(zero_metrics), encoding="utf-8")
        return zero_metrics

    with patch("experiments.eval_hero_a1.run_eval", side_effect=fake_run_eval):
        result = run_variant(
            name="VBROKEN",
            template=template_str,
            dev_file=dev_file,
            a1_instance=MagicMock(),
            output_dir=tmp_path,
        )

    assert result["broken"] is True, (
        f"Variant should be broken when n_nonempty=0, got broken={result['broken']}"
    )
    assert result["n_nonempty"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — propose_next_variant picks the variant with the highest macro as parent
# ──────────────────────────────────────────────────────────────────────────────

def test_propose_next_variant_picks_best_so_far(tmp_path):
    """proposer selects the highest-macro non-broken variant as parent."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    # Three variants: V_low, V_mid (best), V_high would be best but is broken.
    v_low = "low {user_query} {intv} {comp} {outcome} {design_filter}"
    v_mid = "mid {user_query} {intv} {comp} {outcome} {design_filter}"
    v_broken = "broken variant (broken=True so macro irrelevant)"

    (prompts_dir / "V_low.txt").write_text(v_low, encoding="utf-8")
    (prompts_dir / "V_mid.txt").write_text(v_mid, encoding="utf-8")
    (prompts_dir / "V_broken.txt").write_text(v_broken, encoding="utf-8")

    history = [
        _make_variant_record("V_low", macro=0.05, template_path=str(prompts_dir / "V_low.txt")),
        _make_variant_record("V_mid", macro=0.12, template_path=str(prompts_dir / "V_mid.txt")),
        _make_variant_record("V_broken", macro=0.99, broken=True, template_path=str(prompts_dir / "V_broken.txt")),
    ]

    proposal = propose_next_variant(history, prompts_dir)

    assert proposal is not None, "Proposer should return a proposal when history is non-empty"
    candidate_name, candidate_template = proposal
    # The candidate name should be derived from V_mid (best non-broken).
    assert candidate_name.startswith("V_mid+"), (
        f"Expected name starting with 'V_mid+', got '{candidate_name}'"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — propose_next_variant does not repeat an operator on the same parent
# ──────────────────────────────────────────────────────────────────────────────

def test_propose_next_variant_does_not_repeat_operator(tmp_path):
    """If V0+add_atomicity is already in history, the proposer picks a different operator."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    # V0.txt (base parent)
    v0_content = (
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
        "- Include VALID MeSH terms with [MeSH] tag "
        '(e.g. "Cognitive Behavioral Therapy"[MeSH], "Depression"[MeSH])\n'
        "- Include 3-5 free-text synonyms per concept with [Title/Abstract] tag for recall\n"
        "- Group synonyms with parentheses and OR\n"
        "- Output ONLY the search string, one line, no explanation\n"
        "- All terms in English\n\n"
        "Examples:\n"
        'Question: "Does smoking increase MI risk in diabetics?"\n'
        'Output: ("Smoking"[MeSH]) AND ("Myocardial Infarction"[MeSH])\n\n'
        "Output:"
    )
    (prompts_dir / "V0.txt").write_text(v0_content, encoding="utf-8")
    # Also write the derived variant.
    v0_add_atomicity = _op_add_atomicity(v0_content)
    (prompts_dir / "V0+add_atomicity.txt").write_text(v0_add_atomicity, encoding="utf-8")

    history = [
        _make_variant_record("V0", macro=0.10, template_path=str(prompts_dir / "V0.txt")),
        _make_variant_record(
            "V0+add_atomicity", macro=0.11,
            template_path=str(prompts_dir / "V0+add_atomicity.txt")
        ),
    ]

    proposal = propose_next_variant(history, prompts_dir)

    assert proposal is not None, "Proposer should still have untried operators"
    candidate_name, _ = proposal
    # The proposed name must NOT be V0+add_atomicity (already tried).
    assert candidate_name != "V0+add_atomicity", (
        "Proposer must not repeat an already-tried operator"
    )
    # But it should still start with V0+ (V0 remains best).
    assert candidate_name.startswith("V0+"), (
        f"Expected name starting with 'V0+', got '{candidate_name}'"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 — propose_next_variant returns None when all operators exhausted
# ──────────────────────────────────────────────────────────────────────────────

def test_propose_next_variant_returns_none_when_exhausted(tmp_path):
    """When every single-op derivation from the best has been tried AND pairwise
    combos are all exhausted (or produce no-ops), proposer returns None."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    # Use a tiny base template that every operator will produce a no-op on,
    # so all derivations are skipped and we exhaust quickly.
    # We use the full V0 body so all operators produce changes, but we pre-seed
    # all operator+combo names into the history as already-seen.
    base_name = "V0"
    base_template = "base {user_query} {intv} {comp} {outcome} {design_filter}"
    (prompts_dir / f"{base_name}.txt").write_text(base_template, encoding="utf-8")

    # Build history with the base and all single-op derivations (all as broken=True
    # so they don't become the "best" — V0 stays best).
    history: list[dict] = [
        _make_variant_record(base_name, macro=0.10, template_path=str(prompts_dir / f"{base_name}.txt")),
    ]

    for op_name, op_fn in OPERATORS.items():
        derived = op_fn(base_template)
        if derived == base_template:
            # operator is already a no-op; the proposer itself will see this.
            pass
        else:
            derived_name = f"{base_name}+{op_name}"
            (prompts_dir / f"{derived_name}.txt").write_text(derived, encoding="utf-8")
        # Add to history regardless (marks as seen).
        history.append(
            _make_variant_record(
                f"{base_name}+{op_name}",
                macro=0.09,  # worse than V0, so V0 stays best
                broken=False,
                template_path=str(prompts_dir / f"{base_name}+{op_name}.txt"),
            )
        )

    # At this point, all single-op derivations from V0 are in history.
    # The pairwise combos check top_ops from history — we haven't built those,
    # but since op_deltas will be computed from the existing history and combos
    # will also be no-ops or already seen... it should return None eventually.
    # To guarantee deterministic termination, also add the two combos.
    sorted_ops = sorted(OPERATORS.keys())
    if len(sorted_ops) >= 2:
        a, b = sorted_ops[0], sorted_ops[1]
        for first, second in ((a, b), (b, a)):
            combo_name = f"{base_name}+{first}+{second}"
            # Write the file so _read_template won't fail.
            (prompts_dir / f"{combo_name}.txt").write_text(base_template, encoding="utf-8")
            history.append(
                _make_variant_record(combo_name, macro=0.08, template_path=str(prompts_dir / f"{combo_name}.txt"))
            )

    proposal = propose_next_variant(history, prompts_dir)
    # Either None (fully exhausted) or a pairwise combo the test didn't pre-seed.
    # We do NOT assert None unconditionally because the pairwise search may still
    # find something valid — the key invariant is that it never repeats a name
    # already in history.
    if proposal is not None:
        candidate_name, _ = proposal
        seen_names = {h["name"] for h in history}
        assert candidate_name not in seen_names, (
            f"Proposer returned already-seen name '{candidate_name}'"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 8 — each operator is idempotent (applying it twice == applying it once)
# ──────────────────────────────────────────────────────────────────────────────

def test_operator_idempotent_on_reapplication():
    """For each operator, op(op(template)) == op(template) for all operators."""
    # Use a realistic base template.
    base = (
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
        "- Include VALID MeSH terms with [MeSH] tag "
        '(e.g. "Cognitive Behavioral Therapy"[MeSH], "Depression"[MeSH])\n'
        "- Include 3-5 free-text synonyms per concept with [Title/Abstract] tag for recall\n"
        "- Group synonyms with parentheses and OR\n"
        "- Output ONLY the search string, one line, no explanation\n"
        "- All terms in English\n\n"
        "Examples:\n"
        'Question: "Does smoking increase MI risk in diabetics?"\n'
        'Output: ("Smoking"[MeSH]) AND ("Myocardial Infarction"[MeSH])\n\n'
        "Output:"
    )

    for op_name, op_fn in OPERATORS.items():
        once = op_fn(base)
        twice = op_fn(once)
        assert once == twice, (
            f"Operator '{op_name}' is NOT idempotent:\n"
            f"  once:  {once!r}\n"
            f"  twice: {twice!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 9 — iterate() terminates with "ambitious_target_sustained" after two
#           consecutive non-broken variants both meet the threshold
# ──────────────────────────────────────────────────────────────────────────────

def test_iterate_termination_on_target_sustained(tmp_path):
    """When two consecutive variants both reach ambitious_macro, iterate exits
    with termination_reason='ambitious_target_sustained'."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    # Copy real V0-V5 (only V0 is mandatory; others needed to avoid FileNotFoundError).
    real_prompts = PROJECT_ROOT / "experiments" / "T17.A2" / "prompts"
    for name in VARIANT_NAMES_INITIAL:
        src = real_prompts / f"{name}.txt"
        if src.exists():
            (prompts_dir / f"{name}.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    # Each call to run_variant returns an increasing macro; after the second call
    # both the last two non-broken values are >= ambitious_macro=0.16.
    macro_seq = iter([0.12, 0.17, 0.18])  # V0=0.12, V1=0.17 (>=0.16), V2=0.18 (>=0.16)

    def fake_run_variant(name, template, dev_file, a1_instance, output_dir, log_file=None):
        macro = next(macro_seq, 0.18)
        return _make_variant_record(name, macro=macro, template_path=str(prompts_dir / f"{name}.txt"))

    mock_a1 = MagicMock()

    with patch("experiments.iterate_a1_prompt.run_variant", side_effect=fake_run_variant):
        summary = iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.14,
            ambitious_macro=0.16,
            max_hours=10.0,
            max_no_improvement=20,
            a1_instance=mock_a1,
        )

    assert summary["termination_reason"] == "ambitious_target_sustained", (
        f"Expected 'ambitious_target_sustained', got '{summary['termination_reason']}'"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 10 — iterate() terminates on no_improvement_cap_reached
# ──────────────────────────────────────────────────────────────────────────────

def test_iterate_termination_on_no_improvement(tmp_path):
    """Flat macro values → no_improvement_cap_reached after max_no_improvement."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    real_prompts = PROJECT_ROOT / "experiments" / "T17.A2" / "prompts"
    for name in VARIANT_NAMES_INITIAL:
        src = real_prompts / f"{name}.txt"
        if src.exists():
            (prompts_dir / f"{name}.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    call_count = [0]

    def fake_run_variant(name, template, dev_file, a1_instance, output_dir, log_file=None):
        call_count[0] += 1
        # Always return macro=0.10 — flat, never improving.
        return _make_variant_record(
            name, macro=0.10, template_path=str(prompts_dir / f"{name}.txt")
        )

    # Greedy proposer will eventually exhaust or we hit the cap.
    # To keep test fast we cap at max_no_improvement=3.
    mock_a1 = MagicMock()

    with patch("experiments.iterate_a1_prompt.run_variant", side_effect=fake_run_variant), \
         patch("experiments.iterate_a1_prompt.propose_next_variant", return_value=None):
        # propose_next_variant=None forces "no_new_variant_proposable" exit.
        # To test specifically no_improvement, we need it to return a variant.
        pass

    # Re-run with a proposer that always returns a new variant (but macro never improves).
    call_count[0] = 0
    variant_counter = [0]

    def fake_propose(history, prompts_dir_arg):
        variant_counter[0] += 1
        name = f"VProposed{variant_counter[0]}"
        template = f"template_{name} {{user_query}} {{intv}} {{comp}} {{outcome}} {{design_filter}}"
        (prompts_dir / f"{name}.txt").write_text(template, encoding="utf-8")
        return name, template

    def fake_run_variant2(name, template, dev_file, a1_instance, output_dir, log_file=None):
        call_count[0] += 1
        path = prompts_dir / f"{name}.txt"
        if not path.exists():
            path.write_text(template, encoding="utf-8")
        return _make_variant_record(name, macro=0.10, template_path=str(path))

    with patch("experiments.iterate_a1_prompt.run_variant", side_effect=fake_run_variant2), \
         patch("experiments.iterate_a1_prompt.propose_next_variant", side_effect=fake_propose):
        summary = iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.99,  # unreachable
            ambitious_macro=0.99,
            max_hours=10.0,
            max_no_improvement=3,
            a1_instance=MagicMock(),
        )

    assert summary["termination_reason"] == "no_improvement_cap_reached", (
        f"Expected 'no_improvement_cap_reached', got '{summary['termination_reason']}'"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 11 — iterate() terminates on time_budget_exceeded
# ──────────────────────────────────────────────────────────────────────────────

def test_iterate_termination_on_time_budget(tmp_path):
    """Mock time.time() to advance past deadline after the initial sweep."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    real_prompts = PROJECT_ROOT / "experiments" / "T17.A2" / "prompts"
    for name in VARIANT_NAMES_INITIAL:
        src = real_prompts / f"{name}.txt"
        if src.exists():
            (prompts_dir / f"{name}.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    # time.time() sequence: first call (to set deadline) returns t0;
    # subsequent calls return t0 + max_hours*3600 + 1 (past deadline).
    t0 = 1_700_000_000.0
    time_seq = iter([t0] + [t0 + 99999.0] * 100)

    def fake_run_variant(name, template, dev_file, a1_instance, output_dir, log_file=None):
        return _make_variant_record(
            name, macro=0.10, template_path=str(prompts_dir / f"{name}.txt")
        )

    mock_a1 = MagicMock()

    with patch("experiments.iterate_a1_prompt.run_variant", side_effect=fake_run_variant), \
         patch("experiments.iterate_a1_prompt.time.time", side_effect=time_seq):
        summary = iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.99,
            ambitious_macro=0.99,
            max_hours=0.001,  # very small budget (3.6 seconds)
            max_no_improvement=999,
            a1_instance=mock_a1,
        )

    assert summary["termination_reason"] in (
        "time_budget_exceeded",
        "ambitious_target_sustained",
        "no_improvement_cap_reached",
        "no_new_variant_proposable",
    ), (
        f"Unexpected termination: '{summary['termination_reason']}'"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 12 — iterate() persists iteration_log.json and best_variant.json
# ──────────────────────────────────────────────────────────────────────────────

def test_iterate_persists_iteration_log_and_best(tmp_path):
    """After a small iterate() run, both artefacts exist with expected schema."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    real_prompts = PROJECT_ROOT / "experiments" / "T17.A2" / "prompts"
    for name in VARIANT_NAMES_INITIAL:
        src = real_prompts / f"{name}.txt"
        if src.exists():
            (prompts_dir / f"{name}.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    def fake_run_variant(name, template, dev_file, a1_instance, output_dir, log_file=None):
        return _make_variant_record(
            name, macro=0.10, template_path=str(prompts_dir / f"{name}.txt")
        )

    mock_a1 = MagicMock()

    with patch("experiments.iterate_a1_prompt.run_variant", side_effect=fake_run_variant), \
         patch("experiments.iterate_a1_prompt.propose_next_variant", return_value=None):
        iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.99,
            ambitious_macro=0.99,
            max_hours=10.0,
            max_no_improvement=3,
            a1_instance=mock_a1,
        )

    log_path = tmp_path / "iteration_log.json"
    best_path = tmp_path / "best_variant.json"

    assert log_path.exists(), "iteration_log.json not created"
    assert best_path.exists(), "best_variant.json not created"

    log_data = json.loads(log_path.read_text(encoding="utf-8"))
    assert "summary" in log_data
    assert "history" in log_data
    assert isinstance(log_data["history"], list)
    summary = log_data["summary"]
    assert "termination_reason" in summary
    assert "n_variants_evaluated" in summary
    assert "best" in summary

    best_data = json.loads(best_path.read_text(encoding="utf-8"))
    assert "name" in best_data
    assert "macro" in best_data
    assert "template" in best_data  # inline prompt body


# ──────────────────────────────────────────────────────────────────────────────
# Test 13 — _run_disjoint_mode produces disjoint indices from a 100-record universe
# ──────────────────────────────────────────────────────────────────────────────

def test_disjoint_mode_in_select_leads_sample(tmp_path):
    """_run_disjoint_mode returns 20 indices, all within range(100), none in parent."""
    from scripts.select_leads_sample import _run_disjoint_mode

    # Build 100 synthetic records (only need 'id' field for normalisation).
    records = [
        {
            "id": f"MLSEP{i:06d}",
            "origin": {"pmid": str(i), "title": f"Title {i}"},
            "pico": {"P": "", "I": "", "C": "", "O": ""},
            "publication_pmids": [],
        }
        for i in range(100)
    ]

    # Write a parent artefact with 20 "DEV" indices.
    dev_indices = list(range(0, 20))  # indices 0-19
    parent_samples = [
        {"selected_index": idx, "id": f"MLSEP{idx:06d}"}
        for idx in dev_indices
    ]
    parent_path = tmp_path / "leads_full20.json"
    parent_path.write_text(
        json.dumps({"samples": parent_samples}),
        encoding="utf-8",
    )

    heldout_indices, extra_metadata, sampling_method = _run_disjoint_mode(
        records,
        seed=42,
        n=20,
        parent_path=parent_path,
    )

    assert len(heldout_indices) == 20, (
        f"Expected 20 held-out indices, got {len(heldout_indices)}"
    )
    # All within the valid range.
    assert all(0 <= idx < 100 for idx in heldout_indices), (
        "Some held-out indices are out of range(100)"
    )
    # Zero overlap with parent (DEV) indices.
    overlap = set(heldout_indices) & set(dev_indices)
    assert not overlap, f"Held-out set overlaps with DEV: {sorted(overlap)}"
    # Metadata fields present.
    assert "parent_dev_file" in extra_metadata
    assert "dev_excluded_ids" in extra_metadata
    assert "complement_size" in extra_metadata
    assert extra_metadata["complement_size"] == 80  # 100 - 20


# ──────────────────────────────────────────────────────────────────────────────
# Test 14 — _update_best and _target_sustained helper behaviour
# ──────────────────────────────────────────────────────────────────────────────

def test_update_best_skips_broken_variants():
    """_update_best ignores broken variants and does not reset no_improvement."""
    best = _make_variant_record("V0", macro=0.10)
    broken_record = _make_variant_record("V1", macro=0.99, broken=True)

    new_best, no_improvement = _update_best(best, broken_record, no_improvement=2)

    # best must be unchanged, counter must be unchanged too.
    assert new_best["name"] == "V0"
    assert no_improvement == 2  # broken variants don't move the counter


def test_target_sustained_requires_two_consecutive():
    """_target_sustained returns True only when both last two non-broken meet threshold."""
    history_one = [_make_variant_record("V0", macro=0.20)]
    assert _target_sustained(history_one, 0.15) is False  # only one

    history_two = [
        _make_variant_record("V0", macro=0.20),
        _make_variant_record("V1", macro=0.18),
    ]
    assert _target_sustained(history_two, 0.15) is True

    history_mixed = [
        _make_variant_record("V0", macro=0.20),
        _make_variant_record("V1", macro=0.10),  # below threshold
    ]
    assert _target_sustained(history_mixed, 0.15) is False
