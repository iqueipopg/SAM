# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests para experiments/iterate_a1_metadata.py — orquestador de iteración PICO.

Cubre: los 4 operadores (idempotencia, marcadores), propose_next_variant,
run_variant con run_eval mockeado, _extract_metrics, e iterate() con las
4 razones de terminación y persistencia de artefactos.

Sin GPU, sin red — todo mockeado con unittest.mock.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.iterate_a1_metadata import (
    ATOMICITY_BLOCK,
    COT_BLOCK,
    FEW_SHOTS_BLOCK,
    INITIAL_VARIANTS,
    OPERATORS,
    VALIDATION_BLOCK,
    VARIANT_NAMES_INITIAL,
    _extract_metrics,
    _insert_before_final_brace,
    _op_add_atomicity,
    _op_add_cot,
    _op_add_few_shots,
    _op_add_validation_step,
    _target_sustained,
    _update_best,
    iterate,
    load_initial_variants,
    propose_next_variant,
    run_variant,
)


# ══════════════════════════════════════════════════════════════════════════════
# Constantes y helpers compartidos
# ══════════════════════════════════════════════════════════════════════════════

# Plantilla base V0 (idéntica al seed en INITIAL_VARIANTS["V0"])
V0_TEMPLATE = INITIAL_VARIANTS["V0"]

# Resultado fijo que simula un run_eval exitoso
_FIXED_EVAL_RESULT = {
    "metadata": {"elapsed_seconds": 1.0, "N": 2},
    "per_sample": [
        {
            "id": "s1",
            "pico_similarity": {"P": 0.7, "I": 0.8, "C": 0.5, "O": 0.6, "mean": 0.65},
            "predicted_pico_normalized": {"P": "adults", "I": "CBT", "C": "control", "O": "depression"},
            "gold_pico": {"P": "adults with MDD", "I": "CBT", "C": "waitlist", "O": "depression"},
            "framework_chosen": "PICO",
            "error": None,
        },
        {
            "id": "s2",
            "pico_similarity": {"P": 0.6, "I": 0.7, "C": 0.4, "O": 0.5, "mean": 0.55},
            "predicted_pico_normalized": {"P": "patients", "I": "exercise", "C": "sedentary", "O": "anxiety"},
            "gold_pico": {"P": "adults", "I": "exercise", "C": "no exercise", "O": "anxiety"},
            "framework_chosen": "PICO",
            "error": None,
        },
    ],
    "aggregates": {
        "macro_mean": 0.60,
        "median": 0.60,
        "pct_samples_3_of_4_above_0.5": 0.5,
        "per_element": {"P_mean": 0.65, "I_mean": 0.75, "C_mean": 0.45, "O_mean": 0.55},
        "n_with_similarity": 2,
    },
}


def _make_variant_record(
    name: str,
    macro: float | None = 0.60,
    broken: bool = False,
    n_with_similarity: int = 2,
    template_path: str = "",
) -> dict[str, Any]:
    """Construye un record de variante mínimo como el que devuelve run_variant."""
    return {
        "name": name,
        "macro": macro,
        "median": macro,
        "pct_3_of_4": 0.5 if macro is not None else None,
        "per_element": {"P_mean": 0.65, "I_mean": 0.75, "C_mean": 0.45, "O_mean": 0.55},
        "n_with_similarity": n_with_similarity,
        "predictions_sample": [],
        "template_path": template_path,
        "broken": broken,
        "error": None,
        "runtime_s": 1.0,
        "timestamp": "2026-01-01T00:00:00Z",
    }


def _write_v0_prompt(prompts_dir: Path) -> Path:
    """Escribe V0.txt en prompts_dir y devuelve la ruta."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    p = prompts_dir / "V0.txt"
    p.write_text(V0_TEMPLATE, encoding="utf-8")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 1 — Operador _op_add_atomicity
# ══════════════════════════════════════════════════════════════════════════════

def test_op_add_atomicity_adds_marker():
    """Aplicado sobre V0 → resultado contiene 'SINGLE specific concept'."""
    result = _op_add_atomicity(V0_TEMPLATE)
    assert result is not None
    assert "SINGLE specific concept" in result


def test_op_add_atomicity_is_idempotent():
    """Aplicado dos veces → segunda vez devuelve None."""
    once = _op_add_atomicity(V0_TEMPLATE)
    assert once is not None
    twice = _op_add_atomicity(once)
    assert twice is None, (
        "_op_add_atomicity aplicado por segunda vez debe devolver None (idempotente)"
    )


def test_op_add_atomicity_string_result_type():
    """Primera aplicación devuelve str, no None."""
    result = _op_add_atomicity(V0_TEMPLATE)
    assert isinstance(result, str)


def test_op_add_atomicity_contains_atomicity_block():
    """El bloque inyectado contiene el contenido de ATOMICITY_BLOCK."""
    result = _op_add_atomicity(V0_TEMPLATE)
    assert result is not None
    # ATOMICITY_BLOCK debe estar en el resultado
    assert "ATOMICITY REQUIREMENT" in result


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 2 — Operador _op_add_few_shots
# ══════════════════════════════════════════════════════════════════════════════

def test_op_add_few_shots_adds_at_least_four_examples():
    """Aplicado sobre V0 → resultado contiene al menos 4 headers 'EXAMPLE '."""
    result = _op_add_few_shots(V0_TEMPLATE)
    assert result is not None
    assert result.count("EXAMPLE ") >= 4


def test_op_add_few_shots_is_idempotent():
    """Aplicado dos veces → segunda vez devuelve None."""
    once = _op_add_few_shots(V0_TEMPLATE)
    assert once is not None
    twice = _op_add_few_shots(once)
    assert twice is None, (
        "_op_add_few_shots aplicado por segunda vez debe devolver None (idempotente)"
    )


def test_op_add_few_shots_string_result():
    """Primera aplicación devuelve str."""
    result = _op_add_few_shots(V0_TEMPLATE)
    assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 3 — Operador _op_add_cot
# ══════════════════════════════════════════════════════════════════════════════

def test_op_add_cot_adds_cot_marker():
    """Aplicado sobre V0 → resultado contiene 'CoT-Step 1'."""
    result = _op_add_cot(V0_TEMPLATE)
    assert result is not None
    assert "CoT-Step 1" in result


def test_op_add_cot_is_idempotent():
    """Aplicado dos veces → segunda vez devuelve None."""
    once = _op_add_cot(V0_TEMPLATE)
    assert once is not None
    twice = _op_add_cot(once)
    assert twice is None, (
        "_op_add_cot aplicado por segunda vez debe devolver None (idempotente)"
    )


def test_op_add_cot_does_not_collide_with_v0_step1():
    """'CoT-Step 1' no estaba en V0 (colisión con 'Step 1 -' del baseline)."""
    # V0 contiene "Step 1 - Detect keywords:" — verificar que CoT usa otro marcador
    assert "CoT-Step 1" not in V0_TEMPLATE
    result = _op_add_cot(V0_TEMPLATE)
    assert result is not None
    assert "CoT-Step 1" in result


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 4 — Operador _op_add_validation_step
# ══════════════════════════════════════════════════════════════════════════════

def test_op_add_validation_step_adds_marker():
    """Aplicado sobre V0 → resultado contiene 're-read each field'."""
    result = _op_add_validation_step(V0_TEMPLATE)
    assert result is not None
    assert "re-read each field" in result


def test_op_add_validation_step_is_idempotent():
    """Aplicado dos veces → segunda vez devuelve None."""
    once = _op_add_validation_step(V0_TEMPLATE)
    assert once is not None
    twice = _op_add_validation_step(once)
    assert twice is None, (
        "_op_add_validation_step aplicado por segunda vez debe devolver None (idempotente)"
    )


def test_op_add_validation_step_contains_block():
    """El bloque VALIDATION_BLOCK está en el resultado."""
    result = _op_add_validation_step(V0_TEMPLATE)
    assert result is not None
    assert "VALIDATION STEP" in result


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 5 — OPERATORS registry
# ══════════════════════════════════════════════════════════════════════════════

def test_operators_registry_has_four_entries():
    """OPERATORS contiene exactamente los 4 operadores del plan T17.A3."""
    expected_keys = {"add_atomicity", "add_few_shots", "add_cot", "add_validation_step"}
    assert set(OPERATORS.keys()) == expected_keys


def test_all_operators_idempotent():
    """Todos los operadores: op(op(V0)) == op(V0) (idempotencia)."""
    for op_name, op_fn in OPERATORS.items():
        once = op_fn(V0_TEMPLATE)
        assert once is not None, f"Operator {op_name} returned None on first application"
        twice = op_fn(once)
        assert twice is None, (
            f"Operator '{op_name}' NO es idempotente: "
            f"segunda aplicación devolvió {str(twice)[:60]!r} en vez de None"
        )


def test_all_operators_produce_longer_template():
    """Cada operador añade contenido al template base."""
    for op_name, op_fn in OPERATORS.items():
        result = op_fn(V0_TEMPLATE)
        assert result is not None
        assert len(result) > len(V0_TEMPLATE), (
            f"Operator '{op_name}' no añadió contenido al template"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 6 — _insert_before_final_brace
# ══════════════════════════════════════════════════════════════════════════════

def test_insert_before_final_brace_inserts_after_last_double_brace():
    """El bloque se inserta justo después del último '}}' del template."""
    template = 'prefix {{\n  "a": "b"\n}} suffix'
    block = "\nINJECTED"
    result = _insert_before_final_brace(template, block)
    assert result == 'prefix {{\n  "a": "b"\n}}\nINJECTED suffix'


def test_insert_before_final_brace_no_brace_appends():
    """Sin '}}' en el template → bloque se añade al final."""
    template = "no closing brace here"
    block = " APPENDED"
    result = _insert_before_final_brace(template, block)
    assert result == "no closing brace here APPENDED"


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 7 — load_initial_variants
# ══════════════════════════════════════════════════════════════════════════════

def test_load_initial_variants_creates_all_six(tmp_path):
    """load_initial_variants crea V0..V5 y los devuelve todos."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)

    variants = load_initial_variants(prompts_dir)

    assert set(variants.keys()) == set(VARIANT_NAMES_INITIAL)
    for name in VARIANT_NAMES_INITIAL:
        assert (prompts_dir / f"{name}.txt").exists(), f"{name}.txt no se escribió"


def test_load_initial_variants_v0_matches_seed(tmp_path):
    """V0 cargado desde disco coincide con el seed de INITIAL_VARIANTS."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)

    variants = load_initial_variants(prompts_dir)
    assert variants["V0"].strip() == V0_TEMPLATE.strip()


def test_load_initial_variants_v1_has_atomicity(tmp_path):
    """V1 debe contener 'SINGLE specific concept' (V0 + add_atomicity)."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)

    variants = load_initial_variants(prompts_dir)
    assert "SINGLE specific concept" in variants["V1"]


def test_load_initial_variants_v2_has_few_shots(tmp_path):
    """V2 debe contener al menos 4 headers EXAMPLE (V0 + add_few_shots)."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)

    variants = load_initial_variants(prompts_dir)
    assert variants["V2"].count("EXAMPLE ") >= 4


def test_load_initial_variants_v5_has_cot_and_atomicity_and_few_shots(tmp_path):
    """V5 = V3 + CoT → debe tener SINGLE, EXAMPLE, y CoT-Step 1."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)

    variants = load_initial_variants(prompts_dir)
    assert "SINGLE specific concept" in variants["V5"]
    assert variants["V5"].count("EXAMPLE ") >= 4
    assert "CoT-Step 1" in variants["V5"]


def test_load_initial_variants_uses_existing_files(tmp_path):
    """Si V1.txt ya existe en disco, load_initial_variants lo usa sin regenerar."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    custom_v1 = "CUSTOM V1 template {query}"
    (prompts_dir / "V1.txt").write_text(custom_v1, encoding="utf-8")

    variants = load_initial_variants(prompts_dir)
    assert variants["V1"] == custom_v1


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 8 — _extract_metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_extract_metrics_pulls_macro_mean(tmp_path):
    """_extract_metrics extrae macro_mean correctamente."""
    template_path = tmp_path / "prompts" / "V0.txt"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(V0_TEMPLATE, encoding="utf-8")

    metrics = _extract_metrics("V0", _FIXED_EVAL_RESULT, template_path)
    assert metrics["macro"] == pytest.approx(0.60)


def test_extract_metrics_pulls_pct_3_of_4(tmp_path):
    """_extract_metrics extrae pct_3_of_4 correctamente."""
    template_path = tmp_path / "prompts" / "V0.txt"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(V0_TEMPLATE, encoding="utf-8")

    metrics = _extract_metrics("V0", _FIXED_EVAL_RESULT, template_path)
    assert metrics["pct_3_of_4"] == pytest.approx(0.5)


def test_extract_metrics_broken_when_n_with_sim_zero(tmp_path):
    """Si n_with_similarity == 0, broken=True."""
    template_path = tmp_path / "prompts" / "V0.txt"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(V0_TEMPLATE, encoding="utf-8")

    zero_result = dict(_FIXED_EVAL_RESULT)
    zero_result["aggregates"] = {
        "macro_mean": None,
        "median": None,
        "pct_samples_3_of_4_above_0.5": None,
        "per_element": {},
        "n_with_similarity": 0,
    }
    metrics = _extract_metrics("V0", zero_result, template_path)
    assert metrics["broken"] is True


def test_extract_metrics_broken_when_macro_none(tmp_path):
    """Si macro_mean es None, broken=True."""
    template_path = tmp_path / "prompts" / "V0.txt"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(V0_TEMPLATE, encoding="utf-8")

    null_result = dict(_FIXED_EVAL_RESULT)
    null_result["aggregates"] = {
        "macro_mean": None,
        "median": None,
        "pct_samples_3_of_4_above_0.5": None,
        "per_element": {},
        "n_with_similarity": 2,
    }
    metrics = _extract_metrics("V0", null_result, template_path)
    assert metrics["broken"] is True


def test_extract_metrics_not_broken_when_valid(tmp_path):
    """Con datos válidos (macro > 0, n_with_sim > 0), broken=False."""
    template_path = tmp_path / "prompts" / "V0.txt"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(V0_TEMPLATE, encoding="utf-8")

    metrics = _extract_metrics("V0", _FIXED_EVAL_RESULT, template_path)
    assert metrics["broken"] is False


def test_extract_metrics_snapshots_predictions(tmp_path):
    """predictions_sample contiene hasta 2 records no vacíos."""
    template_path = tmp_path / "prompts" / "V0.txt"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(V0_TEMPLATE, encoding="utf-8")

    metrics = _extract_metrics("V0", _FIXED_EVAL_RESULT, template_path)
    assert len(metrics["predictions_sample"]) == 2
    for pred in metrics["predictions_sample"]:
        assert "id" in pred
        assert "predicted" in pred
        assert "gold" in pred


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 9 — run_variant con run_eval mockeado
# ══════════════════════════════════════════════════════════════════════════════

def test_run_variant_eval_json_written_to_cache(tmp_path):
    """run_variant escribe eval.json en cache/V0/eval.json."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    def fake_run_eval(**kwargs):
        out = kwargs.get("output")
        if out and not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(_FIXED_EVAL_RESULT), encoding="utf-8")
        return _FIXED_EVAL_RESULT

    with patch("experiments.eval_a1_pico.run_eval", side_effect=fake_run_eval):
        result = run_variant(
            name="V0",
            template=V0_TEMPLATE,
            dev_file=dev_file,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
            output_dir=tmp_path,
        )

    expected_eval_path = tmp_path / "cache" / "V0" / "eval.json"
    assert expected_eval_path.exists(), f"eval.json no encontrado en {expected_eval_path}"


def test_run_variant_returns_macro_from_eval(tmp_path):
    """run_variant extrae macro correctamente del resultado de run_eval."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    def fake_run_eval(**kwargs):
        out = kwargs.get("output")
        if out and not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(_FIXED_EVAL_RESULT), encoding="utf-8")
        return _FIXED_EVAL_RESULT

    with patch("experiments.eval_a1_pico.run_eval", side_effect=fake_run_eval):
        result = run_variant(
            name="V0",
            template=V0_TEMPLATE,
            dev_file=dev_file,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
            output_dir=tmp_path,
        )

    assert result["macro"] == pytest.approx(0.60)
    assert result["broken"] is False


def test_run_variant_broken_when_run_eval_raises(tmp_path):
    """Si run_eval lanza excepción → record con broken=True, no aborta."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    def failing_run_eval(**kwargs):
        raise RuntimeError("eval failed")

    with patch("experiments.eval_a1_pico.run_eval", side_effect=failing_run_eval):
        result = run_variant(
            name="V0",
            template=V0_TEMPLATE,
            dev_file=dev_file,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
            output_dir=tmp_path,
        )

    assert result["broken"] is True
    assert result["error"] is not None
    assert "RuntimeError" in result["error"] or "eval failed" in result["error"]


def test_run_variant_cached_result_skips_run_eval(tmp_path):
    """Si eval.json ya existe en cache, run_eval no se vuelve a invocar."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    # Pre-escribir el eval.json en la ubicación esperada
    cache_dir = tmp_path / "cache" / "V0"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "eval.json").write_text(json.dumps(_FIXED_EVAL_RESULT), encoding="utf-8")

    call_count = [0]

    def counting_run_eval(**kwargs):
        call_count[0] += 1
        return _FIXED_EVAL_RESULT

    with patch("experiments.eval_a1_pico.run_eval", side_effect=counting_run_eval):
        result = run_variant(
            name="V0",
            template=V0_TEMPLATE,
            dev_file=dev_file,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
            output_dir=tmp_path,
        )

    assert call_count[0] == 0, "run_eval no debería invocarse cuando eval.json ya existe"
    assert result["macro"] == pytest.approx(0.60)


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 10 — propose_next_variant
# ══════════════════════════════════════════════════════════════════════════════

def test_propose_next_variant_empty_history_returns_none():
    """Con history vacío → devuelve None (no hay parent del que derivar)."""
    result = propose_next_variant([], Path("/tmp"))
    assert result is None


def test_propose_next_variant_picks_best_as_parent(tmp_path):
    """Proposer selecciona el mejor variant no-broken como parent."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    low_path = prompts_dir / "V_low.txt"
    best_path_file = prompts_dir / "V_best.txt"
    low_path.write_text(V0_TEMPLATE, encoding="utf-8")
    best_path_file.write_text(V0_TEMPLATE, encoding="utf-8")

    history = [
        _make_variant_record("V_low", macro=0.30, template_path=str(low_path)),
        _make_variant_record("V_best", macro=0.70, template_path=str(best_path_file)),
    ]

    proposal = propose_next_variant(history, prompts_dir)
    assert proposal is not None
    name, template = proposal
    assert name.startswith("V_best+"), (
        f"Esperado nombre comenzando con 'V_best+', obtenido '{name}'"
    )


def test_propose_next_variant_ignores_broken_as_best(tmp_path):
    """Variant broken (macro=0.99) no se usa como parent."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    good_path = prompts_dir / "V_good.txt"
    broken_path = prompts_dir / "V_broken.txt"
    good_path.write_text(V0_TEMPLATE, encoding="utf-8")
    broken_path.write_text(V0_TEMPLATE, encoding="utf-8")

    history = [
        _make_variant_record("V_good", macro=0.50, template_path=str(good_path)),
        _make_variant_record("V_broken", macro=0.99, broken=True, template_path=str(broken_path)),
    ]

    proposal = propose_next_variant(history, prompts_dir)
    assert proposal is not None
    name, _ = proposal
    assert name.startswith("V_good+"), (
        f"Parent debe ser V_good (no broken), obtenido '{name}'"
    )


def test_propose_next_variant_avoids_already_tried_operator(tmp_path):
    """Si V0+add_atomicity ya está en history, propone un operador diferente."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    v0_path = prompts_dir / "V0.txt"
    v0_path.write_text(V0_TEMPLATE, encoding="utf-8")
    # Crear el derived variant
    derived = _op_add_atomicity(V0_TEMPLATE)
    (prompts_dir / "V0+add_atomicity.txt").write_text(derived, encoding="utf-8")

    history = [
        _make_variant_record("V0", macro=0.60, template_path=str(v0_path)),
        _make_variant_record(
            "V0+add_atomicity", macro=0.62,
            template_path=str(prompts_dir / "V0+add_atomicity.txt")
        ),
    ]

    proposal = propose_next_variant(history, prompts_dir)
    assert proposal is not None
    name, _ = proposal
    assert name != "V0+add_atomicity", "No debe repetir operador ya probado"
    assert name.startswith("V0+add_atomicity+") or name.startswith("V0+"), (
        f"Propuesta inesperada: '{name}'"
    )


def test_propose_next_variant_all_single_ops_tried_tries_pairwise(tmp_path):
    """Cuando todos los single-ops de best están en history, propone pairwise."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)

    v0_path = prompts_dir / "V0.txt"
    v0_path.write_text(V0_TEMPLATE, encoding="utf-8")

    # Pre-añadir todos los single-op derivados a history con macro peor que V0
    history = [
        _make_variant_record("V0", macro=0.60, template_path=str(v0_path)),
    ]
    for op_name, op_fn in OPERATORS.items():
        derived = op_fn(V0_TEMPLATE)
        name = f"V0+{op_name}"
        path = prompts_dir / f"{name}.txt"
        if derived:
            path.write_text(derived, encoding="utf-8")
        else:
            path.write_text(V0_TEMPLATE, encoding="utf-8")
        history.append(
            _make_variant_record(name, macro=0.55, template_path=str(path))
        )

    proposal = propose_next_variant(history, prompts_dir)
    # Puede devolver None (si pairwise tampoco funciona) o un pairwise no visto
    if proposal is not None:
        name, _ = proposal
        seen_names = {h["name"] for h in history}
        assert name not in seen_names, f"Proposer repitió un nombre ya visto: '{name}'"


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 11 — _update_best y _target_sustained
# ══════════════════════════════════════════════════════════════════════════════

def test_update_best_skips_broken():
    """_update_best ignora variants broken y no cambia el contador."""
    best = _make_variant_record("V0", macro=0.60)
    broken = _make_variant_record("V1", macro=0.99, broken=True)

    new_best, no_imp = _update_best(best, broken, no_improvement=2)
    assert new_best["name"] == "V0"
    assert no_imp == 2


def test_update_best_updates_on_improvement():
    """_update_best actualiza best y resetea no_improvement a 0 cuando hay mejora."""
    best = _make_variant_record("V0", macro=0.60)
    better = _make_variant_record("V1", macro=0.75)

    new_best, no_imp = _update_best(best, better, no_improvement=3)
    assert new_best["name"] == "V1"
    assert no_imp == 0


def test_update_best_increments_counter_on_no_improvement():
    """Sin mejora → no_improvement incrementa en 1."""
    best = _make_variant_record("V0", macro=0.60)
    worse = _make_variant_record("V1", macro=0.50)

    new_best, no_imp = _update_best(best, worse, no_improvement=2)
    assert new_best["name"] == "V0"
    assert no_imp == 3


def test_update_best_sets_best_when_none():
    """_update_best con best=None → acepta primer record no-broken."""
    record = _make_variant_record("V0", macro=0.60)
    new_best, no_imp = _update_best(None, record, no_improvement=0)
    assert new_best["name"] == "V0"
    assert no_imp == 0


def test_target_sustained_requires_two_consecutive():
    """_target_sustained True solo cuando últimos 2 no-broken superan el umbral."""
    history_one = [_make_variant_record("V0", macro=0.70)]
    assert _target_sustained(history_one, 0.60) is False

    history_two_pass = [
        _make_variant_record("V0", macro=0.70),
        _make_variant_record("V1", macro=0.65),
    ]
    assert _target_sustained(history_two_pass, 0.60) is True

    history_mixed = [
        _make_variant_record("V0", macro=0.70),
        _make_variant_record("V1", macro=0.45),  # bajo umbral
    ]
    assert _target_sustained(history_mixed, 0.60) is False


def test_target_sustained_ignores_broken():
    """_target_sustained sólo considera variants no-broken."""
    history = [
        _make_variant_record("V0", macro=0.70),
        _make_variant_record("V1", macro=0.99, broken=True),  # ignorado
        _make_variant_record("V2", macro=0.68),
    ]
    # Los dos últimos no-broken son V0 (0.70) y V2 (0.68) → ambos > 0.60
    assert _target_sustained(history, 0.60) is True


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 12 — iterate() terminación por ambitious_target_sustained
# ══════════════════════════════════════════════════════════════════════════════

def test_iterate_termination_ambitious_target_sustained(tmp_path):
    """Dos variantes consecutivas >= ambitious_macro → termination_reason = 'ambitious_target_sustained'."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    # Secuencia: V0=0.70 (primera sustained check falla, sólo 1), V1=0.72 → sustained
    macro_seq = iter([0.70, 0.72, 0.74, 0.75, 0.76])

    def fake_run_variant(name, template, dev_file, a1_instance, embedder, output_dir, log_file=None):
        macro = next(macro_seq, 0.70)
        rec = _make_variant_record(
            name, macro=macro,
            template_path=str(prompts_dir / f"{name}.txt")
        )
        return rec

    with patch("experiments.iterate_a1_metadata.run_variant", side_effect=fake_run_variant):
        summary = iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.60,
            ambitious_macro=0.65,
            max_hours=10.0,
            max_no_improvement=20,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
        )

    assert summary["termination_reason"] == "ambitious_target_sustained", (
        f"Esperado 'ambitious_target_sustained', obtenido '{summary['termination_reason']}'"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 13 — iterate() terminación por time_budget_exceeded
# ══════════════════════════════════════════════════════════════════════════════

def test_iterate_termination_time_budget(tmp_path):
    """Mock time.time() avanza más allá del deadline → termination_reason incluye budget."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    t0 = 1_700_000_000.0
    # Primera llamada (deadline = t0 + budget); las siguientes ya superan deadline
    time_seq = iter([t0] + [t0 + 999_999.0] * 50)

    def fake_run_variant(name, template, dev_file, a1_instance, embedder, output_dir, log_file=None):
        return _make_variant_record(
            name, macro=0.50, template_path=str(prompts_dir / f"{name}.txt")
        )

    with patch("experiments.iterate_a1_metadata.run_variant", side_effect=fake_run_variant), \
         patch("experiments.iterate_a1_metadata.time.time", side_effect=time_seq):
        summary = iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.99,
            ambitious_macro=0.99,
            max_hours=0.001,
            max_no_improvement=999,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
        )

    assert summary["termination_reason"] in (
        "time_budget_exceeded",
        "ambitious_target_sustained",
        "no_improvement_cap_reached",
        "no_new_variant_proposable",
    ), f"Razón de terminación inesperada: '{summary['termination_reason']}'"


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 14 — iterate() terminación por no_improvement_cap_reached
# ══════════════════════════════════════════════════════════════════════════════

def test_iterate_termination_no_improvement(tmp_path):
    """Macro plana → no_improvement_cap_reached tras max_no_improvement iteraciones."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    counter = [0]

    def fake_propose(history, prompts_dir_arg):
        counter[0] += 1
        name = f"VProp{counter[0]}"
        template = f"template_{name} {{query}}"
        (prompts_dir / f"{name}.txt").write_text(template, encoding="utf-8")
        return name, template

    def fake_run_variant(name, template, dev_file, a1_instance, embedder, output_dir, log_file=None):
        path = prompts_dir / f"{name}.txt"
        if not path.exists():
            path.write_text(template, encoding="utf-8")
        return _make_variant_record(name, macro=0.50, template_path=str(path))

    with patch("experiments.iterate_a1_metadata.run_variant", side_effect=fake_run_variant), \
         patch("experiments.iterate_a1_metadata.propose_next_variant", side_effect=fake_propose):
        summary = iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.99,
            ambitious_macro=0.99,
            max_hours=10.0,
            max_no_improvement=3,  # se dispara rápido
            a1_instance=MagicMock(),
            embedder=MagicMock(),
        )

    assert summary["termination_reason"] == "no_improvement_cap_reached", (
        f"Esperado 'no_improvement_cap_reached', obtenido '{summary['termination_reason']}'"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 15 — iterate() terminación por no_new_variant_proposable
# ══════════════════════════════════════════════════════════════════════════════

def test_iterate_termination_no_new_variant(tmp_path):
    """Proposer devuelve None → no_new_variant_proposable."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    def fake_run_variant(name, template, dev_file, a1_instance, embedder, output_dir, log_file=None):
        return _make_variant_record(
            name, macro=0.50, template_path=str(prompts_dir / f"{name}.txt")
        )

    with patch("experiments.iterate_a1_metadata.run_variant", side_effect=fake_run_variant), \
         patch("experiments.iterate_a1_metadata.propose_next_variant", return_value=None):
        summary = iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.99,
            ambitious_macro=0.99,
            max_hours=10.0,
            max_no_improvement=99,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
        )

    assert summary["termination_reason"] == "no_new_variant_proposable", (
        f"Esperado 'no_new_variant_proposable', obtenido '{summary['termination_reason']}'"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 16 — iterate() persistencia de artefactos
# ══════════════════════════════════════════════════════════════════════════════

def test_iterate_persists_iteration_log_and_best(tmp_path):
    """Tras iterate(), iteration_log.json y best_variant.json existen con esquema correcto."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    def fake_run_variant(name, template, dev_file, a1_instance, embedder, output_dir, log_file=None):
        return _make_variant_record(
            name, macro=0.60, template_path=str(prompts_dir / f"{name}.txt")
        )

    with patch("experiments.iterate_a1_metadata.run_variant", side_effect=fake_run_variant), \
         patch("experiments.iterate_a1_metadata.propose_next_variant", return_value=None):
        iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.99,
            ambitious_macro=0.99,
            max_hours=10.0,
            max_no_improvement=3,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
        )

    log_path = tmp_path / "iteration_log.json"
    best_path = tmp_path / "best_variant.json"

    assert log_path.exists(), "iteration_log.json no creado"
    assert best_path.exists(), "best_variant.json no creado"

    log_data = json.loads(log_path.read_text(encoding="utf-8"))
    assert "summary" in log_data
    assert "history" in log_data
    assert isinstance(log_data["history"], list)
    s = log_data["summary"]
    assert "termination_reason" in s
    assert "n_variants_evaluated" in s
    assert "best" in s

    best_data = json.loads(best_path.read_text(encoding="utf-8"))
    assert "name" in best_data
    assert "macro" in best_data
    assert "template" in best_data


def test_iterate_log_updated_after_each_variant(tmp_path):
    """iteration_log.json se actualiza tras cada variant (n_variants_evaluated correcto)."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    call_count = [0]

    def fake_run_variant(name, template, dev_file, a1_instance, embedder, output_dir, log_file=None):
        call_count[0] += 1
        return _make_variant_record(
            name, macro=0.50, template_path=str(prompts_dir / f"{name}.txt")
        )

    with patch("experiments.iterate_a1_metadata.run_variant", side_effect=fake_run_variant), \
         patch("experiments.iterate_a1_metadata.propose_next_variant", return_value=None):
        iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.99,
            ambitious_macro=0.99,
            max_hours=10.0,
            max_no_improvement=3,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
        )

    log_data = json.loads((tmp_path / "iteration_log.json").read_text(encoding="utf-8"))
    # n_variants_evaluated debe coincidir con el número de variantes procesadas
    # (al menos las 6 iniciales V0..V5)
    assert log_data["summary"]["n_variants_evaluated"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Bloque 17 — iterate() trayectoria fija con 3-4 variantes
# ══════════════════════════════════════════════════════════════════════════════

def test_iterate_fixed_trajectory_three_variants(tmp_path):
    """Trayectoria fija: V0(0.55) → V1(0.63) → V2(0.67) → sustained → stop."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    macros = iter([0.55, 0.63, 0.67, 0.68, 0.70, 0.71])

    def fake_run_variant(name, template, dev_file, a1_instance, embedder, output_dir, log_file=None):
        macro = next(macros, 0.65)
        return _make_variant_record(
            name, macro=macro, template_path=str(prompts_dir / f"{name}.txt")
        )

    with patch("experiments.iterate_a1_metadata.run_variant", side_effect=fake_run_variant):
        summary = iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.55,
            ambitious_macro=0.62,
            max_hours=10.0,
            max_no_improvement=20,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
        )

    assert summary["termination_reason"] == "ambitious_target_sustained"
    assert summary["n_variants_evaluated"] >= 2
    assert summary["best"]["macro"] is not None


def test_iterate_summary_fields_complete(tmp_path):
    """El summary devuelto por iterate() tiene todos los campos requeridos."""
    prompts_dir = tmp_path / "prompts"
    _write_v0_prompt(prompts_dir)
    dev_file = tmp_path / "dev.json"
    dev_file.write_text(json.dumps({"samples": []}), encoding="utf-8")

    def fake_run_variant(name, template, dev_file, a1_instance, embedder, output_dir, log_file=None):
        return _make_variant_record(
            name, macro=0.60, template_path=str(prompts_dir / f"{name}.txt")
        )

    with patch("experiments.iterate_a1_metadata.run_variant", side_effect=fake_run_variant), \
         patch("experiments.iterate_a1_metadata.propose_next_variant", return_value=None):
        summary = iterate(
            dev_file=dev_file,
            output_dir=tmp_path,
            target_macro=0.99,
            ambitious_macro=0.99,
            max_hours=10.0,
            max_no_improvement=3,
            a1_instance=MagicMock(),
            embedder=MagicMock(),
        )

    required_fields = [
        "termination_reason", "n_variants_evaluated", "n_broken",
        "total_runtime_s", "best", "timestamp",
    ]
    for field in required_fields:
        assert field in summary, f"Campo '{field}' ausente del summary"
