# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests minimos para scripts/build_gold_standard.py (GS1).

Cubre:
  - Formato registro-maestro Covidence
  - Formato gold-standard extendido (id/title/decision)
  - Graceful failure si el CSV no existe (exit code 1)
  - Schema compatible con experiments/exp8_gold_standard_eval.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "build_gold_standard.py"


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _run_script(csv_path: Path, output_path: Path, *extra: str) -> subprocess.CompletedProcess:
    """Invoca el script como subprocess para test de integracion end-to-end."""
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--csv",
            str(csv_path),
            "--output",
            str(output_path),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def registro_csv(tmp_path: Path) -> Path:
    """CSV minimo con el formato registro maestro Covidence."""
    csv_path = tmp_path / "registro_maestro.csv"
    csv_path.write_text(
        "Covidence ID;Cita (Autores);Archivos Físicos Extrapolados;Aviso\n"
        "#12345;Smith et al. (2020);paper1.pdf | paper1_sup.pdf;Multiples PDFs\n"
        "#67890;Johnson & Lee (2021);paper2.pdf;\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def gold_extended_csv(tmp_path: Path) -> Path:
    """CSV en formato gold standard extendido."""
    csv_path = tmp_path / "gold.csv"
    csv_path.write_text(
        "id,title,abstract,authors,year,journal,decision,reason,stage\n"
        "11111,Exercise and depression,An RCT of exercise,Smith J;Jones K,2020,BMJ,INCLUDE,,title_abstract\n"
        "22222,Unrelated study,Not about depression,Doe J,2019,NEJM,EXCLUDE,off topic,title_abstract\n",
        encoding="utf-8",
    )
    return csv_path


# ══════════════════════════════════════════════════════════════════
# Test: formato registro-maestro
# ══════════════════════════════════════════════════════════════════

def test_registro_maestro_produces_valid_json(registro_csv: Path, tmp_path: Path):
    out = tmp_path / "gs.json"
    result = _run_script(registro_csv, out)

    assert result.returncode == 0, f"Script fallo: stderr={result.stderr}"
    assert out.exists(), "No se genero el JSON"

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "papers" in data
    assert data["source_format"] == "registro"
    assert len(data["papers"]) == 2

    first = data["papers"][0]
    assert first["gold_decision"] == "INCLUDE"
    assert first["stage_decided"] == "full_text"
    assert first["pmid"] == "12345"
    assert first["year"] == 2020
    assert "Smith et al" in first["authors"][0]
    assert len(first["files"]) == 2


# ══════════════════════════════════════════════════════════════════
# Test: formato gold standard extendido
# ══════════════════════════════════════════════════════════════════

def test_gold_extended_format(gold_extended_csv: Path, tmp_path: Path):
    out = tmp_path / "gs.json"
    result = _run_script(gold_extended_csv, out)

    assert result.returncode == 0, f"stderr={result.stderr}"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["source_format"] == "gold"
    assert len(data["papers"]) == 2

    decisions = {p["pmid"]: p["gold_decision"] for p in data["papers"]}
    assert decisions["11111"] == "INCLUDE"
    assert decisions["22222"] == "EXCLUDE"

    included = next(p for p in data["papers"] if p["pmid"] == "11111")
    assert included["title"] == "Exercise and depression"
    assert included["authors"] == ["Smith J", "Jones K"]
    assert included["year"] == 2020
    assert included["journal"] == "BMJ"


# ══════════════════════════════════════════════════════════════════
# Test: graceful failure si el CSV no existe
# ══════════════════════════════════════════════════════════════════

def test_missing_csv_exits_with_error(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out = tmp_path / "gs.json"
    result = _run_script(missing, out)

    assert result.returncode == 1, f"Expected exit 1, got {result.returncode}"
    combined = (result.stdout + result.stderr).lower()
    assert "no se encuentra" in combined or "does_not_exist" in combined
    assert not out.exists(), "No debe crear JSON si el CSV falta"


# ══════════════════════════════════════════════════════════════════
# Test: CSV con columnas invalidas rechazado
# ══════════════════════════════════════════════════════════════════

def test_invalid_format_rejected(tmp_path: Path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("foo,bar,baz\n1,2,3\n", encoding="utf-8")
    out = tmp_path / "gs.json"
    result = _run_script(bad_csv, out)

    assert result.returncode != 0, "Debe fallar con columnas desconocidas"
    assert not out.exists()


# ══════════════════════════════════════════════════════════════════
# Test: schema compatible con exp8_gold_standard_eval.load_gold_standard
# ══════════════════════════════════════════════════════════════════

def test_output_compatible_with_exp8(gold_extended_csv: Path, tmp_path: Path):
    """El JSON generado debe satisfacer las validaciones de exp8.load_gold_standard."""
    out = tmp_path / "gs.json"
    result = _run_script(gold_extended_csv, out)
    assert result.returncode == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    # exp8 requiere: papers no vacio, cada paper con pmid y gold_decision INCLUDE/EXCLUDE
    assert data.get("papers"), "papers vacio"
    for p in data["papers"]:
        assert "pmid" in p, f"Falta pmid en {p}"
        assert p.get("gold_decision") in ("INCLUDE", "EXCLUDE"), \
            f"gold_decision invalido: {p.get('gold_decision')}"


# ══════════════════════════════════════════════════════════════════
# Test: decision invalida en gold extendido -> error
# ══════════════════════════════════════════════════════════════════

def test_invalid_decision_value_rejected(tmp_path: Path):
    csv_path = tmp_path / "bad_decision.csv"
    csv_path.write_text(
        "id,title,decision\n"
        "1,A study,MAYBE\n",
        encoding="utf-8",
    )
    out = tmp_path / "gs.json"
    result = _run_script(csv_path, out)

    assert result.returncode != 0
    assert not out.exists()


# ══════════════════════════════════════════════════════════════════
# Test: proteccion de zona sagrada (no sobrescribe gold_standard.json)
# ══════════════════════════════════════════════════════════════════

def test_sacred_path_is_protected(registro_csv: Path, tmp_path: Path, monkeypatch):
    """Si el usuario apunta --output a data/gold_standard.json, debe abortar."""
    # Creamos una carpeta data/ dentro de tmp_path y apuntamos cwd ahi
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sacred = data_dir / "gold_standard.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--csv",
            str(registro_csv),
            "--output",
            str(sacred),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )

    assert result.returncode != 0, "Debe rechazar output en zona sagrada"
    assert not sacred.exists()
