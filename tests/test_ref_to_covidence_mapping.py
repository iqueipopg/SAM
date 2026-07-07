# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests de validacion para el output de T16.D.4.A.

Valida el CSV generado por scripts/build_ref_to_covidence_mapping.py:
  data/gold_standard/deporte/unified/ref_to_covidence_mapping.csv

NO re-ejecuta el script de generacion. Lee el CSV y el xlsx ya presentes.
Si el CSV no existe, los tests son omitidos con pytest.skip.

Todos los tests son sin GPU, sin red. Tiempo esperado del modulo: < 5 s.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Rutas canonicas
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING_CSV = (
    PROJECT_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "unified"
    / "ref_to_covidence_mapping.csv"
)
XLSX_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "Resumen_Total_PDFs.xlsx"
)

EXPECTED_COLUMNS = [
    "ref_number",
    "covidence_id",
    "confidence",
    "study_label",
    "match_method",
    "notes",
]

VALID_CONFIDENCE = {"high", "medium", "low", "unresolved"}

REF_DOMAIN_MIN = 1
REF_DOMAIN_MAX = 73

# Verificaciones manuales del usuario (ver spec T16.D.4.A).
# match_method acepta tanto exact_author_year como exact_author_year_title_tiebreaker
# porque en los datos reales Keyserling/Bennett no colisionan por year (2002 vs 2008,
# 2007 vs 2008) y el matcher devuelve exact_author_year directamente.
EXPECTED_HIGH_CONFIDENCE_MATCHES = [
    (1,  "#10313", "Hartman"),
    (26, "#2098",  "Keyserling"),
    (44, "#4076",  "Keyserling"),
    (41, "#3490",  "Bennett"),
    (42, "#3558",  "Bennett"),
]

# Grupos de multi-ref con covidence_id confirmado por inspeccion del script.
# Para grupos con expected_cov=None solo verificamos igualdad entre miembros del grupo.
MULTIREF_GROUPS = [
    ([3, 4, 5],   "#10657"),  # Thomsen
    ([12, 13],    "#12417"),  # Balducci
    ([36, 37],    "#30369"),  # Hochsmann
    ([46, 47],    None),       # Costanzo — igualdad verificada, cov_id no hardcodeado
    ([49, 50],    None),       # Van Keulen — igualdad verificada, cov_id no hardcodeado
    ([66, 67],    None),       # James — igualdad verificada, cov_id no hardcodeado
]


# ---------------------------------------------------------------------------
# Fixtures de modulo (cargados una sola vez)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mapping_df() -> pd.DataFrame:
    """Carga el CSV de mapping. Salta todos los tests si no existe."""
    if not MAPPING_CSV.exists():
        pytest.skip(
            f"CSV no encontrado: {MAPPING_CSV}. "
            "Ejecutar scripts/build_ref_to_covidence_mapping.py primero."
        )
    return pd.read_csv(str(MAPPING_CSV))


@pytest.fixture(scope="module")
def xlsx_df() -> pd.DataFrame:
    """Carga el xlsx fuente. Salta si no existe."""
    if not XLSX_PATH.exists():
        pytest.skip(
            f"xlsx no encontrado: {XLSX_PATH}. "
            "El dataset debe estar presente en data/gold_standard/deporte/."
        )
    return pd.read_excel(str(XLSX_PATH))


# ---------------------------------------------------------------------------
# Test 1: schema del CSV
# ---------------------------------------------------------------------------


def test_schema(mapping_df: pd.DataFrame):
    """El CSV existe, tiene exactamente 6 columnas en el orden correcto y dtypes razonables."""
    assert MAPPING_CSV.exists(), f"CSV no encontrado: {MAPPING_CSV}"
    assert isinstance(mapping_df, pd.DataFrame)

    # Columnas en orden exacto
    assert list(mapping_df.columns) == EXPECTED_COLUMNS, (
        f"Columnas incorrectas o fuera de orden. "
        f"Esperado: {EXPECTED_COLUMNS}. "
        f"Obtenido: {list(mapping_df.columns)}"
    )

    # ref_number es int
    assert pd.api.types.is_integer_dtype(mapping_df["ref_number"]), (
        f"ref_number debe ser dtype entero, es: {mapping_df['ref_number'].dtype}"
    )

    # covidence_id es object (string con NaN permitido)
    assert mapping_df["covidence_id"].dtype == object, (
        f"covidence_id debe ser dtype object, es: {mapping_df['covidence_id'].dtype}"
    )

    # confidence, study_label, match_method son object string
    for col in ("confidence", "study_label", "match_method"):
        assert mapping_df[col].dtype == object, (
            f"'{col}' debe ser dtype object, es: {mapping_df[col].dtype}"
        )

    # notes es object con NaN permitido
    assert mapping_df["notes"].dtype == object, (
        f"notes debe ser dtype object, es: {mapping_df['notes'].dtype}"
    )


# ---------------------------------------------------------------------------
# Test 2: ref_number unico y en rango
# ---------------------------------------------------------------------------


def test_ref_number_unique_and_in_range(mapping_df: pd.DataFrame):
    """ref_number es unico, todos los valores estan en [1, 73], exactamente 73 filas."""
    assert len(mapping_df) == REF_DOMAIN_MAX, (
        f"Se esperaban {REF_DOMAIN_MAX} filas (una por ref_number). "
        f"Se obtuvieron: {len(mapping_df)}"
    )

    # Sin duplicados
    duplicates = mapping_df[mapping_df.duplicated(subset=["ref_number"])]
    assert len(duplicates) == 0, (
        f"ref_number no es unico. Duplicados encontrados: "
        f"{duplicates['ref_number'].tolist()}"
    )

    # Todos en rango [1..73]
    out_of_range = mapping_df[
        (mapping_df["ref_number"] < REF_DOMAIN_MIN)
        | (mapping_df["ref_number"] > REF_DOMAIN_MAX)
    ]
    assert len(out_of_range) == 0, (
        f"Valores de ref_number fuera de [{REF_DOMAIN_MIN}, {REF_DOMAIN_MAX}]: "
        f"{out_of_range['ref_number'].tolist()}"
    )

    # Cobertura completa del rango 1..73
    expected_refs = set(range(REF_DOMAIN_MIN, REF_DOMAIN_MAX + 1))
    actual_refs = set(mapping_df["ref_number"].tolist())
    missing = expected_refs - actual_refs
    assert not missing, (
        f"Refs ausentes en el CSV (cobertura incompleta): {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Test 3: valores validos de confidence
# ---------------------------------------------------------------------------


def test_confidence_values_valid(mapping_df: pd.DataFrame):
    """Cada valor de confidence pertenece a {high, medium, low, unresolved}."""
    actual_values = set(mapping_df["confidence"].dropna().unique())
    unexpected = actual_values - VALID_CONFIDENCE
    assert not unexpected, (
        f"Valores de confidence fuera del conjunto valido: {unexpected}. "
        f"Solo se permiten: {VALID_CONFIDENCE}"
    )

    # Sin nulos en confidence
    null_count = mapping_df["confidence"].isna().sum()
    assert null_count == 0, (
        f"{null_count} fila(s) con confidence nulo — todos los registros deben "
        "tener un nivel de confidence asignado."
    )


# ---------------------------------------------------------------------------
# Test 4: formato de covidence_id
# ---------------------------------------------------------------------------


def test_covidence_id_format(mapping_df: pd.DataFrame):
    """Valida el formato de covidence_id segun el nivel de confidence.

    - Cuando covidence_id no es NaN: formato ^#\\d+$
    - Cuando confidence=="unresolved": covidence_id debe ser NaN
    - Cuando confidence in {"high","medium","low"}: covidence_id NO debe ser NaN
    """
    cov_id_re = re.compile(r"^#\d+$")

    non_null_mask = mapping_df["covidence_id"].notna()

    # Filas con covidence_id presente: deben seguir formato #NNN
    bad_format = mapping_df.loc[
        non_null_mask,
        "covidence_id",
    ].apply(lambda v: not bool(cov_id_re.match(str(v))))
    bad_values = mapping_df.loc[non_null_mask][bad_format]["covidence_id"].tolist()
    assert not bad_values, (
        f"covidence_id con formato invalido (esperado ^#\\d+$): {bad_values}"
    )

    # Filas unresolved: covidence_id debe ser NaN
    unresolved_mask = mapping_df["confidence"] == "unresolved"
    unresolved_with_cov = mapping_df.loc[
        unresolved_mask & non_null_mask
    ]
    assert len(unresolved_with_cov) == 0, (
        f"{len(unresolved_with_cov)} fila(s) con confidence=unresolved tienen "
        f"covidence_id no-nulo: "
        f"{unresolved_with_cov[['ref_number','covidence_id']].to_dict('records')}"
    )

    # Filas high/medium/low: covidence_id NO debe ser NaN
    resolved_mask = mapping_df["confidence"].isin({"high", "medium", "low"})
    resolved_without_cov = mapping_df.loc[resolved_mask & ~non_null_mask]
    assert len(resolved_without_cov) == 0, (
        f"{len(resolved_without_cov)} fila(s) con confidence high/medium/low "
        f"tienen covidence_id nulo: "
        f"{resolved_without_cov[['ref_number','confidence']].to_dict('records')}"
    )


# ---------------------------------------------------------------------------
# Test 5: cobertura de Incluidos
# ---------------------------------------------------------------------------


def test_coverage_includes(mapping_df: pd.DataFrame, xlsx_df: pd.DataFrame):
    """Al menos el 80 % de los 66 covidence_ids Incluido aparecen en el mapping
    con confidence != unresolved.

    El spec del usuario explicita no forzar 100 %. La cobertura actual es 95.5 %
    (63/66), que supera el umbral ampliamente.
    """
    incluido = xlsx_df[xlsx_df["Clasificacion"] if "Clasificacion" in xlsx_df.columns
                       else xlsx_df["Clasificación"] == "Incluido"].copy()

    # Manejar ambas formas del nombre de columna (con/sin tilde)
    col_clasif = "Clasificación" if "Clasificación" in xlsx_df.columns else "Clasificacion"
    col_cov = (
        "ID del Estudio (Covidence)"
        if "ID del Estudio (Covidence)" in xlsx_df.columns
        else "covidence_id"
    )

    incluido_filtered = xlsx_df[xlsx_df[col_clasif] == "Incluido"]
    incluido_cov_ids = set(
        incluido_filtered[col_cov].dropna().astype(str).unique()
    )

    assert len(incluido_cov_ids) > 0, (
        "No se encontraron covidence_ids con Clasificacion==Incluido en el xlsx. "
        "Verificar nombre de columna."
    )

    resolved_cov_ids = set(
        mapping_df.loc[
            mapping_df["confidence"] != "unresolved",
            "covidence_id",
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    overlap = incluido_cov_ids & resolved_cov_ids
    ratio = len(overlap) / len(incluido_cov_ids)

    assert ratio >= 0.80, (
        f"Cobertura de Incluidos insuficiente: {len(overlap)}/{len(incluido_cov_ids)} "
        f"= {ratio:.1%}. Umbral minimo: 80 %. "
        f"covidence_ids sin resolver: {sorted(incluido_cov_ids - resolved_cov_ids)}"
    )


# ---------------------------------------------------------------------------
# Test 6: filas high/medium/low tienen match_method
# ---------------------------------------------------------------------------


def test_high_matches_have_match_method(mapping_df: pd.DataFrame):
    """Toda fila con confidence in {high, medium, low} tiene match_method no vacio."""
    resolved = mapping_df[mapping_df["confidence"].isin({"high", "medium", "low"})]

    missing_method = resolved[
        resolved["match_method"].isna() | (resolved["match_method"].str.strip() == "")
    ]
    assert len(missing_method) == 0, (
        f"{len(missing_method)} fila(s) con confidence high/medium/low tienen "
        f"match_method vacio o nulo: "
        f"{missing_method[['ref_number','confidence','match_method']].to_dict('records')}"
    )


# ---------------------------------------------------------------------------
# Test 7: filas unresolved tienen razon concreta
# ---------------------------------------------------------------------------


def test_unresolved_have_reason(mapping_df: pd.DataFrame):
    """Toda fila con confidence==unresolved tiene match_method != NaN y notes != NaN."""
    unresolved = mapping_df[mapping_df["confidence"] == "unresolved"]

    assert len(unresolved) > 0, (
        "No hay filas unresolved en el mapping — si el dataset cambio "
        "revisar si este test sigue siendo relevante."
    )

    null_method = unresolved[
        unresolved["match_method"].isna() | (unresolved["match_method"].str.strip() == "")
    ]
    assert len(null_method) == 0, (
        f"{len(null_method)} fila(s) unresolved sin match_method: "
        f"{null_method[['ref_number','match_method']].to_dict('records')}"
    )

    null_notes = unresolved[unresolved["notes"].isna()]
    assert len(null_notes) == 0, (
        f"{len(null_notes)} fila(s) unresolved sin notes: "
        f"{null_notes[['ref_number','notes']].to_dict('records')}"
    )


# ---------------------------------------------------------------------------
# Test 8: casos de alta confianza verificados manualmente
# ---------------------------------------------------------------------------


def test_expected_high_confidence_matches(mapping_df: pd.DataFrame):
    """Verifica los 5 matches confirmados manualmente por el usuario.

    match_method acepta exact_author_year o exact_author_year_title_tiebreaker:
    en los datos reales Keyserling/Bennett no colisionan por year (2002 vs 2008,
    2007 vs 2008) y el matcher resuelve via exact_author_year directamente.
    """
    accepted_methods = {"exact_author_year", "exact_author_year_title_tiebreaker"}

    for ref_num, expected_cov_id, surname_fragment in EXPECTED_HIGH_CONFIDENCE_MATCHES:
        rows = mapping_df[mapping_df["ref_number"] == ref_num]

        assert len(rows) == 1, (
            f"ref_number={ref_num}: se esperaba exactamente 1 fila, "
            f"se encontraron {len(rows)}."
        )

        row = rows.iloc[0]

        assert row["covidence_id"] == expected_cov_id, (
            f"ref_number={ref_num}: covidence_id esperado={expected_cov_id!r}, "
            f"obtenido={row['covidence_id']!r}"
        )

        assert row["confidence"] == "high", (
            f"ref_number={ref_num}: confidence esperado='high', "
            f"obtenido={row['confidence']!r}"
        )

        study_label = str(row["study_label"]) if pd.notna(row["study_label"]) else ""
        assert surname_fragment.lower() in study_label.lower(), (
            f"ref_number={ref_num}: apellido '{surname_fragment}' no encontrado "
            f"(case-insensitive) en study_label={study_label!r}"
        )

        assert row["match_method"] in accepted_methods, (
            f"ref_number={ref_num}: match_method={row['match_method']!r} "
            f"no esta en los metodos aceptados: {accepted_methods}"
        )


# ---------------------------------------------------------------------------
# Test 9: multi-refs comparten covidence_id y study_label
# ---------------------------------------------------------------------------


def test_multirefs_share_covidence_id(mapping_df: pd.DataFrame):
    """Verifica que los grupos de multi-ref tienen covidence_id y study_label identicos.

    Para grupos con expected_cov=None solo verifica igualdad entre miembros;
    para grupos con expected_cov != None verifica ademas el valor esperado.
    No se inventan covidence_ids para grupos sin confirmacion externa.
    """
    for refs, expected_cov in MULTIREF_GROUPS:
        group_rows = mapping_df[mapping_df["ref_number"].isin(refs)]

        assert len(group_rows) == len(refs), (
            f"Grupo {refs}: se esperaban {len(refs)} filas, "
            f"se encontraron {len(group_rows)}. "
            f"refs presentes: {group_rows['ref_number'].tolist()}"
        )

        cov_ids = group_rows["covidence_id"].dropna().unique().tolist()
        assert len(cov_ids) == 1, (
            f"Grupo {refs}: los miembros tienen covidence_id distintos: {cov_ids}. "
            "Un multi-ref debe mapear a un unico estudio."
        )

        study_labels = group_rows["study_label"].dropna().unique().tolist()
        assert len(study_labels) == 1, (
            f"Grupo {refs}: los miembros tienen study_label distintos: {study_labels}. "
            "Un multi-ref debe tener el mismo study_label."
        )

        if expected_cov is not None:
            assert cov_ids[0] == expected_cov, (
                f"Grupo {refs}: covidence_id esperado={expected_cov!r}, "
                f"obtenido={cov_ids[0]!r}"
            )


# ---------------------------------------------------------------------------
# Test 10: ref_number=52 es unresolved con razon missing_in_gold_rob
# ---------------------------------------------------------------------------


def test_ref_52_missing_in_gold_rob(mapping_df: pd.DataFrame):
    """ref_number=52 esta en el output con confidence=unresolved y
    match_method=missing_in_gold_rob.

    Valida que el rango 1..73 es completo incluso cuando gold_rob no cubre
    todos los ref_numbers (ref=52 esta ausente en gold_rob).
    """
    rows = mapping_df[mapping_df["ref_number"] == 52]

    assert len(rows) == 1, (
        f"ref_number=52: se esperaba exactamente 1 fila, se encontraron {len(rows)}."
    )

    row = rows.iloc[0]

    assert row["confidence"] == "unresolved", (
        f"ref_number=52: confidence esperado='unresolved', obtenido={row['confidence']!r}"
    )

    assert row["match_method"] == "missing_in_gold_rob", (
        f"ref_number=52: match_method esperado='missing_in_gold_rob', "
        f"obtenido={row['match_method']!r}"
    )

    assert pd.isna(row["covidence_id"]), (
        f"ref_number=52: covidence_id debe ser NaN cuando confidence=unresolved, "
        f"obtenido={row['covidence_id']!r}"
    )
