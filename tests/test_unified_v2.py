# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the schema-v2 unified gold-standard CSV.

Validates the 13-column schema introduced in D.3.B:
  covidence_id, title, authors, abstract, doi, year,
  human_label_ta, human_label_ft, exclusion_reason_ft,
  source, notes, source_method, source_abstract

No GPU, no network, no CSV regeneration — if the CSV does not exist the
tests skip with a clear message.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the CSV
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "unified"
    / "gold_screening_unified.csv"
)

SKIP_REASON = (
    f"Gold-standard CSV not found at {CSV_PATH}. "
    "Generate it first with: python scripts/build_unified_gold_screening_v2.py"
)

EXPECTED_COLUMNS_V2 = [
    "covidence_id",
    "title",
    "authors",
    "abstract",
    "doi",
    "year",
    "human_label_ta",
    "human_label_ft",
    "exclusion_reason_ft",
    "source",
    "notes",
    "source_method",
    "source_abstract",
]

VALID_SOURCE_METHOD_VALUES = {
    "pdf_metadata",
    "doi_crossref",
    "pdfplumber_fallback",
    "crossref_by_title_low_confidence",
    "manual",
}

VALID_SOURCE_ABSTRACT_VALUES = {
    "pdf_metadata",
    "crossref",
    "pdfplumber_fallback",
    "semantic_scholar",
    "manual",
}


# ---------------------------------------------------------------------------
# Shared fixture — loaded once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df():
    """Carga el CSV unificado una vez para todos los tests del modulo."""
    if not CSV_PATH.exists():
        pytest.skip(SKIP_REASON)
    return pd.read_csv(CSV_PATH)


# ---------------------------------------------------------------------------
# a) Schema v2: shape and column order
# ---------------------------------------------------------------------------

def test_schema_v2(df):
    """El CSV debe tener exactamente 20750 filas y 13 columnas en orden exacto."""
    assert df.shape == (20750, 13), (
        f"Expected (20750, 13), got {df.shape}"
    )
    assert list(df.columns) == EXPECTED_COLUMNS_V2, (
        f"Column mismatch.\n  Expected: {EXPECTED_COLUMNS_V2}\n  Got:      {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# b) source_method allowed values
# ---------------------------------------------------------------------------

def test_source_method_values(df):
    """Los valores unicos de source_method (excl. NaN) deben estar en el conjunto permitido."""
    observed = set(df["source_method"].dropna().unique())
    unexpected = observed - VALID_SOURCE_METHOD_VALUES
    assert not unexpected, (
        f"source_method contains unexpected values: {unexpected}"
    )


# ---------------------------------------------------------------------------
# c) source_abstract allowed values
# ---------------------------------------------------------------------------

def test_source_abstract_values(df):
    """Los valores unicos de source_abstract (excl. NaN) deben estar en el conjunto permitido."""
    observed = set(df["source_abstract"].dropna().unique())
    unexpected = observed - VALID_SOURCE_ABSTRACT_VALUES
    assert not unexpected, (
        f"source_abstract contains unexpected values: {unexpected}"
    )


# ---------------------------------------------------------------------------
# d) source_method only populated on xlsx-side
# ---------------------------------------------------------------------------

def test_source_method_only_xlsx_side(df):
    """source_method debe ser nulo en todas las filas csv y no-nulo en las 1552 filas xlsx."""
    csv_all_nan = df.loc[df["source"] == "csv", "source_method"].isna().all()
    assert csv_all_nan, (
        "source_method debe ser NaN en todas las filas con source=='csv', "
        "pero se encontraron valores no nulos."
    )
    xlsx_notna = df.loc[df["source"] == "xlsx", "source_method"].notna().sum()
    assert xlsx_notna == 1552, (
        f"Se esperaban 1552 filas xlsx con source_method no nulo, se obtuvieron {xlsx_notna}"
    )


# ---------------------------------------------------------------------------
# e) Low-confidence rows have empty main metadata fields
# ---------------------------------------------------------------------------

def test_low_conf_main_fields_empty(df):
    """Las filas crossref_by_title_low_confidence no deben tener title/authors/abstract/doi/year."""
    low_conf = df[df["source_method"] == "crossref_by_title_low_confidence"]
    assert len(low_conf) > 0, (
        "No se encontraron filas con source_method=='crossref_by_title_low_confidence'"
    )
    for field in ("title", "authors", "abstract", "doi", "year"):
        non_null = low_conf[field].notna().sum()
        assert non_null == 0, (
            f"Se esperaban 0 valores no nulos en '{field}' para filas low_confidence, "
            f"pero se encontraron {non_null}"
        )


# ---------------------------------------------------------------------------
# f) Manual recovery: #2098
# ---------------------------------------------------------------------------

def test_manual_recovery_2098(df):
    """#2098 debe tener exactamente 1 fila, source_method='manual', title/doi/year no nulos."""
    rows = df[df["covidence_id"] == "#2098"]
    assert len(rows) == 1, (
        f"Se esperaba exactamente 1 fila para #2098, se obtuvieron {len(rows)}"
    )
    row = rows.iloc[0]
    assert row["source_method"] == "manual", (
        f"#2098: source_method esperado='manual', obtenido={row['source_method']!r}"
    )
    assert pd.notna(row["title"]) and str(row["title"]).strip() != "", (
        f"#2098: title debe ser no nulo y no vacio, se obtuvo {row['title']!r}"
    )
    assert pd.notna(row["doi"]), (
        f"#2098: doi debe ser no nulo, se obtuvo {row['doi']!r}"
    )
    assert pd.notna(row["year"]), (
        f"#2098: year debe ser no nulo, se obtuvo {row['year']!r}"
    )


# ---------------------------------------------------------------------------
# g) Manual recovery: #3490
# ---------------------------------------------------------------------------

def test_manual_recovery_3490(df):
    """#3490 debe tener exactamente 1 fila, source_method='manual', title/doi/year no nulos."""
    rows = df[df["covidence_id"] == "#3490"]
    assert len(rows) == 1, (
        f"Se esperaba exactamente 1 fila para #3490, se obtuvieron {len(rows)}"
    )
    row = rows.iloc[0]
    assert row["source_method"] == "manual", (
        f"#3490: source_method esperado='manual', obtenido={row['source_method']!r}"
    )
    assert pd.notna(row["title"]) and str(row["title"]).strip() != "", (
        f"#3490: title debe ser no nulo y no vacio, se obtuvo {row['title']!r}"
    )
    assert pd.notna(row["doi"]), (
        f"#3490: doi debe ser no nulo, se obtuvo {row['doi']!r}"
    )
    assert pd.notna(row["year"]), (
        f"#3490: year debe ser no nulo, se obtuvo {row['year']!r}"
    )


# ---------------------------------------------------------------------------
# h) #6854 was recovered (not failed)
# ---------------------------------------------------------------------------

def test_6854_recovered(df):
    """#6854 debe tener 1 fila y source_method distinto de 'failed'."""
    rows = df[df["covidence_id"] == "#6854"]
    assert len(rows) == 1, (
        f"Se esperaba exactamente 1 fila para #6854, se obtuvieron {len(rows)}"
    )
    row = rows.iloc[0]
    assert row["source_method"] != "failed", (
        f"#6854: source_method no debe ser 'failed', se obtuvo {row['source_method']!r}"
    )


# ---------------------------------------------------------------------------
# i) #32107 preserved with pdfplumber_fallback and correct notes
# ---------------------------------------------------------------------------

def test_32107_preserved(df):
    """#32107 debe tener source_method='pdfplumber_fallback' y notes con 'PDF descarga falló'."""
    rows = df[df["covidence_id"] == "#32107"]
    assert len(rows) == 1, (
        f"Se esperaba exactamente 1 fila para #32107, se obtuvieron {len(rows)}"
    )
    row = rows.iloc[0]
    assert row["source_method"] == "pdfplumber_fallback", (
        f"#32107: source_method esperado='pdfplumber_fallback', obtenido={row['source_method']!r}"
    )
    notes = str(row["notes"]) if pd.notna(row["notes"]) else ""
    assert "PDF descarga falló" in notes, (
        f"#32107: se esperaba 'PDF descarga falló' en notes, se obtuvo {notes!r}"
    )


# ---------------------------------------------------------------------------
# k) No 'failed' source_method anywhere in the unified CSV
# ---------------------------------------------------------------------------

def test_no_failed_in_unified(df):
    """No debe quedar ninguna fila con source_method == 'failed' en el CSV unificado."""
    failed_count = (df["source_method"] == "failed").sum()
    assert failed_count == 0, (
        f"Se encontraron {failed_count} filas con source_method=='failed'. "
        "D.3.bis debio haber eliminado todos los 'failed'."
    )
