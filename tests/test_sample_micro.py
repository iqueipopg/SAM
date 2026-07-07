# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for T16.D.4.B: sample micro estratificado de 50 papers.

Validates data/gold_standard/deporte/samples/micro/sample_50.csv generated
by scripts/build_micro_sample.py.

No GPU, no network. Tests skip if required data files are missing.
"""

from __future__ import annotations

import hashlib
import sys
import os
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path — allow importing from scripts/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLE_CSV = (
    PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "samples" / "micro" / "sample_50.csv"
)
UNIFIED_CSV = (
    PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "unified" / "gold_screening_unified.csv"
)
XLSX_PATH = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "Resumen_Total_PDFs.xlsx"
PDFS_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_todos"

SKIP_SAMPLE = pytest.mark.skipif(
    not SAMPLE_CSV.exists(),
    reason=f"sample_50.csv not found at {SAMPLE_CSV}. Run: python scripts/build_micro_sample.py",
)
SKIP_UNIFIED = pytest.mark.skipif(
    not UNIFIED_CSV.exists(),
    reason=f"Unified CSV not found at {UNIFIED_CSV}. Run: python scripts/build_unified_gold_screening_v2.py",
)
SKIP_XLSX = pytest.mark.skipif(
    not XLSX_PATH.exists(),
    reason=f"XLSX not found at {XLSX_PATH}.",
)
SKIP_PDFS = pytest.mark.skipif(
    not PDFS_DIR.exists(),
    reason=f"pdfs_todos directory not found at {PDFS_DIR}.",
)

# ---------------------------------------------------------------------------
# Expected constants (mirror spec)
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "covidence_id",
    "strata",
    "human_label_ta",
    "human_label_ft",
    "exclusion_reason_ft",
    "has_abstract",
    "has_pdf",
    "source_method",
    "purpose",
    "pdf_filename",
]

EXPECTED_STRATA: dict[str, int] = {
    "include": 10,
    "Wrong age": 10,
    "Wrong outcome": 8,
    "Wrong study design": 6,
    "Wrong intervention": 6,
    "Wrong publication type": 4,
    "Wrong population": 3,
    "Language": 2,
    "Sin motivo registrado": 1,
}

EXCLUDED_SOURCE_METHODS = {"crossref_by_title_low_confidence", "failed"}


# ---------------------------------------------------------------------------
# Module-level fixtures — loaded once to keep the suite fast
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df():
    """Load sample_50.csv once for the whole module."""
    if not SAMPLE_CSV.exists():
        pytest.skip(f"sample_50.csv not found: {SAMPLE_CSV}")
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture(scope="module")
def unified_df():
    """Load unified CSV once for the whole module."""
    if not UNIFIED_CSV.exists():
        pytest.skip(f"Unified CSV not found: {UNIFIED_CSV}")
    return pd.read_csv(UNIFIED_CSV)


# ---------------------------------------------------------------------------
# 1. Shape
# ---------------------------------------------------------------------------

@SKIP_SAMPLE
def test_shape(sample_df):
    """CSV must exist, have shape (50, 10) and columns in the exact declared order."""
    assert SAMPLE_CSV.exists(), f"sample_50.csv does not exist at {SAMPLE_CSV}"
    assert sample_df.shape == (50, 10), (
        f"Shape esperada (50, 10), obtenida {sample_df.shape}"
    )
    assert list(sample_df.columns) == EXPECTED_COLUMNS, (
        f"Columnas inesperadas.\nEsperadas: {EXPECTED_COLUMNS}\n"
        f"Obtenidas:  {list(sample_df.columns)}"
    )


# ---------------------------------------------------------------------------
# 2. Strata distribution
# ---------------------------------------------------------------------------

@SKIP_SAMPLE
def test_strata_distribution_exact(sample_df):
    """Cada estrato debe tener exactamente el numero de filas declarado."""
    counts = sample_df["strata"].value_counts().to_dict()
    total = sum(counts.values())
    assert total == 50, f"Total esperado 50, obtenido {total}"

    for strata_name, expected_n in EXPECTED_STRATA.items():
        got = counts.get(strata_name, 0)
        assert got == expected_n, (
            f"Estrato '{strata_name}': esperado {expected_n}, obtenido {got}"
        )

    # No extra strata outside the declared set
    allowed = set(EXPECTED_STRATA.keys())
    found = set(counts.keys())
    extra = found - allowed
    assert not extra, f"Estratos no declarados encontrados: {extra}"


# ---------------------------------------------------------------------------
# 3. No low-confidence or failed source_method
# ---------------------------------------------------------------------------

@SKIP_SAMPLE
def test_no_low_confidence_or_failed(sample_df):
    """Cero filas con source_method in {crossref_by_title_low_confidence, failed}."""
    bad = sample_df[sample_df["source_method"].isin(EXCLUDED_SOURCE_METHODS)]
    assert len(bad) == 0, (
        f"{len(bad)} filas con source_method excluido: "
        f"{bad[['covidence_id', 'source_method']].to_dict(orient='records')}"
    )


# ---------------------------------------------------------------------------
# 4. Includes are subset of the 66 real includes from unified
# ---------------------------------------------------------------------------

@SKIP_SAMPLE
@SKIP_UNIFIED
def test_includes_subset_of_66_real(sample_df, unified_df):
    """Los 10 includes del sample deben ser subset de los 66 includes del unified."""
    real_includes = set(
        unified_df[unified_df["human_label_ft"] == "include"]["covidence_id"].tolist()
    )
    assert len(real_includes) == 66, (
        f"Se esperan 66 includes reales en el unified, se encontraron {len(real_includes)}"
    )

    sample_includes = set(
        sample_df[sample_df["strata"] == "include"]["covidence_id"].tolist()
    )
    assert len(sample_includes) == 10, (
        f"El sample debe tener exactamente 10 includes, tiene {len(sample_includes)}"
    )

    extra = sample_includes - real_includes
    assert not extra, (
        f"Los siguientes covidence_ids del sample no son includes reales: {extra}"
    )


# ---------------------------------------------------------------------------
# 5. Includes have has_pdf == True (invariante critico post D.3.bis)
# ---------------------------------------------------------------------------

@SKIP_SAMPLE
def test_includes_have_pdf(sample_df):
    """Todas las filas con strata=='include' deben tener has_pdf==True."""
    includes = sample_df[sample_df["strata"] == "include"]
    no_pdf = includes[~includes["has_pdf"]]
    assert len(no_pdf) == 0, (
        f"Includes sin PDF (invariante critico violado): "
        f"{no_pdf['covidence_id'].tolist()}"
    )


# ---------------------------------------------------------------------------
# 6. Seed reproducibility — import approach
# ---------------------------------------------------------------------------

@SKIP_UNIFIED
@SKIP_XLSX
@SKIP_PDFS
def test_seed_reproducibility():
    """build_sample dos veces con la misma seed produce resultados identicos byte-a-byte."""
    # Import approach (preferred: pure function, no side effects)
    from scripts.build_micro_sample import build_sample, SEED

    unified_df = pd.read_csv(str(UNIFIED_CSV))
    xlsx_df = pd.read_excel(str(XLSX_PATH))
    pdfs_existing = {p.name for p in PDFS_DIR.iterdir() if p.is_file()}

    sample_a = build_sample(unified_df, xlsx_df, pdfs_existing, seed=SEED)
    sample_b = build_sample(unified_df, xlsx_df, pdfs_existing, seed=SEED)

    csv_a = sample_a.to_csv(index=False)
    csv_b = sample_b.to_csv(index=False)

    assert csv_a == csv_b, (
        "Reproducibilidad rota: dos llamadas a build_sample con la misma seed "
        "generaron CSVs distintos."
    )


# ---------------------------------------------------------------------------
# 7. Purpose consistent with strata
# ---------------------------------------------------------------------------

@SKIP_SAMPLE
def test_purpose_consistent_with_strata(sample_df):
    """includes -> purpose=='hero1,hero2,hero3,hero4'. Resto -> purpose=='hero1,hero2'."""
    PURPOSE_INCLUDE = "hero1,hero2,hero3,hero4"
    PURPOSE_EXCLUDE = "hero1,hero2"

    includes = sample_df[sample_df["strata"] == "include"]
    wrong_inc = includes[includes["purpose"] != PURPOSE_INCLUDE]
    assert len(wrong_inc) == 0, (
        f"{len(wrong_inc)} includes con purpose incorrecto: "
        f"{wrong_inc[['covidence_id', 'purpose']].to_dict(orient='records')}"
    )

    excludes = sample_df[sample_df["strata"] != "include"]
    wrong_exc = excludes[excludes["purpose"] != PURPOSE_EXCLUDE]
    assert len(wrong_exc) == 0, (
        f"{len(wrong_exc)} excludes con purpose incorrecto: "
        f"{wrong_exc[['covidence_id', 'strata', 'purpose']].to_dict(orient='records')}"
    )


# ---------------------------------------------------------------------------
# 8. Human labels consistent with strata
# ---------------------------------------------------------------------------

@SKIP_SAMPLE
def test_human_labels_consistent(sample_df):
    """
    strata=='include' -> human_label_ft=='include'.
    Otros estratos -> human_label_ft=='exclude' Y exclusion_reason_ft==strata.
    """
    includes = sample_df[sample_df["strata"] == "include"]
    wrong_label = includes[includes["human_label_ft"] != "include"]
    assert len(wrong_label) == 0, (
        f"Includes con human_label_ft incorrecto: "
        f"{wrong_label[['covidence_id', 'human_label_ft']].to_dict(orient='records')}"
    )

    excludes = sample_df[sample_df["strata"] != "include"]

    wrong_ft_label = excludes[excludes["human_label_ft"] != "exclude"]
    assert len(wrong_ft_label) == 0, (
        f"Excludes con human_label_ft != 'exclude': "
        f"{wrong_ft_label[['covidence_id', 'strata', 'human_label_ft']].to_dict(orient='records')}"
    )

    reason_mismatch = excludes[excludes["exclusion_reason_ft"] != excludes["strata"]]
    assert len(reason_mismatch) == 0, (
        f"exclusion_reason_ft no coincide con strata en {len(reason_mismatch)} filas: "
        f"{reason_mismatch[['covidence_id', 'strata', 'exclusion_reason_ft']].to_dict(orient='records')}"
    )


# ---------------------------------------------------------------------------
# 9. pdf_filename when has_pdf
# ---------------------------------------------------------------------------

@SKIP_SAMPLE
@SKIP_PDFS
def test_pdf_filename_when_has_pdf(sample_df):
    """Filas con has_pdf==True: pdf_filename no es NaN y el archivo existe en pdfs_todos."""
    has_pdf_rows = sample_df[sample_df["has_pdf"] == True]

    for _, row in has_pdf_rows.iterrows():
        assert pd.notna(row["pdf_filename"]), (
            f"covidence_id {row['covidence_id']}: has_pdf=True pero pdf_filename es NaN"
        )
        pdf_path = PDFS_DIR / str(row["pdf_filename"])
        assert pdf_path.exists(), (
            f"covidence_id {row['covidence_id']}: pdf_filename={row['pdf_filename']} "
            f"no existe en {PDFS_DIR}"
        )


# ---------------------------------------------------------------------------
# 10. Abstract coverage metric (informativo, no falla por bajo coverage)
# ---------------------------------------------------------------------------

@SKIP_SAMPLE
def test_abstract_coverage_metric(sample_df, capsys):
    """Imprime cobertura de has_abstract overall y por estrato. No falla por bajo coverage."""
    overall_cov = float(sample_df["has_abstract"].sum()) / len(sample_df)
    assert 0.0 <= overall_cov <= 1.0, (
        f"Cobertura overall fuera de [0,1]: {overall_cov}"
    )

    print(f"\n[abstract coverage] overall: {overall_cov:.2%} ({int(sample_df['has_abstract'].sum())}/50)")
    for strata_name in EXPECTED_STRATA:
        sub = sample_df[sample_df["strata"] == strata_name]
        if len(sub) == 0:
            continue
        cov = float(sub["has_abstract"].sum()) / len(sub)
        assert 0.0 <= cov <= 1.0, (
            f"Cobertura de estrato '{strata_name}' fuera de [0,1]: {cov}"
        )
        print(f"  {strata_name:<25}: {cov:.0%} ({int(sub['has_abstract'].sum())}/{len(sub)})")

    # Soft assertion: coverage in total must be a valid float in [0,1]
    # (no strict threshold — low abstract coverage for excludes is expected)
    captured = capsys.readouterr()
    assert "[abstract coverage]" in captured.out


# ---------------------------------------------------------------------------
# 11. Imports of pure helpers
# ---------------------------------------------------------------------------

def test_imports_of_pure_helpers():
    """build_pool, sample_stratum, compute_has_pdf, pick_pdf_filename, build_sample importables y callables."""
    from scripts.build_micro_sample import (
        build_pool,
        sample_stratum,
        compute_has_pdf,
        pick_pdf_filename,
        build_sample,
    )
    assert callable(build_pool), "build_pool no es callable"
    assert callable(sample_stratum), "sample_stratum no es callable"
    assert callable(compute_has_pdf), "compute_has_pdf no es callable"
    assert callable(pick_pdf_filename), "pick_pdf_filename no es callable"
    assert callable(build_sample), "build_sample no es callable"
