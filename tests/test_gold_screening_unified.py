# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for data/gold_standard/deporte/unified/gold_screening_unified.csv.

These tests read the already-generated CSV and validate its canonical structure,
label distributions, DOI normalization, year handling, and specific documented
decisions.

No GPU, no network, no CSV regeneration — if the CSV does not exist the tests
skip with a clear message.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

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
    "Generate it first with: python scripts/build_unified_gold_screening.py"
)


# ---------------------------------------------------------------------------
# Shared fixture — loaded once per module via session scope
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df():
    """Load the unified CSV once for all tests in this module."""
    if not CSV_PATH.exists():
        pytest.skip(SKIP_REASON)
    return pd.read_csv(CSV_PATH)


# ---------------------------------------------------------------------------
# 1. Shape
# ---------------------------------------------------------------------------

def test_shape_exact(df):
    """CSV must have exactly 20750 rows and 13 columns (schema v2)."""
    assert df.shape == (20750, 13), (
        f"Expected (20750, 13), got {df.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Columns — exact names in exact order
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
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


def test_columns_exact_order(df):
    """Column names and order must match the canonical schema v2 exactly."""
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Column mismatch.\n  Expected: {EXPECTED_COLUMNS}\n  Got:      {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# 3. covidence_id integrity
# ---------------------------------------------------------------------------

_COVIDENCE_RE = re.compile(r"^#\d+$")


def test_covidence_id_count_unique(df):
    """All 20750 covidence_ids must be unique."""
    assert df["covidence_id"].nunique() == 20750, (
        f"Expected 20750 unique ids, got {df['covidence_id'].nunique()}"
    )


def test_covidence_id_no_nulls(df):
    """covidence_id must have zero null values."""
    null_count = df["covidence_id"].isna().sum()
    assert null_count == 0, f"Found {null_count} null covidence_ids"


def test_covidence_id_format(df):
    """Every covidence_id must match the pattern ^#[0-9]+$."""
    bad = df[~df["covidence_id"].apply(lambda x: bool(_COVIDENCE_RE.match(str(x))))][
        "covidence_id"
    ]
    assert len(bad) == 0, (
        f"Found {len(bad)} ids with invalid format. Examples: {bad.head(5).tolist()}"
    )


# ---------------------------------------------------------------------------
# 4. Overlap xlsx vs csv = 0
# ---------------------------------------------------------------------------

def test_no_overlap_xlsx_csv(df):
    """No covidence_id should appear in both source=='xlsx' and source=='csv'."""
    xlsx_ids = set(df.loc[df["source"] == "xlsx", "covidence_id"])
    csv_ids = set(df.loc[df["source"] == "csv", "covidence_id"])
    overlap = xlsx_ids & csv_ids
    assert len(overlap) == 0, (
        f"Found {len(overlap)} ids in both sources: {sorted(overlap)[:10]}"
    )


# ---------------------------------------------------------------------------
# 5. Source counts
# ---------------------------------------------------------------------------

def test_source_counts(df):
    """xlsx=1552 rows, csv=19198 rows."""
    counts = df["source"].value_counts().to_dict()
    assert counts.get("xlsx") == 1552, (
        f"Expected 1552 xlsx rows, got {counts.get('xlsx')}"
    )
    assert counts.get("csv") == 19198, (
        f"Expected 19198 csv rows, got {counts.get('csv')}"
    )


# ---------------------------------------------------------------------------
# 6. human_label_ta
# ---------------------------------------------------------------------------

def test_human_label_ta_values(df):
    """human_label_ta must contain only 'relevant' and 'irrelevant'."""
    unique_vals = set(df["human_label_ta"].dropna())
    assert unique_vals == {"relevant", "irrelevant"}, (
        f"Unexpected values in human_label_ta: {unique_vals}"
    )


def test_human_label_ta_counts(df):
    """relevant=1552, irrelevant=19198."""
    counts = df["human_label_ta"].value_counts().to_dict()
    assert counts.get("relevant") == 1552, (
        f"Expected 1552 relevant, got {counts.get('relevant')}"
    )
    assert counts.get("irrelevant") == 19198, (
        f"Expected 19198 irrelevant, got {counts.get('irrelevant')}"
    )


# ---------------------------------------------------------------------------
# 7. human_label_ft
# ---------------------------------------------------------------------------

def test_human_label_ft_counts(df):
    """include=66, exclude=1486, null=19198."""
    counts = df["human_label_ft"].value_counts(dropna=False).to_dict()
    # pandas may use NaN as the key for null
    null_count = df["human_label_ft"].isna().sum()
    assert counts.get("include") == 66, (
        f"Expected 66 include, got {counts.get('include')}"
    )
    assert counts.get("exclude") == 1486, (
        f"Expected 1486 exclude, got {counts.get('exclude')}"
    )
    assert null_count == 19198, (
        f"Expected 19198 null human_label_ft, got {null_count}"
    )


def test_irrelevant_ta_implies_null_ft(df):
    """All rows with human_label_ta=='irrelevant' must have null human_label_ft."""
    irrelevant = df[df["human_label_ta"] == "irrelevant"]
    non_null_ft = irrelevant["human_label_ft"].notna().sum()
    assert non_null_ft == 0, (
        f"Found {non_null_ft} irrelevant T/A rows with non-null human_label_ft"
    )


# ---------------------------------------------------------------------------
# 8 & 9. exclusion_reason_ft
# ---------------------------------------------------------------------------

EXPECTED_REASON_COUNTS = {
    "Wrong age": 1178,
    "Wrong outcome": 137,
    "Wrong study design": 75,
    "Wrong intervention": 50,
    "Wrong publication type": 25,
    "Wrong population": 14,
    "Language": 3,
    "Sin motivo registrado": 4,
}


def test_exclusion_reason_ft_counts(df):
    """Canonical exclusion reason counts among relevant T/A rows."""
    relevant = df[df["human_label_ta"] == "relevant"]
    counts = relevant["exclusion_reason_ft"].value_counts(dropna=True).to_dict()
    for reason, expected_count in EXPECTED_REASON_COUNTS.items():
        observed = counts.get(reason, 0)
        assert observed == expected_count, (
            f"exclusion_reason_ft '{reason}': expected {expected_count}, got {observed}"
        )
    # No unexpected reason values
    known_reasons = set(EXPECTED_REASON_COUNTS.keys())
    observed_reasons = set(counts.keys())
    unexpected = observed_reasons - known_reasons
    assert not unexpected, (
        f"Unexpected exclusion_reason_ft values found: {unexpected}"
    )


def test_null_reason_for_included_and_irrelevant(df):
    """exclusion_reason_ft must be null for every include row and every irrelevant row."""
    include_rows = df[df["human_label_ft"] == "include"]
    bad_include = include_rows["exclusion_reason_ft"].notna().sum()
    assert bad_include == 0, (
        f"Found {bad_include} included rows with non-null exclusion_reason_ft"
    )

    irrelevant_rows = df[df["human_label_ta"] == "irrelevant"]
    bad_irrelevant = irrelevant_rows["exclusion_reason_ft"].notna().sum()
    assert bad_irrelevant == 0, (
        f"Found {bad_irrelevant} irrelevant rows with non-null exclusion_reason_ft"
    )


def test_exclusion_reason_null_count(df):
    """Null exclusion_reason_ft count = 66 (include) + 19198 (irrelevant) = 19264."""
    null_count = df["exclusion_reason_ft"].isna().sum()
    assert null_count == 19264, (
        f"Expected 19264 null exclusion_reason_ft rows, got {null_count}"
    )


# ---------------------------------------------------------------------------
# 10. Specific row: #32107
# ---------------------------------------------------------------------------

def test_row_32107(df):
    """Row #32107: source=xlsx, relevant, exclude, Wrong publication type, notes contains 'PDF descarga falló'."""
    rows = df[df["covidence_id"] == "#32107"]
    assert len(rows) == 1, f"Expected exactly 1 row for #32107, got {len(rows)}"
    row = rows.iloc[0]
    assert row["source"] == "xlsx", f"source={row['source']!r}"
    assert row["human_label_ta"] == "relevant", f"human_label_ta={row['human_label_ta']!r}"
    assert row["human_label_ft"] == "exclude", f"human_label_ft={row['human_label_ft']!r}"
    assert row["exclusion_reason_ft"] == "Wrong publication type", (
        f"exclusion_reason_ft={row['exclusion_reason_ft']!r}"
    )
    notes = str(row["notes"]) if pd.notna(row["notes"]) else ""
    assert "PDF descarga falló" in notes, (
        f"Expected 'PDF descarga falló' in notes, got {notes!r}"
    )


# ---------------------------------------------------------------------------
# 11. Anomalous years
# ---------------------------------------------------------------------------

ANOMALOUS_YEAR_EXPECTED = {
    "#51": "1982-1983",
    "#127": "1985-1986",
    "#208": "1988-1989",
    "#4660": "2010-2011",
}


def test_anomalous_year_rows(df):
    """The 4 anomalous year rows must have null year and notes containing 'year original: <value>'."""
    for cid, expected_original in ANOMALOUS_YEAR_EXPECTED.items():
        rows = df[df["covidence_id"] == cid]
        assert len(rows) == 1, f"Expected 1 row for {cid}, got {len(rows)}"
        row = rows.iloc[0]
        assert pd.isna(row["year"]), (
            f"{cid}: expected null year, got {row['year']!r}"
        )
        notes = str(row["notes"]) if pd.notna(row["notes"]) else ""
        expected_substr = f"year original: {expected_original}"
        assert expected_substr in notes, (
            f"{cid}: expected '{expected_substr}' in notes, got {notes!r}"
        )


# ---------------------------------------------------------------------------
# 12. DOI normalization
# ---------------------------------------------------------------------------

def test_doi_no_url_prefix(df):
    """No non-null DOI should start with 'https://', 'http://', or 'doi:'."""
    doi_nonnull = df["doi"].dropna()
    bad_https = doi_nonnull.str.startswith("https://").sum()
    bad_http = doi_nonnull.str.startswith("http://").sum()
    bad_doi_prefix = doi_nonnull.str.startswith("doi:").sum()
    assert bad_https == 0, f"Found {bad_https} DOIs starting with 'https://'"
    assert bad_http == 0, f"Found {bad_http} DOIs starting with 'http://'"
    assert bad_doi_prefix == 0, f"Found {bad_doi_prefix} DOIs starting with 'doi:'"


def test_doi_all_lowercase(df):
    """All non-null DOIs must be lowercase (no uppercase letters)."""
    doi_nonnull = df["doi"].dropna()
    not_lower = (doi_nonnull != doi_nonnull.str.lower()).sum()
    assert not_lower == 0, (
        f"Found {not_lower} DOIs with uppercase characters"
    )


# ---------------------------------------------------------------------------
# 13. DOI duplicates post-normalization
# ---------------------------------------------------------------------------

def test_doi_duplicate_extra_rows(df):
    """Exactly 322 extra duplicate DOI rows (rows_with_doi - unique_non_null_dois).

    v2: xlsx-side rows now carry DOIs from extracted_metadata.csv, raising
    the duplicate count from 308 (v1) to 322 (v2).
    """
    doi_nonnull = df["doi"].dropna()
    n_with_doi = len(doi_nonnull)
    n_unique = doi_nonnull.nunique()
    dup_extra = n_with_doi - n_unique
    assert dup_extra == 322, (
        f"Expected 322 duplicate-extra DOI rows, got {dup_extra} "
        f"(rows_with_doi={n_with_doi}, unique_dois={n_unique})"
    )


# ---------------------------------------------------------------------------
# 14. Year range and type
# ---------------------------------------------------------------------------

def test_year_nonnull_count(df):
    """Non-null year count must be 20570.

    v2: xlsx-side rows recovered year from extracted_metadata.csv,
    adding ~1394 previously-null entries (v1 count was 19176).
    """
    nonnull_count = df["year"].notna().sum()
    assert nonnull_count == 20570, (
        f"Expected 20570 non-null year values, got {nonnull_count}"
    )


def test_year_integer_valued(df):
    """All non-null year values must be integer-valued (no fractional years)."""
    year_nonnull = df["year"].dropna()
    fractional = (year_nonnull % 1 != 0).sum()
    assert fractional == 0, (
        f"Found {fractional} non-integer year values"
    )


def test_year_range(df):
    """Non-null years must fall between 1967 and 2027 (inclusive).

    v2: xlsx-side metadata includes papers published up to 2027; v1 max was 2021.
    """
    year_nonnull = df["year"].dropna()
    assert int(year_nonnull.min()) == 1967, (
        f"Expected min year 1967, got {year_nonnull.min()}"
    )
    assert int(year_nonnull.max()) == 2027, (
        f"Expected max year 2027, got {year_nonnull.max()}"
    )


def test_xlsx_rows_year_coverage(df):
    """En v2, las filas xlsx-side recuperan year cuando la extraccion fue exitosa.

    Esperamos ~1394 filas xlsx-side con year no nulo (1552 - 158 sin year extraible).
    """
    xlsx_year_nonnull = df[df["source"] == "xlsx"]["year"].notna().sum()
    assert xlsx_year_nonnull == 1394, (
        f"Expected 1394 xlsx rows with non-null year, got {xlsx_year_nonnull}"
    )


# ---------------------------------------------------------------------------
# Optional: metadata coverage (csv-only rows)
# ---------------------------------------------------------------------------

def test_csv_rows_without_doi(df):
    """Among csv rows, 1336 have no DOI."""
    csv_rows = df[df["source"] == "csv"]
    no_doi = csv_rows["doi"].isna().sum()
    assert no_doi == 1336, (
        f"Expected 1336 csv rows without DOI, got {no_doi}"
    )


def test_csv_rows_without_abstract(df):
    """Among csv rows, 136 have no abstract."""
    csv_rows = df[df["source"] == "csv"]
    no_abstract = csv_rows["abstract"].isna().sum()
    assert no_abstract == 136, (
        f"Expected 136 csv rows without abstract, got {no_abstract}"
    )


def test_csv_rows_without_title(df):
    """Among csv rows, 7 have no title."""
    csv_rows = df[df["source"] == "csv"]
    no_title = csv_rows["title"].isna().sum()
    assert no_title == 7, (
        f"Expected 7 csv rows without title, got {no_title}"
    )
