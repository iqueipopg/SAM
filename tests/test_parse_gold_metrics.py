"""Tests unitarios para scripts/parse_gold_metrics.py (tarea T16 Fase 2 Bloque C.5).

Estos tests validan el CSV de salida ya generado:
  data/gold_standard_normalized/gold_metrics.csv

NO re-ejecutan el script ni mockean datos: leen el CSV real generado por el parser.
Si el CSV no existe, los tests son omitidos con pytest.skip.

Notas sobre datos sucios conocidos:
  - covidence_id: el 72 % aprox de filas empieza por '#'. El resto son IDs
    alternativos como 'GC (Implementation intentions)', 'E2', 'E3', 'E4', etc.
    presentes en el gold standard original. Se tolera: el umbral del test es
    conservador (>= 50 % de filas con '#').
  - outcome_name: solo presente en BASELINE (~490/824 filas); BF1/BF2/BF3
    incluyen los mismos outcome names pero pueden ser null.
  - N_GC, N_GI y otras columnas numericas tienen NaN esperados.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "gold_standard_normalized" / "gold_metrics.csv"

CANONICAL_COLUMNS = [
    "covidence_id",
    "timepoint",
    "outcome_name",
    "outcome_category",
    "instrument",
    "instrument_type",
    "primario_secundario",
    "N_GC",
    "N_GI",
    "X_GC",
    "SD_GC",
    "X_GI",
    "SD_GI",
    "control_flag",
    "filtro_flag",
    "tendencia",
]

VALID_TIMEPOINTS = {"BASELINE", "BF1", "BF2", "BF3"}

NUMERIC_COLS = ["N_GC", "N_GI", "X_GC", "SD_GC", "X_GI", "SD_GI"]


# ---------------------------------------------------------------------------
# Fixture principal — carga el CSV una sola vez para todos los tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gold_df() -> pd.DataFrame:
    """Carga gold_metrics.csv. Salta todos los tests del modulo si no existe."""
    if not CSV_PATH.exists():
        pytest.skip(
            "CSV de gold_metrics no generado — ejecutar "
            "scripts/parse_gold_metrics.py primero"
        )
    df = pd.read_csv(str(CSV_PATH), encoding="utf-8")
    return df


# ---------------------------------------------------------------------------
# Test 1: el CSV existe y se lee sin error
# ---------------------------------------------------------------------------


def test_csv_exists_after_parse(gold_df: pd.DataFrame):
    """El CSV existe en la ruta canonica y pandas lo carga sin excepcion."""
    assert CSV_PATH.exists(), f"CSV no encontrado en: {CSV_PATH}"
    assert isinstance(gold_df, pd.DataFrame), "pd.read_csv no retorno un DataFrame"
    assert len(gold_df) > 0, "El CSV esta vacio"


# ---------------------------------------------------------------------------
# Test 2: columnas canonicas presentes
# ---------------------------------------------------------------------------


def test_required_columns_present(gold_df: pd.DataFrame):
    """El DataFrame contiene exactamente las 16 columnas canonicas."""
    missing = [col for col in CANONICAL_COLUMNS if col not in gold_df.columns]
    assert not missing, f"Columnas ausentes en el CSV: {missing}"


# ---------------------------------------------------------------------------
# Test 3: valores validos de timepoint
# ---------------------------------------------------------------------------


def test_timepoint_values_valid(gold_df: pd.DataFrame):
    """La columna timepoint solo contiene BASELINE, BF1, BF2 o BF3. Sin nulos."""
    assert gold_df["timepoint"].notna().all(), (
        "Se encontraron nulos en la columna timepoint — "
        "cada fila debe pertenecer a un timepoint conocido"
    )
    actual = set(gold_df["timepoint"].unique())
    unexpected = actual - VALID_TIMEPOINTS
    assert not unexpected, (
        f"Valores inesperados en timepoint: {unexpected}. "
        f"Solo se permiten {VALID_TIMEPOINTS}"
    )
    assert actual.issubset(VALID_TIMEPOINTS)


# ---------------------------------------------------------------------------
# Test 4: distribucion por timepoint — todos presentes al menos una vez
# ---------------------------------------------------------------------------


def test_timepoint_distribution(gold_df: pd.DataFrame):
    """Cada timepoint aparece al menos una vez en el CSV."""
    counts = gold_df["timepoint"].value_counts()
    for tp in VALID_TIMEPOINTS:
        assert tp in counts.index and counts[tp] >= 1, (
            f"Timepoint '{tp}' ausente o sin filas en el CSV"
        )


# ---------------------------------------------------------------------------
# Test 5: covidence_id es string; mayoria empieza por '#'
# ---------------------------------------------------------------------------


def test_covidence_id_parseable(gold_df: pd.DataFrame):
    """covidence_id contiene strings. Al menos el 50 % de filas no-null
    empieza por '#'.

    Nota: el gold standard tiene IDs alternativos conocidos (p.ej. 'E2', 'E3',
    'GC (Implementation intentions)') que no siguen el formato '#XXXXX'.
    El umbral conservador del 50 % detecta fallos catastroficos de parseo
    (p.ej. "solo se extrajeron 3 estudios") sin rechazar datos sucios esperados.
    El valor real del dataset actual es aprox 73 %.
    """
    non_null = gold_df["covidence_id"].dropna()
    assert len(non_null) > 0, "La columna covidence_id esta completamente vacia"

    # Asegurar que son strings (no numeros)
    assert non_null.dtype == object, (
        f"covidence_id deberia ser dtype object (str), es: {non_null.dtype}"
    )

    pct_with_hash = non_null.str.startswith("#").mean()
    assert pct_with_hash >= 0.50, (
        f"Solo el {pct_with_hash:.1%} de covidence_ids empieza por '#'. "
        "Umbral minimo: 50 %. Posible fallo de parseo."
    )


# ---------------------------------------------------------------------------
# Test 6: columnas numericas tienen dtype numerico
# ---------------------------------------------------------------------------


def test_numeric_columns_are_numeric(gold_df: pd.DataFrame):
    """N_GC, N_GI, X_GC, SD_GC, X_GI, SD_GI son dtype numerico. NaN permitido."""
    for col in NUMERIC_COLS:
        assert col in gold_df.columns, f"Columna numerica ausente: {col}"
        # pd.to_numeric no debe lanzar excepcion; errors='raise' solo falla con no-numericos
        try:
            converted = pd.to_numeric(gold_df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            pytest.fail(
                f"Columna '{col}' contiene valores no convertibles a numerico: {exc}"
            )
        # El dtype debe ser numerico (float64, int64, etc.)
        assert pd.api.types.is_numeric_dtype(gold_df[col]), (
            f"Columna '{col}' tiene dtype no-numerico: {gold_df[col].dtype}"
        )


# ---------------------------------------------------------------------------
# Test 7: no hay filas completamente vacias
# ---------------------------------------------------------------------------


def test_no_completely_empty_rows(gold_df: pd.DataFrame):
    """Ninguna fila tiene todos los campos nulos simultaneamente.

    Como minimo, covidence_id o timepoint deben estar presentes en cada fila
    (son los campos de identidad del registro).
    """
    anchor_cols = ["covidence_id", "timepoint"]
    all_anchors_null = gold_df[anchor_cols].isnull().all(axis=1)
    n_empty = all_anchors_null.sum()
    assert n_empty == 0, (
        f"{n_empty} fila(s) tienen covidence_id y timepoint ambos nulos. "
        "Indica filas huerfanas en el CSV."
    )


# ---------------------------------------------------------------------------
# Test 8: numero de estudios unicos en rango razonable
# ---------------------------------------------------------------------------


def test_n_studies_reasonable(gold_df: pd.DataFrame):
    """El numero de covidence_id unicos esta en rango [50, 100].

    Banda ancha para tolerar variaciones del dataset sin detectar desastres
    de parseo (p.ej. 'el parser solo extrajo 3 estudios').
    Valor esperado del snapshot actual: 76.
    """
    n_unique = gold_df["covidence_id"].nunique()
    assert 50 <= n_unique <= 100, (
        f"Numero de estudios unicos fuera de rango: {n_unique}. "
        "Rango esperado: [50, 100]. Valor en snapshot actual: 76."
    )
