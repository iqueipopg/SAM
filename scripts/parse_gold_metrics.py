# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Parser y normalizador del gold standard de deporte.

Lee `data/gold_standard/deporte/estadisticos_deporte.xlsx` (4 hojas:
BASELINE, BF1, BF2, BF3) y produce un CSV unificado con esquema canonico
en `data/gold_standard_normalized/gold_metrics.csv`.

Uso:
    ~/meta-analisis-automatizado/venv/bin/python scripts/parse_gold_metrics.py

Tarea T16 Fase 2 Bloque C.3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# Paths (absolutos, no hardcodear modelos/rutas de pipeline — esto es script estatico de datos)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "estadisticos_deporte.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "gold_standard_normalized"
OUTPUT_CSV = OUTPUT_DIR / "gold_metrics.csv"


# Esquema canonico unificado (orden de columnas en CSV de salida)
CANONICAL_COLUMNS: List[str] = [
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


# Mapeos por hoja: nombre real en Excel -> nombre canonico.
# Las columnas no listadas en cada mapeo se rellenan con NaN para esa hoja.
SHEET_MAPPINGS: Dict[str, Dict[str, str]] = {
    "BASELINE": {
        "CovidenceID": "covidence_id",
        "Control": "control_flag",
        "Filtro": "filtro_flag",
        "Outcome": "outcome_name",
        "tipo": "outcome_category",
        "instrumento": "instrument",
        "Instrumento": "instrument_type",
        "Primario/Secundario": "primario_secundario",
        "Tendencia": "tendencia",
        "N_GC": "N_GC",
        "N_GI": "N_GI",
        "X_GC": "X_GC",
        "SD_GC": "SD_GC",
        "X_GI": "X_GI",
        "SD_GI": "SD_GI",
    },
    "BF1": {
        "CovidenceID": "covidence_id",
        "Control": "control_flag",
        "Filtro": "filtro_flag",
        "Outcome": "outcome_name",
        "N_GC": "N_GC",
        "N_GI": "N_GI",
        "X_GC": "X_GC",
        "SD_GC": "SD_GC",
        "X_GI": "X_GI",
        "SD_GI": "SD_GI",
    },
    "BF2": {
        "CovidenceID": "covidence_id",
        "Control": "control_flag",
        "Filtro": "filtro_flag",
        "Outcome": "outcome_name",
        "N_GC": "N_GC",
        "N_GI": "N_GI",
        "X_GC": "X_GC",
        "SD_GC": "SD_GC",
        "X_GI": "X_GI",
        "SD_GI": "SD_GI",
    },
    "BF3": {
        "CovidenceID": "covidence_id",
        "Control": "control_flag",
        "Filtro": "filtro_flag",
        "Outcome": "outcome_name",
        "N_GC": "N_GC",
        "N_GI": "N_GI",
        "X_GC": "X_GC",
        "SD_GC": "SD_GC",
        "X_GI": "X_GI",
        "SD_GI": "SD_GI",
    },
}


NUMERIC_COLS: Tuple[str, ...] = ("N_GC", "N_GI", "X_GC", "SD_GC", "X_GI", "SD_GI")
STRING_COLS: Tuple[str, ...] = (
    "covidence_id",
    "outcome_name",
    "outcome_category",
    "instrument",
    "instrument_type",
    "primario_secundario",
    "tendencia",
    "control_flag",
    "filtro_flag",
)


def _strip_str(value):
    """Trim whitespace en strings; deja NaN/None tal cual."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    # Para flags numericos (Control=1, Filtro=3, etc.) los convertimos a str
    return str(value).strip()


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convierte strings sucios a numeric con NaN para no-numericos."""
    return pd.to_numeric(series, errors="coerce")


def normalize_sheet(
    df: pd.DataFrame, sheet_name: str, mapping: Dict[str, str], anomalies: List[str]
) -> pd.DataFrame:
    """Aplica mapeo + rellena columnas ausentes + tipa correctamente."""
    # Validar que las columnas esperadas existen en la hoja
    missing_in_sheet = [c for c in mapping if c not in df.columns]
    if missing_in_sheet:
        anomalies.append(
            f"[{sheet_name}] columnas declaradas en mapeo pero ausentes en Excel: {missing_in_sheet}"
        )

    # Filtrar filas que tengan al menos CovidenceID
    cov_col = "CovidenceID"
    if cov_col not in df.columns:
        raise ValueError(f"Hoja {sheet_name} no tiene columna CovidenceID")
    df = df[df[cov_col].notna()].copy()

    # Construir DataFrame normalizado
    normalized = pd.DataFrame(index=df.index)
    for excel_col, canonical_col in mapping.items():
        if excel_col in df.columns:
            normalized[canonical_col] = df[excel_col]

    # Anyadir timepoint
    normalized["timepoint"] = sheet_name

    # Anyadir columnas ausentes como NaN
    for canonical in CANONICAL_COLUMNS:
        if canonical not in normalized.columns:
            normalized[canonical] = pd.NA

    # Reordenar
    normalized = normalized[CANONICAL_COLUMNS]

    # Limpiar strings
    for col in STRING_COLS:
        if col in normalized.columns:
            normalized[col] = normalized[col].apply(_strip_str)

    # Coercer numericos
    for col in NUMERIC_COLS:
        if col in normalized.columns:
            before_nan = normalized[col].isna().sum()
            normalized[col] = _coerce_numeric(normalized[col])
            after_nan = normalized[col].isna().sum()
            dirty_cells = after_nan - before_nan
            if dirty_cells > 0:
                anomalies.append(
                    f"[{sheet_name}] {dirty_cells} valor(es) no-numerico(s) en columna '{col}' "
                    f"convertido(s) a NaN"
                )

    return normalized


def main() -> None:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el Excel del gold standard: {EXCEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo Excel: {EXCEL_PATH}")
    sheets = pd.read_excel(EXCEL_PATH, sheet_name=None)
    print(f"Hojas detectadas: {list(sheets.keys())}")

    anomalies: List[str] = []
    normalized_frames: List[pd.DataFrame] = []
    mappings_log: Dict[str, Dict[str, str]] = {}

    for sheet_name, mapping in SHEET_MAPPINGS.items():
        if sheet_name not in sheets:
            anomalies.append(f"Hoja esperada ausente en Excel: {sheet_name}")
            continue
        df = sheets[sheet_name]
        print(f"\nProcesando hoja '{sheet_name}' (shape original: {df.shape})")
        print(f"  Columnas reales: {df.columns.tolist()}")
        normalized = normalize_sheet(df, sheet_name, mapping, anomalies)
        print(f"  Filas normalizadas: {len(normalized)}")
        normalized_frames.append(normalized)
        mappings_log[sheet_name] = dict(mapping)

    # Concatenar
    combined = pd.concat(normalized_frames, ignore_index=True)

    # Persistir
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nCSV escrito en: {OUTPUT_CSV}")

    # ==========================================================
    # Reporte stdout (en espanol)
    # ==========================================================
    print("\n" + "=" * 60)
    print("REPORTE DE NORMALIZACION")
    print("=" * 60)
    print(f"Filas totales resultantes: {len(combined)}")
    print(f"Outcomes unicos (outcome_name distinto, no-nulo): "
          f"{combined['outcome_name'].dropna().nunique()}")
    print(f"Estudios unicos (covidence_id distinto): "
          f"{combined['covidence_id'].dropna().nunique()}")

    print("\nBreakdown por timepoint:")
    tp_counts = combined["timepoint"].value_counts().reindex(
        ["BASELINE", "BF1", "BF2", "BF3"]
    )
    for tp, cnt in tp_counts.items():
        print(f"  {tp}: {cnt} filas")

    print("\nDiccionario de mapeos aplicado por hoja:")
    for sheet_name, mp in mappings_log.items():
        print(f"\n  [{sheet_name}]")
        for excel_col, canonical in mp.items():
            print(f"    '{excel_col}' -> '{canonical}'")
        # Columnas canonicas no presentes -> NaN
        present_canonical = set(mp.values())
        absent = [c for c in CANONICAL_COLUMNS if c not in present_canonical and c != "timepoint"]
        if absent:
            print(f"    (NaN forzado para: {absent})")

    print("\nAnomalias detectadas:")
    if anomalies:
        for a in anomalies:
            print(f"  - {a}")
    else:
        print("  Ninguna anomalia critica detectada.")

    # Notas de auditoria sobre ambiguedades resueltas
    print("\nNotas de auditoria (decisiones de mapeo ambiguas resueltas):")
    print("  - BASELINE tiene dos columnas similares: 'instrumento' (minus.) e")
    print("    'Instrumento' (mayus.). Decisi'on: 'instrumento' -> instrument")
    print("    (texto largo p.ej. 'A (ActiGraph GT3X+)'); 'Instrumento' ->")
    print("    instrument_type (etiqueta breve p.ej. 'objective').")
    print("  - BASELINE 'tipo' (p.ej. 'MVPA', 'walk') -> outcome_category.")
    print("  - BASELINE 'Tipo' (p.ej. 'Cuantitativa') NO se mapea; categoria")
    print("    redundante con instrument_type, omitida del esquema canonico.")
    print("  - BF1/BF2/BF3 no tienen instrument/instrument_type/category/")
    print("    primario_secundario/tendencia -> NaN.")
    print("  - 'control_flag' y 'filtro_flag' se preservan como string (los")
    print("    valores son codigos numericos pequenos: 0,1,2,3,4).")
    print("  - IDs con prefijo 'B' (p.ej. '#B10657') se mantienen como filas")
    print("    independientes (NO deduplicar).")


if __name__ == "__main__":
    main()
