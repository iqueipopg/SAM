#!/usr/bin/env python3
"""
build_unified_gold_screening_v2.py — Unified gold CSV v2 (13 cols).

Sprint T16.D.3 / step D.3.B: merge v1 unified gold screening (xlsx + irrelevant
csv) with the PDF-extracted metadata (`extracted_metadata.csv`) and the manual
recovery file (`manual_recovery.csv`), producing the v2 schema with two extra
columns: `source_method` and `source_abstract`.

Outputs:
  - data/gold_standard/deporte/unified/gold_screening_unified.csv (13 cols)
    A pre-overwrite backup `gold_screening_unified.csv.bak_before_d3b` is
    created if it does not already exist; otherwise the script aborts.
  - data/gold_standard/deporte/unified/report_v2.md (Spanish)

Identifiers and comments are in English. Report, logs and user-facing messages
are in Spanish, per project rule #12.

This script does NOT modify v1 (build_unified_gold_screening.py); it imports
its helpers and reuses its xlsx/csv transforms via composition.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Reuse v1 helpers via import (script lives in the same scripts/ directory).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# These imports rely on build_unified_gold_screening.py living alongside us.
from build_unified_gold_screening import (  # noqa: E402
    CANONICAL_COLUMNS as V1_CANONICAL_COLUMNS,
    XLSX_PATH,
    CSV_PATH,
    OUTPUT_DIR,
    _covidence_sort_key,
    _merge_notes,
    _none_if_empty,
    _normalize_doi,
    _parse_year,
    collapse_xlsx,
    transform_csv,
    validate_xlsx_consistency,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXTRACTED_METADATA_PATH = OUTPUT_DIR / "extracted_metadata.csv"
MANUAL_RECOVERY_PATH = OUTPUT_DIR / "manual_recovery.csv"
MANUAL_RECOVERY_MOCK_PATH = OUTPUT_DIR / "manual_recovery.csv.mock"
OUTPUT_CSV = OUTPUT_DIR / "gold_screening_unified.csv"
OUTPUT_BACKUP = OUTPUT_DIR / "gold_screening_unified.csv.bak_before_d3b"
OUTPUT_REPORT_V2 = OUTPUT_DIR / "report_v2.md"

# Canonical 13-column schema for v2 (order matters).
V2_CANONICAL_COLUMNS = [
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

# Sentinels
LOW_CONF_METHOD = "crossref_by_title_low_confidence"
MANUAL_METHOD = "manual"
MANUAL_ABSTRACT_SOURCE = "manual"

# Notes templates (Spanish — these end up in the user-facing CSV `notes` column,
# but they document data provenance so we keep them in Spanish per rule #12).
NOTE_MANUAL = "Metadata anotada manualmente: PDF escaneado sin texto extraíble"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye la versión v2 del gold standard unificado de deporte "
            "(13 columnas), mergeando extracted_metadata + manual_recovery."
        )
    )
    parser.add_argument(
        "--use-mock-manual-recovery",
        action="store_true",
        help=(
            "Usar el fichero mock manual_recovery.csv.mock en lugar del real. "
            "Sólo aplica si manual_recovery.csv NO existe."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_extracted_metadata(path: Path) -> pd.DataFrame:
    """Load extracted_metadata.csv and index by covidence_id.

    Returns a DataFrame indexed by covidence_id (string with '#' prefix).
    """
    if not path.is_file():
        raise FileNotFoundError(f"No existe extracted_metadata.csv: {path}")
    df = pd.read_csv(path)
    required = {
        "covidence_id",
        "title",
        "authors",
        "abstract",
        "doi",
        "year",
        "source_method",
        "source_abstract",
        "warnings",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"extracted_metadata.csv carece de columnas requeridas: {missing}"
        )
    df["covidence_id"] = df["covidence_id"].astype(str).str.strip()
    if df["covidence_id"].duplicated().any():
        dup = df.loc[df["covidence_id"].duplicated(), "covidence_id"].tolist()
        raise RuntimeError(
            f"extracted_metadata.csv contiene covidence_ids duplicados: {dup[:10]}"
        )
    df = df.set_index("covidence_id", drop=False)
    return df


def load_manual_recovery(
    real_path: Path,
    mock_path: Path,
    use_mock: bool,
) -> tuple[pd.DataFrame, str]:
    """Load manual recovery CSV.

    Resolution order:
      - If `real_path` exists -> use it (regardless of --use-mock flag).
      - Else if use_mock and mock_path exists -> use mock.
      - Else -> return an empty DataFrame with the expected columns.

    Returns (df_indexed_by_covidence_id, source_label).
    """
    cols = ["covidence_id", "title", "authors", "abstract", "doi", "year"]
    chosen: Optional[Path] = None
    label = "none"
    if real_path.is_file():
        chosen = real_path
        label = "real"
    elif use_mock and mock_path.is_file():
        chosen = mock_path
        label = "mock"

    if chosen is None:
        empty = pd.DataFrame(columns=cols)
        empty["covidence_id"] = empty["covidence_id"].astype(str)
        empty = empty.set_index("covidence_id", drop=False)
        return empty, label

    df = pd.read_csv(chosen)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"manual_recovery ({chosen.name}) carece de columnas: {missing}"
        )
    df["covidence_id"] = df["covidence_id"].astype(str).str.strip()
    if df["covidence_id"].duplicated().any():
        dup = df.loc[df["covidence_id"].duplicated(), "covidence_id"].tolist()
        raise RuntimeError(
            f"manual_recovery contiene covidence_ids duplicados: {dup}"
        )
    df = df.set_index("covidence_id", drop=False)
    return df, label


# ---------------------------------------------------------------------------
# Warnings parsing
# ---------------------------------------------------------------------------
_LOW_CONF_RE = re.compile(
    r"crossref by title low confidence:\s*score\s*=\s*(\d+)\s*,\s*year_diff\s*=\s*(\d+)",
    re.IGNORECASE,
)


def _parse_low_conf_warning(warnings: object) -> tuple[Optional[int], Optional[int]]:
    """Extract (score, year_diff) from the warnings string of a low-conf row."""
    if warnings is None:
        return None, None
    if isinstance(warnings, float) and pd.isna(warnings):
        return None, None
    s = str(warnings)
    m = _LOW_CONF_RE.search(s)
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except (TypeError, ValueError):
        return None, None


def _format_low_conf_note(score: Optional[int], year_diff: Optional[int]) -> str:
    """Build the Spanish low-confidence note for the `notes` column."""
    score_s = "?" if score is None else str(score)
    yd_s = "?" if year_diff is None else str(year_diff)
    return f"Low-confidence crossref-by-title: score={score_s}, year_diff={yd_s}"


# ---------------------------------------------------------------------------
# Merge logic for xlsx-side rows
# ---------------------------------------------------------------------------
def _coerce_year(raw) -> Optional[int]:
    """Parse year using v1 helper, dropping any unparseable-original string."""
    y, _ = _parse_year(raw)
    return y


def merge_xlsx_rows(
    xlsx_rows: pd.DataFrame,
    extracted: pd.DataFrame,
    manual: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich the 1552 xlsx-side rows with extracted/manual metadata.

    Mutates a copy of xlsx_rows and returns the new DataFrame with the two
    extra v2 columns (`source_method`, `source_abstract`) populated and the
    existing fields (title/authors/abstract/doi/year/notes) updated.
    """
    rows: list[dict] = []
    n_manual = 0
    n_low_conf = 0
    n_extracted_success = 0
    n_missing = 0
    missing_ids: list[str] = []

    for _, base in xlsx_rows.iterrows():
        cid = base["covidence_id"]
        row = base.to_dict()
        # v2-only columns default to None
        row["source_method"] = None
        row["source_abstract"] = None
        existing_note = row.get("notes")

        # 1) Manual recovery first
        if cid in manual.index:
            m = manual.loc[cid]
            title = _none_if_empty(m.get("title"))
            authors = _none_if_empty(m.get("authors"))
            abstract = _none_if_empty(m.get("abstract"))
            doi = _normalize_doi(m.get("doi"))
            year = _coerce_year(m.get("year"))
            row["title"] = title
            row["authors"] = authors
            row["abstract"] = abstract
            row["doi"] = doi
            row["year"] = year
            row["source_method"] = MANUAL_METHOD
            row["source_abstract"] = MANUAL_ABSTRACT_SOURCE if abstract else None
            row["notes"] = _merge_notes([existing_note, NOTE_MANUAL])
            n_manual += 1
            rows.append(row)
            continue

        # 2) Extracted metadata
        if cid not in extracted.index:
            # No data — keep xlsx defaults; this should not happen given the
            # spec but we tolerate it so the script does not abort on partial
            # extracted_metadata.csv during development.
            n_missing += 1
            if len(missing_ids) < 20:
                missing_ids.append(cid)
            rows.append(row)
            continue

        e = extracted.loc[cid]
        sm = _none_if_empty(e.get("source_method"))

        if sm == LOW_CONF_METHOD:
            # Do NOT trust the metadata; leave fields None.
            row["title"] = None
            row["authors"] = None
            row["abstract"] = None
            row["doi"] = None
            row["year"] = None
            row["source_method"] = LOW_CONF_METHOD
            row["source_abstract"] = None
            score, year_diff = _parse_low_conf_warning(e.get("warnings"))
            row["notes"] = _merge_notes(
                [existing_note, _format_low_conf_note(score, year_diff)]
            )
            n_low_conf += 1
            rows.append(row)
            continue

        # 3) Success: pdf_metadata / doi_crossref / pdfplumber_fallback
        row["title"] = _none_if_empty(e.get("title"))
        row["authors"] = _none_if_empty(e.get("authors"))
        row["abstract"] = _none_if_empty(e.get("abstract"))
        row["doi"] = _normalize_doi(e.get("doi"))
        row["year"] = _coerce_year(e.get("year"))
        row["source_method"] = sm
        row["source_abstract"] = _none_if_empty(e.get("source_abstract"))
        # notes: keep existing (e.g. 'PDF descarga falló' for #32107) untouched
        row["notes"] = existing_note
        n_extracted_success += 1
        rows.append(row)

    df = pd.DataFrame(rows, columns=V2_CANONICAL_COLUMNS)

    # Stash a small stats dict on the frame for the report.
    df.attrs["xlsx_merge_stats"] = {
        "n_manual": n_manual,
        "n_low_conf": n_low_conf,
        "n_extracted_success": n_extracted_success,
        "n_missing_in_extracted": n_missing,
        "missing_ids_sample": missing_ids,
    }
    return df


# ---------------------------------------------------------------------------
# v2 row builder for csv-side rows (no extracted metadata applies)
# ---------------------------------------------------------------------------
def upgrade_csv_rows_to_v2(csv_rows: pd.DataFrame) -> pd.DataFrame:
    """Add `source_method`/`source_abstract` (both None) to the csv-side rows."""
    out = csv_rows.copy()
    out["source_method"] = None
    out["source_abstract"] = None
    # Reorder
    return out[V2_CANONICAL_COLUMNS]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_unified_v2(unified: pd.DataFrame) -> None:
    if unified.shape != (20750, 13):
        raise RuntimeError(
            f"Shape v2 inesperado: {unified.shape} (esperado (20750, 13))"
        )
    if list(unified.columns) != V2_CANONICAL_COLUMNS:
        raise RuntimeError(
            "Orden de columnas v2 incorrecto:\n"
            f"  got     = {list(unified.columns)}\n"
            f"  expected= {V2_CANONICAL_COLUMNS}"
        )
    if unified["covidence_id"].duplicated().any():
        dup = unified.loc[unified["covidence_id"].duplicated(), "covidence_id"].tolist()
        raise RuntimeError(f"covidence_id duplicado tras union v2: {dup[:10]}")


# ---------------------------------------------------------------------------
# Report v2 (Spanish)
# ---------------------------------------------------------------------------
def _pct(n: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * n / total):.2f}%"


def build_report_v2(
    unified: pd.DataFrame,
    extracted: pd.DataFrame,
    manual: pd.DataFrame,
    manual_source_label: str,
    xlsx_stats: dict,
) -> str:
    """Compose the Spanish markdown report for v2."""
    xlsx_side = unified[unified["source"] == "xlsx"]
    csv_side = unified[unified["source"] == "csv"]
    n_xlsx = len(xlsx_side)
    n_csv = len(csv_side)
    n_total = len(unified)

    # Counts for source_method (xlsx-side only — csv-side rows have None)
    sm_counts: Counter = Counter()
    for v in xlsx_side["source_method"].tolist():
        sm_counts[v if v else "None"] += 1

    sa_counts: Counter = Counter()
    for v in xlsx_side["source_abstract"].tolist():
        sa_counts[v if v else "None"] += 1

    # Low-confidence detail list from extracted_metadata
    low_conf_df = extracted[extracted["source_method"] == LOW_CONF_METHOD]
    low_conf_rows: list[dict] = []
    for _, e in low_conf_df.iterrows():
        cid = str(e["covidence_id"]).strip()
        score, year_diff = _parse_low_conf_warning(e.get("warnings"))
        low_conf_rows.append(
            {
                "covidence_id": cid,
                "score": "?" if score is None else score,
                "year_diff": "?" if year_diff is None else year_diff,
            }
        )
    low_conf_rows.sort(key=lambda r: _covidence_sort_key(r["covidence_id"]))

    # Manual recoveries
    manual_ids = sorted(manual["covidence_id"].astype(str).tolist(),
                        key=_covidence_sort_key) if len(manual) else []

    lines: list[str] = []
    lines.append("# Reporte v2 — Gold standard deporte (D.3.B)")
    lines.append("")
    lines.append(
        "Versión v2 del CSV canónico (13 columnas) generada al mergear el "
        "unified v1 con `extracted_metadata.csv` (PDF + DOI + fallbacks) y "
        "`manual_recovery.csv` (anotaciones humanas para PDFs escaneados)."
    )
    lines.append("")
    lines.append("## 0. Fuentes de entrada")
    lines.append("")
    lines.append(f"- XLSX (relevant T/A): `{XLSX_PATH.name}` → {n_xlsx} filas")
    lines.append(f"- CSV (irrelevant T/A): `{CSV_PATH.name}` → {n_csv} filas")
    lines.append(
        f"- Metadata extraída: `extracted_metadata.csv` → {len(extracted)} filas"
    )
    lines.append(
        f"- Recovery manual: fuente=`{manual_source_label}` → "
        f"{len(manual)} filas"
    )
    lines.append(f"- **Total unificado v2**: {n_total} filas × 13 columnas")
    lines.append("")

    # 1. Cobertura por source_method (solo xlsx-side)
    lines.append("## 1. Cobertura por `source_method` (filas xlsx-side, n=1552)")
    lines.append("")
    lines.append("| source_method | Conteo | % |")
    lines.append("|---|---:|---:|")
    sm_order = [
        "pdf_metadata",
        "doi_crossref",
        "pdfplumber_fallback",
        LOW_CONF_METHOD,
        MANUAL_METHOD,
        "None",
    ]
    seen = set()
    for k in sm_order:
        if k in sm_counts:
            lines.append(f"| `{k}` | {sm_counts[k]} | {_pct(sm_counts[k], n_xlsx)} |")
            seen.add(k)
    for k in sorted(sm_counts.keys()):
        if k not in seen:
            lines.append(f"| `{k}` | {sm_counts[k]} | {_pct(sm_counts[k], n_xlsx)} |")
    lines.append("")

    # 2. Cobertura por source_abstract
    lines.append("## 2. Cobertura por `source_abstract` (filas xlsx-side, n=1552)")
    lines.append("")
    lines.append("| source_abstract | Conteo | % |")
    lines.append("|---|---:|---:|")
    sa_order = [
        "semantic_scholar",
        "crossref",
        "pdfplumber_fallback",
        MANUAL_ABSTRACT_SOURCE,
        "None",
    ]
    seen = set()
    for k in sa_order:
        if k in sa_counts:
            lines.append(f"| `{k}` | {sa_counts[k]} | {_pct(sa_counts[k], n_xlsx)} |")
            seen.add(k)
    for k in sorted(sa_counts.keys()):
        if k not in seen:
            lines.append(f"| `{k}` | {sa_counts[k]} | {_pct(sa_counts[k], n_xlsx)} |")
    lines.append("")

    # 3. Lista detallada de low-confidence crossref
    lines.append(
        f"## 3. Low-confidence crossref-by-title (n={len(low_conf_rows)})"
    )
    lines.append("")
    lines.append(
        "Estos covidence_ids tienen metadata candidata desde Crossref por "
        "búsqueda por título, pero con confianza baja (score < umbral). En el "
        "v2 se mantienen `title/authors/abstract/doi/year = None` y se anota "
        "el motivo en `notes`."
    )
    lines.append("")
    lines.append("| covidence_id | score | year_diff |")
    lines.append("|---|---:|---:|")
    for r in low_conf_rows:
        lines.append(f"| {r['covidence_id']} | {r['score']} | {r['year_diff']} |")
    lines.append("")

    # 4. Recuperaciones manuales
    lines.append("## 4. Recuperaciones manuales")
    lines.append("")
    if manual_ids:
        lines.append(
            f"Fuente: `{manual_source_label}` "
            f"({'mock placeholder' if manual_source_label == 'mock' else 'fichero real'}). "
            "Justificación: PDF escaneado sin texto extraíble."
        )
        lines.append("")
        lines.append("| covidence_id | título | autores | doi | año |")
        lines.append("|---|---|---|---|---:|")
        for cid in manual_ids:
            row = manual.loc[cid]
            title = _none_if_empty(row.get("title")) or ""
            authors = _none_if_empty(row.get("authors")) or ""
            doi = _none_if_empty(row.get("doi")) or ""
            year = _none_if_empty(row.get("year")) or ""
            lines.append(f"| {cid} | {title} | {authors} | {doi} | {year} |")
    else:
        lines.append("_No hay recuperaciones manuales en esta corrida._")
    lines.append("")

    # 5. Estadísticas del merge
    lines.append("## 5. Estadísticas del merge xlsx-side")
    lines.append("")
    lines.append(f"- Manual recovery aplicado: {xlsx_stats['n_manual']}")
    lines.append(
        f"- Low-confidence crossref preservado sin metadata: "
        f"{xlsx_stats['n_low_conf']}"
    )
    lines.append(
        f"- Extraído con éxito (pdf_metadata / doi_crossref / "
        f"pdfplumber_fallback): {xlsx_stats['n_extracted_success']}"
    )
    lines.append(
        f"- Sin entrada en extracted_metadata.csv: "
        f"{xlsx_stats['n_missing_in_extracted']}"
    )
    if xlsx_stats["missing_ids_sample"]:
        sample = ", ".join(xlsx_stats["missing_ids_sample"])
        lines.append(f"  - Muestra: {sample}")
    lines.append("")

    # 6. Notas de procedencia (clarifica para reviewers humanos)
    lines.append("## 6. Notas")
    lines.append("")
    lines.append(
        "- Las 19198 filas csv-side mantienen `source_method = None` y "
        "`source_abstract = None`: su metadata proviene directamente del "
        "export de Covidence."
    )
    lines.append(
        "- La normalización de DOI se re-aplica en el merge por seguridad "
        "aunque `extracted_metadata.csv` ya viene normalizado."
    )
    lines.append(
        "- El backup pre-overwrite `gold_screening_unified.csv.bak_before_d3b` "
        "preserva el estado v1 (11 columnas) para diff/rollback."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # 0) Backup guard — must run BEFORE we overwrite the v1 file.
    if OUTPUT_CSV.exists():
        if OUTPUT_BACKUP.exists():
            print(
                f"ERROR: el backup ya existe en {OUTPUT_BACKUP}.\n"
                "       Aborto para no sobreescribir un backup previo. "
                "Bórralo o renómbralo manualmente si quieres regenerar v2.",
                file=sys.stderr,
            )
            return 2
        shutil.copy2(OUTPUT_CSV, OUTPUT_BACKUP)
        print(f"[0/7] Backup v1 → {OUTPUT_BACKUP.name}")
    else:
        print("[0/7] No existe v1 previo; se omite backup.")

    # 1) Load xlsx + csv (v1 sources)
    print(f"[1/7] Leyendo XLSX: {XLSX_PATH.name}")
    xlsx_df = pd.read_excel(XLSX_PATH)
    validate_xlsx_consistency(xlsx_df)
    xlsx_rows_v1 = collapse_xlsx(xlsx_df)
    print(f"      filas colapsadas={len(xlsx_rows_v1)}")

    print(f"[2/7] Leyendo CSV irrelevantes: {CSV_PATH.name}")
    csv_df = pd.read_csv(CSV_PATH)
    csv_rows_v1 = transform_csv(csv_df)
    print(f"      filas={len(csv_rows_v1)}")

    # 2) Load extracted metadata + manual recovery
    print(f"[3/7] Leyendo extracted_metadata.csv ({EXTRACTED_METADATA_PATH.name})")
    extracted = load_extracted_metadata(EXTRACTED_METADATA_PATH)
    print(f"      filas={len(extracted)}")

    print("[4/7] Cargando manual_recovery (real | mock | none)")
    manual, manual_label = load_manual_recovery(
        MANUAL_RECOVERY_PATH,
        MANUAL_RECOVERY_MOCK_PATH,
        use_mock=args.use_mock_manual_recovery,
    )
    print(f"      fuente={manual_label}, filas={len(manual)}")
    if manual_label == "none" and args.use_mock_manual_recovery:
        print(
            "      AVISO: --use-mock-manual-recovery activo pero no se "
            "encontró ni real ni mock.",
            file=sys.stderr,
        )

    # 3) Merge xlsx-side rows with extracted + manual
    print("[5/7] Mergeando xlsx-side (1552) con extracted + manual...")
    xlsx_rows_v2 = merge_xlsx_rows(xlsx_rows_v1, extracted, manual)
    stats = xlsx_rows_v2.attrs.get("xlsx_merge_stats", {})
    print(
        f"      manual={stats.get('n_manual', 0)}, "
        f"low_conf={stats.get('n_low_conf', 0)}, "
        f"extracted_ok={stats.get('n_extracted_success', 0)}, "
        f"missing_in_extracted={stats.get('n_missing_in_extracted', 0)}"
    )

    # 4) Upgrade csv-side to v2 (just adds two None columns)
    csv_rows_v2 = upgrade_csv_rows_to_v2(csv_rows_v1)

    # 5) Concatenate + sort + validate
    print("[6/7] Concatenando y validando...")
    for col in V2_CANONICAL_COLUMNS:
        if col != "year":
            xlsx_rows_v2[col] = xlsx_rows_v2[col].astype("object")
            csv_rows_v2[col] = csv_rows_v2[col].astype("object")
    unified = pd.concat([xlsx_rows_v2, csv_rows_v2], ignore_index=True)
    unified = unified.sort_values(
        by="covidence_id", key=lambda s: s.map(_covidence_sort_key)
    ).reset_index(drop=True)
    validate_unified_v2(unified)
    print(f"      shape={unified.shape}, columnas={len(unified.columns)}")

    # 6) Write outputs
    print("[7/7] Escribiendo CSV v2 + report_v2.md")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unified.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    report = build_report_v2(
        unified=unified,
        extracted=extracted,
        manual=manual,
        manual_source_label=manual_label,
        xlsx_stats=stats,
    )
    OUTPUT_REPORT_V2.write_text(report, encoding="utf-8")

    print("")
    print("=== RESUMEN v2 ===")
    print(f"CSV v2     : {OUTPUT_CSV}")
    print(f"  shape     = {unified.shape}")
    print(f"  size      = {OUTPUT_CSV.stat().st_size} bytes")
    print(f"Backup v1  : {OUTPUT_BACKUP if OUTPUT_BACKUP.exists() else '(n/a)'}")
    print(f"Reporte v2 : {OUTPUT_REPORT_V2}")
    print(f"  size      = {OUTPUT_REPORT_V2.stat().st_size} bytes")
    print("")
    print("source_method (xlsx-side):")
    sm_counts = unified.loc[unified["source"] == "xlsx", "source_method"].value_counts(
        dropna=False
    )
    for k, v in sm_counts.items():
        print(f"  {k!s:40s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
