#!/usr/bin/env python3
# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
build_unified_gold_screening.py — Unify Covidence deporte gold-standard exports.

Combines two Covidence exports for SAM's deporte gold standard into a single
canonical CSV (11 columns, 20750 rows):
  - Resumen_Total_PDFs.xlsx (1552 studies relevant at title/abstract; FT decisions)
  - review_163210_irrelevant_csv_*.csv (19198 studies irrelevant at title/abstract)

Outputs:
  - data/gold_standard/deporte/unified/gold_screening_unified.csv
  - data/gold_standard/deporte/unified/report.md  (Spanish)

Identifiers and comments are in English. Report/outputs are in Spanish
per project rule #12.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPORTE_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte"
XLSX_PATH = DEPORTE_DIR / "Resumen_Total_PDFs.xlsx"
CSV_PATH = DEPORTE_DIR / "review_163210_irrelevant_csv_20260513173759.csv"
OUTPUT_DIR = DEPORTE_DIR / "unified"
OUTPUT_CSV = OUTPUT_DIR / "gold_screening_unified.csv"
OUTPUT_REPORT = OUTPUT_DIR / "report.md"

# Canonical exclusion-reason mapping (xlsx -> canonical).
EXCLUSION_REASON_MAP = {
    "Excluido (sin motivo registrado)": "Sin motivo registrado",
}

# Special-case notes
NOTE_PDF_DOWNLOAD_FAILED = "PDF descarga falló"

# DOI prefixes to strip during normalization (order matters: longest first).
DOI_STRIP_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "doi:",
)

CANONICAL_COLUMNS = [
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
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _none_if_empty(value) -> Optional[str]:
    """Return None for NaN / empty / whitespace-only strings; else stripped str."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def _normalize_doi(raw) -> Optional[str]:
    """Normalize DOI: lowercase, strip known prefixes and whitespace.

    Returns None if the raw value is empty or normalizes to an empty string.
    """
    s = _none_if_empty(raw)
    if s is None:
        return None
    s = s.lower().strip()
    # Strip prefixes iteratively (cover compositions like 'https://doi.org/doi:').
    changed = True
    while changed:
        changed = False
        for prefix in DOI_STRIP_PREFIXES:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                changed = True
    return s or None


def _parse_year(raw) -> tuple[Optional[int], Optional[str]]:
    """Parse a year value to int.

    Returns (year_int_or_None, original_unparseable_str_or_None).
    Handles float-like values (e.g. 2019.0 -> 2019).
    """
    if raw is None:
        return None, None
    if isinstance(raw, float) and pd.isna(raw):
        return None, None
    # Pure ints
    if isinstance(raw, int):
        return int(raw), None
    # Floats: only accept if it represents an integer year (e.g. 2019.0)
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw), None
        return None, str(raw)
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None, None
    # Try pure int
    try:
        return int(s), None
    except ValueError:
        pass
    # Try float that is integer-valued
    try:
        f = float(s)
        if f.is_integer():
            return int(f), None
    except ValueError:
        pass
    return None, s


def _covidence_sort_key(cid: str) -> int:
    """Extract numeric part of a covidence id like '#3647' for sorting."""
    return int(str(cid).lstrip("#"))


def _merge_notes(parts: Iterable[Optional[str]]) -> Optional[str]:
    cleaned = [p for p in parts if p]
    if not cleaned:
        return None
    return "; ".join(cleaned)


# ---------------------------------------------------------------------------
# XLSX (relevant T/A; FT decisions)
# ---------------------------------------------------------------------------
def validate_xlsx_consistency(df: pd.DataFrame) -> None:
    """Abort if any covidence_id has multiple Clasificacion or Motivo values."""
    grouped = df.groupby("ID del Estudio (Covidence)").agg(
        clasif_n=("Clasificación", "nunique"),
        motivo_n=("Motivo de Exclusión", "nunique"),
    )
    bad_clasif = grouped[grouped["clasif_n"] > 1].index.tolist()
    bad_motivo = grouped[grouped["motivo_n"] > 1].index.tolist()
    if bad_clasif or bad_motivo:
        msg = ["XLSX consistency check FAILED:"]
        if bad_clasif:
            msg.append(f"  - IDs con multiples Clasificacion: {bad_clasif}")
        if bad_motivo:
            msg.append(f"  - IDs con multiples Motivo: {bad_motivo}")
        raise RuntimeError("\n".join(msg))


def collapse_xlsx(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce xlsx to one row per covidence_id, building canonical schema.

    Notes:
      - xlsx has no title/authors/abstract/doi/year columns -> all None.
      - human_label_ta = "relevant" (xlsx only carries the 1552 relevant T/A).
      - human_label_ft = "include" if Clasificacion="Incluido", else "exclude".
      - exclusion_reason_ft: None for include, else mapped Motivo (None if include).
      - notes: "PDF descarga falló" for the single Estado="Falló HTTP" row.
    """
    # Collect the set of covidence_ids that had ANY 'Falló HTTP' row.
    failed_ids = set(
        df.loc[df["Estado"] == "Falló HTTP", "ID del Estudio (Covidence)"].unique()
    )

    first = df.drop_duplicates("ID del Estudio (Covidence)", keep="first").copy()
    rows: list[dict] = []
    for _, r in first.iterrows():
        cid = str(r["ID del Estudio (Covidence)"]).strip()
        clasif = _none_if_empty(r["Clasificación"])
        motivo = _none_if_empty(r["Motivo de Exclusión"])
        if clasif == "Incluido":
            label_ft = "include"
            reason = None
        elif clasif == "Excluido":
            label_ft = "exclude"
            # Map canonical reason; if missing, use "Sin motivo registrado".
            if motivo is None:
                reason = "Sin motivo registrado"
            else:
                reason = EXCLUSION_REASON_MAP.get(motivo, motivo)
        else:
            raise RuntimeError(
                f"Clasificación inesperada para {cid}: {clasif!r}"
            )
        notes = NOTE_PDF_DOWNLOAD_FAILED if cid in failed_ids else None
        rows.append(
            {
                "covidence_id": cid,
                "title": None,
                "authors": None,
                "abstract": None,
                "doi": None,
                "year": None,
                "human_label_ta": "relevant",
                "human_label_ft": label_ft,
                "exclusion_reason_ft": reason,
                "source": "xlsx",
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


# ---------------------------------------------------------------------------
# CSV (irrelevant T/A)
# ---------------------------------------------------------------------------
def transform_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the irrelevant-export CSV into canonical schema rows."""
    rows: list[dict] = []
    for _, r in df.iterrows():
        cid = str(r["Covidence #"]).strip()
        year_int, year_original = _parse_year(r["Published Year"])
        notes_parts: list[Optional[str]] = []
        if year_original is not None:
            notes_parts.append(f"year original: {year_original}")
        rows.append(
            {
                "covidence_id": cid,
                "title": _none_if_empty(r["Title"]),
                "authors": _none_if_empty(r["Authors"]),
                "abstract": _none_if_empty(r["Abstract"]),
                "doi": _normalize_doi(r["DOI"]),
                "year": year_int,
                "human_label_ta": "irrelevant",
                "human_label_ft": None,
                "exclusion_reason_ft": None,
                "source": "csv",
                "notes": _merge_notes(notes_parts),
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


# ---------------------------------------------------------------------------
# DOI duplicate analysis
# ---------------------------------------------------------------------------
def _raw_doi(raw) -> Optional[str]:
    """Trim whitespace only, preserve case. Empty/NaN -> None."""
    return _none_if_empty(raw)


def analyze_doi_duplicates(csv_df_raw: pd.DataFrame, unified_df: pd.DataFrame) -> dict:
    """Compute raw vs normalized DOI duplicate stats and collision groups.

    csv_df_raw is the source CSV (so we can extract raw DOI strings).
    unified_df gives normalized DOIs aligned to covidence_id.
    """
    raw_pairs: list[tuple[str, str]] = []  # (raw_doi, covidence_id)
    norm_pairs: list[tuple[str, str]] = []  # (norm_doi, covidence_id)
    raw_to_norm: dict[str, str] = {}        # last-seen mapping for display
    cid_to_raw: dict[str, str] = {}

    for _, r in csv_df_raw.iterrows():
        cid = str(r["Covidence #"]).strip()
        raw = _raw_doi(r["DOI"])
        norm = _normalize_doi(r["DOI"])
        if raw is not None:
            raw_pairs.append((raw, cid))
            cid_to_raw[cid] = raw
        if norm is not None:
            norm_pairs.append((norm, cid))
            if raw is not None:
                raw_to_norm[raw] = norm

    n_with_doi = len(raw_pairs)
    raw_unique = len(set(p[0] for p in raw_pairs))
    norm_unique = len(set(p[0] for p in norm_pairs))
    raw_dup_extra = n_with_doi - raw_unique
    norm_dup_extra = n_with_doi - norm_unique

    # Norm duplicate groups: all norm DOIs appearing >1 time
    norm_groups: dict[str, list[str]] = defaultdict(list)
    for norm, cid in norm_pairs:
        norm_groups[norm].append(cid)
    norm_duplicates = {k: v for k, v in norm_groups.items() if len(v) > 1}

    # Raw duplicate groups (for revealed-by-normalization analysis)
    raw_groups: dict[str, list[str]] = defaultdict(list)
    for raw, cid in raw_pairs:
        raw_groups[raw].append(cid)
    raw_duplicates = {k: v for k, v in raw_groups.items() if len(v) > 1}

    # "Collisions revealed by normalization": norm-duplicate groups whose
    # corresponding raw forms were NOT all duplicates of a single raw key
    # (i.e. the raw forms of the cids in this group split into multiple raw
    # groups, none of which alone would have flagged the duplication).
    revealed_groups: dict[str, dict] = {}
    for norm_key, cids in norm_duplicates.items():
        raw_forms = {cid: cid_to_raw.get(cid) for cid in cids}
        distinct_raw = sorted(set(v for v in raw_forms.values() if v is not None))
        # If the cids do NOT share a single raw form, normalization revealed
        # the collision (their differences were only prefix/case/spacing).
        if len(distinct_raw) > 1:
            revealed_groups[norm_key] = {
                "covidence_ids": sorted(cids, key=_covidence_sort_key),
                "raw_forms": distinct_raw,
            }

    return {
        "n_with_doi": n_with_doi,
        "raw_unique": raw_unique,
        "norm_unique": norm_unique,
        "raw_dup_extra": raw_dup_extra,
        "norm_dup_extra": norm_dup_extra,
        "norm_duplicates": norm_duplicates,           # all 305 groups
        "raw_duplicates": raw_duplicates,
        "revealed_groups": revealed_groups,           # the 26 revealed
        "cid_to_raw": cid_to_raw,
    }


# ---------------------------------------------------------------------------
# Validation of final unified table
# ---------------------------------------------------------------------------
def validate_unified(unified: pd.DataFrame) -> None:
    if unified.shape != (20750, 11):
        raise RuntimeError(
            f"Unified shape mismatch: {unified.shape} (esperado (20750, 11))"
        )
    if unified["covidence_id"].duplicated().any():
        dup = unified[unified["covidence_id"].duplicated()]["covidence_id"].tolist()
        raise RuntimeError(f"covidence_id duplicado tras union: {dup[:10]}")
    # Overlap check: ids from xlsx vs csv
    xlsx_ids = set(unified.loc[unified["source"] == "xlsx", "covidence_id"])
    csv_ids = set(unified.loc[unified["source"] == "csv", "covidence_id"])
    overlap = xlsx_ids & csv_ids
    if overlap:
        raise RuntimeError(f"Overlap xlsx/csv: {sorted(overlap)[:10]}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(unified: pd.DataFrame, doi_stats: dict, anomalous_years: pd.DataFrame) -> str:
    """Compose the Spanish markdown report."""
    total = len(unified)
    n_relevant = int((unified["human_label_ta"] == "relevant").sum())
    n_irrelevant = int((unified["human_label_ta"] == "irrelevant").sum())
    n_include_ft = int((unified["human_label_ft"] == "include").sum())
    n_exclude_ft = int((unified["human_label_ft"] == "exclude").sum())
    n_none_ft = int(unified["human_label_ft"].isna().sum())

    reason_counts = (
        unified.loc[unified["exclusion_reason_ft"].notna(), "exclusion_reason_ft"]
        .value_counts()
        .to_dict()
    )

    n_no_doi = int(unified["doi"].isna().sum())
    n_no_abstract = int(unified["abstract"].isna().sum())
    n_no_title = int(unified["title"].isna().sum())

    # Temporal distribution
    years_series = unified["year"].dropna().astype(int)
    year_counts = years_series.value_counts().sort_index()

    cid_unique = "sí" if not unified["covidence_id"].duplicated().any() else "no"
    xlsx_ids = set(unified.loc[unified["source"] == "xlsx", "covidence_id"])
    csv_ids = set(unified.loc[unified["source"] == "csv", "covidence_id"])
    overlap_ok = "sí" if not (xlsx_ids & csv_ids) else "no"

    lines: list[str] = []
    lines.append("# Reporte de unificación — Gold standard deporte")
    lines.append("")
    lines.append(
        "CSV canónico generado a partir de los dos exports de Covidence "
        "(`Resumen_Total_PDFs.xlsx` + `review_163210_irrelevant_csv_*.csv`)."
    )
    lines.append("")

    # 1. Validation counts
    lines.append("## 1. Cifras de validación")
    lines.append("")
    lines.append("| Métrica | Esperado | Observado |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Total filas | 20750 | {total} |")
    lines.append(f"| relevant T/A | 1552 | {n_relevant} |")
    lines.append(f"| irrelevant T/A | 19198 | {n_irrelevant} |")
    lines.append(f"| include FT | 66 | {n_include_ft} |")
    lines.append(f"| exclude FT | 1486 | {n_exclude_ft} |")
    lines.append(f"| None FT | 19198 | {n_none_ft} |")
    lines.append(f"| covidence_id único | sí | {cid_unique} |")
    lines.append(f"| 0 overlap xlsx/csv | sí | {overlap_ok} |")
    lines.append("")
    lines.append("### Distribución de `exclusion_reason_ft` (estudios excluidos en FT)")
    lines.append("")
    lines.append("| Motivo | Conteo |")
    lines.append("|---|---:|")
    # Use a fixed order matching the spec
    reason_order = [
        "Wrong age",
        "Wrong outcome",
        "Wrong study design",
        "Wrong intervention",
        "Wrong publication type",
        "Wrong population",
        "Language",
        "Sin motivo registrado",
    ]
    for k in reason_order:
        lines.append(f"| {k} | {reason_counts.get(k, 0)} |")
    # Add any extras not in the canonical list
    extras = {k: v for k, v in reason_counts.items() if k not in reason_order}
    for k in sorted(extras):
        lines.append(f"| {k} | {extras[k]} |")
    lines.append("")

    # 2. DOI duplicates
    lines.append("## 2. DOIs duplicados")
    lines.append("")
    lines.append(
        f"- Filas con DOI: **{doi_stats['n_with_doi']}**"
    )
    lines.append(
        f"- Raw (sin normalizar): **{doi_stats['raw_unique']}** únicos → "
        f"**{doi_stats['raw_dup_extra']}** filas duplicadas extra"
    )
    lines.append(
        f"- Post-normalización: **{doi_stats['norm_unique']}** únicos → "
        f"**{doi_stats['norm_dup_extra']}** filas duplicadas extra, "
        f"**{len(doi_stats['norm_duplicates'])}** DOIs distintos repetidos"
    )
    revealed_n = doi_stats["norm_dup_extra"] - doi_stats["raw_dup_extra"]
    lines.append(
        f"- **Colisiones reveladas por la normalización**: "
        f"{doi_stats['norm_dup_extra']} − {doi_stats['raw_dup_extra']} = "
        f"**{revealed_n}**"
    )
    lines.append("")
    lines.append(
        "### 2.1 Las 26 colisiones reveladas por la normalización"
    )
    lines.append("")
    lines.append(
        "Estos pares de filas comparten DOI tras normalizar prefijos / "
        "espacios / mayúsculas, pero sus formas crudas eran distintas, por "
        "lo que un dedup sobre el campo crudo no las habría detectado."
    )
    lines.append("")
    lines.append("| DOI normalizado | Covidence IDs | Formas raw distintas |")
    lines.append("|---|---|---|")
    for norm_key in sorted(doi_stats["revealed_groups"].keys()):
        info = doi_stats["revealed_groups"][norm_key]
        cids = ", ".join(info["covidence_ids"])
        raws = " ⟂ ".join(f"`{r}`" for r in info["raw_forms"])
        lines.append(f"| `{norm_key}` | {cids} | {raws} |")
    lines.append("")
    lines.append(
        "### 2.2 Tabla completa de DOIs duplicados (post-normalización)"
    )
    lines.append("")
    lines.append(
        f"Total: {len(doi_stats['norm_duplicates'])} DOIs con más de un "
        "covidence_id asociado."
    )
    lines.append("")
    lines.append("| DOI normalizado | Nº filas | Covidence IDs |")
    lines.append("|---|---:|---|")
    # Order by descending group size, then by DOI
    dup_sorted = sorted(
        doi_stats["norm_duplicates"].items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    for norm_key, cids in dup_sorted:
        cids_sorted = sorted(cids, key=_covidence_sort_key)
        cids_str = ", ".join(cids_sorted)
        lines.append(f"| `{norm_key}` | {len(cids)} | {cids_str} |")
    lines.append("")

    # 3. Metadata coverage
    lines.append("## 3. Cobertura de metadatos")
    lines.append("")
    lines.append(f"- Sin DOI: **{n_no_doi}** filas")
    lines.append(f"- Sin abstract: **{n_no_abstract}** filas")
    lines.append(f"- Sin título: **{n_no_title}** filas")
    lines.append("")
    lines.append(
        "_Nota_: las 1552 filas procedentes del xlsx no traen `title`, "
        "`authors`, `abstract`, `doi` ni `year` (el export de Covidence sólo "
        "incluye el covidence_id y la decisión)."
    )
    lines.append("")

    # 4. Temporal distribution
    lines.append("## 4. Distribución temporal")
    lines.append("")
    lines.append(
        "Rango temporal de las 19198 filas con metadatos (csv). 2021 es un "
        "año parcial (export ejecutado mid-2021)."
    )
    lines.append("")
    lines.append("| Año | Conteo |")
    lines.append("|---:|---:|")
    for y, c in year_counts.items():
        lines.append(f"| {int(y)} | {int(c)} |")
    lines.append(f"| _año=None_ | {int(unified['year'].isna().sum())} |")
    lines.append("")

    # 5. Documented decisions
    lines.append("## 5. Decisiones documentadas")
    lines.append("")
    lines.append("### 5.1 `#32107` — Falló HTTP")
    lines.append("")
    lines.append(
        "El único estudio con `Estado=\"Falló HTTP\"` en el xlsx. Se mantiene "
        "como `human_label_ft=\"exclude\"` y `exclusion_reason_ft="
        "\"Wrong publication type\"` (su Motivo registrado), y se añade "
        f"`notes=\"{NOTE_PDF_DOWNLOAD_FAILED}\"` para trazabilidad."
    )
    lines.append("")
    lines.append("### 5.2 Years anómalos (patrón YYYY-YYYY)")
    lines.append("")
    lines.append(
        "4 filas del csv tienen `Published Year` con patrón YYYY-YYYY no "
        "parseable a int. Regla aplicada: `year=None`, valor original "
        "preservado en `notes`."
    )
    lines.append("")
    lines.append("| covidence_id | year original | notes |")
    lines.append("|---|---|---|")
    for _, r in anomalous_years.iterrows():
        lines.append(
            f"| {r['covidence_id']} | {r['original']} | "
            f"`year original: {r['original']}` |"
        )
    lines.append("")

    # 6. Anomalies
    lines.append("## 6. Anomalías encontradas durante el proceso")
    lines.append("")
    lines.append(
        "- 0 inconsistencias internas en el xlsx: cada covidence_id agrupa "
        "una sola Clasificación y un solo Motivo de Exclusión a lo largo "
        "de sus 1–5 filas."
    )
    lines.append(
        "- 0 overlap de covidence_id entre xlsx (1552) y csv (19198) → "
        "20750 ids únicos totales."
    )
    lines.append(
        f"- {revealed_n} colisiones de DOI sólo visibles tras normalizar "
        "prefijos (`https://doi.org/`, `doi:`, etc.) y mayúsculas. Detalle "
        "en §2.1."
    )
    lines.append(
        "- 5 filas del xlsx con `Motivo de Exclusión = \"Excluido (sin "
        "motivo registrado)\"` (4 estudios únicos) → mapeadas a la categoría "
        "canónica `\"Sin motivo registrado\"`."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[1/6] Leyendo XLSX: {XLSX_PATH}")
    xlsx_df = pd.read_excel(XLSX_PATH)
    print(f"      filas={len(xlsx_df)}, estudios únicos="
          f"{xlsx_df['ID del Estudio (Covidence)'].nunique()}")

    print("[2/6] Validando consistencia interna del xlsx...")
    validate_xlsx_consistency(xlsx_df)
    print("      OK: 0 inconsistencias.")

    print(f"[3/6] Leyendo CSV: {CSV_PATH}")
    csv_df = pd.read_csv(CSV_PATH)
    print(f"      filas={len(csv_df)}, cols={len(csv_df.columns)}")

    print("[4/6] Construyendo schema canónico...")
    xlsx_rows = collapse_xlsx(xlsx_df)
    csv_rows = transform_csv(csv_df)
    # Coerce to identical dtypes to silence pandas FutureWarning on concat
    # of frames where some columns are all-None.
    for col in CANONICAL_COLUMNS:
        if col != "year":
            xlsx_rows[col] = xlsx_rows[col].astype("object")
            csv_rows[col] = csv_rows[col].astype("object")
    unified = pd.concat([xlsx_rows, csv_rows], ignore_index=True)
    unified = unified.sort_values(
        by="covidence_id", key=lambda s: s.map(_covidence_sort_key)
    ).reset_index(drop=True)
    print(f"      filas unificadas={len(unified)}")

    print("[5/6] Validando tabla unificada...")
    validate_unified(unified)
    print("      OK: shape (20750, 11), ids únicos, 0 overlap xlsx/csv.")

    print("[6/6] Calculando estadísticas de DOI y escribiendo outputs...")
    doi_stats = analyze_doi_duplicates(csv_df, unified)

    # Build anomalous-years frame from csv source (the 4 dash-pattern rows)
    anomalous_rows = []
    for _, r in csv_df.iterrows():
        _, orig = _parse_year(r["Published Year"])
        if orig is not None and "-" in orig:
            anomalous_rows.append(
                {"covidence_id": str(r["Covidence #"]).strip(), "original": orig}
            )
    anomalous_years = pd.DataFrame(anomalous_rows).sort_values(
        by="covidence_id", key=lambda s: s.map(_covidence_sort_key)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unified.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    report = build_report(unified, doi_stats, anomalous_years)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    print("")
    print("=== RESUMEN ===")
    print(f"CSV unificado : {OUTPUT_CSV}")
    print(f"  shape        = {unified.shape}")
    print(f"  size         = {OUTPUT_CSV.stat().st_size} bytes")
    print(f"Reporte MD    : {OUTPUT_REPORT}")
    print(f"  size         = {OUTPUT_REPORT.stat().st_size} bytes")
    print("")
    print("Cifras clave:")
    print(f"  relevant T/A   = "
          f"{(unified['human_label_ta'] == 'relevant').sum()}")
    print(f"  irrelevant T/A = "
          f"{(unified['human_label_ta'] == 'irrelevant').sum()}")
    print(f"  include FT     = "
          f"{(unified['human_label_ft'] == 'include').sum()}")
    print(f"  exclude FT     = "
          f"{(unified['human_label_ft'] == 'exclude').sum()}")
    print(f"  None FT        = "
          f"{unified['human_label_ft'].isna().sum()}")
    print(f"  DOI raw dup extra        = {doi_stats['raw_dup_extra']}")
    print(f"  DOI norm dup extra       = {doi_stats['norm_dup_extra']}")
    print(f"  DOI revealed collisions  = "
          f"{doi_stats['norm_dup_extra'] - doi_stats['raw_dup_extra']}")
    print(f"  DOIs distintos repetidos = {len(doi_stats['norm_duplicates'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
