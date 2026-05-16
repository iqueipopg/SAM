#!/usr/bin/env python3
"""
build_micro_sample.py — Sample micro estratificado de 50 papers (T16.D.4.B).

Construye el sample reproducible que se usara en T16.D.5 (evaluadores Heroes
1-4) para auditar el pipeline SAM contra el gold standard humano de deporte.

Composicion (50 = 10 includes + 40 excludes FT):
    include                10
    Wrong age              10
    Wrong outcome           8
    Wrong study design      6
    Wrong intervention      6
    Wrong publication type  4
    Wrong population        3
    Language                2
    Sin motivo registrado   1

Razonamiento del pool:
- Solo xlsx-side aporta al sample: las 1552 filas xlsx son las que pasaron
  T/A en Covidence y por tanto recibieron decision FT (include o exclude +
  razon). Las 19198 filas csv-side son los descartados en T/A (irrelevant),
  no tienen `exclusion_reason_ft` ni `human_label_ft`, por lo que no cubren
  ninguno de los estratos FT del sample.
- Se filtran los `source_method ∈ {crossref_by_title_low_confidence, failed}`
  porque su metadata es de baja confianza y podria contaminar la evaluacion.
  Pool xlsx eligible: 1552 - 45 (low_confidence) - 0 (failed) = 1507 filas.

Seed (`SEED = 20260514`) garantiza reproducibilidad byte-a-byte. El script
verifica internamente generando el sample dos veces y abortando si difieren.

Salida:
    data/gold_standard/deporte/samples/micro/sample_50.csv (50 x 10 cols)

Identificadores y comentarios en ingles. Prints y docstrings en espanol
(regla #12 del proyecto).

Uso:
    python scripts/build_micro_sample.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte"
UNIFIED_CSV = GOLD_DIR / "unified" / "gold_screening_unified.csv"
XLSX_PATH = GOLD_DIR / "Resumen_Total_PDFs.xlsx"
PDFS_DIR = GOLD_DIR / "pdfs_todos"
REF_MAPPING_CSV = GOLD_DIR / "unified" / "ref_to_covidence_mapping.csv"
OUTPUT_DIR = GOLD_DIR / "samples" / "micro"
OUTPUT_CSV = OUTPUT_DIR / "sample_50.csv"
OUTPUT_REPORT = OUTPUT_DIR / "sample_report.md"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 20260514

# Strata composition (declaration order matters for output sort).
STRATA_COMPOSITION: Dict[str, int] = {
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

# source_method values that are excluded from the pool (low quality metadata).
EXCLUDED_SOURCE_METHODS = {"crossref_by_title_low_confidence", "failed"}

# Output columns in order.
OUTPUT_COLUMNS = [
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

# Purpose tags by strata.
PURPOSE_INCLUDE = "hero1,hero2,hero3,hero4"
PURPOSE_EXCLUDE = "hero1,hero2"


# ---------------------------------------------------------------------------
# Pure helpers (importables from tests)
# ---------------------------------------------------------------------------
def build_pool(unified_df: pd.DataFrame) -> pd.DataFrame:
    """Return the xlsx-side eligible pool.

    Filters out csv-side rows (they lack FT strata) and rows whose
    `source_method` is in EXCLUDED_SOURCE_METHODS (low-quality metadata).
    NaN `source_method` is kept (xlsx rows without recovered metadata).
    """
    xlsx_side = unified_df[unified_df["source"] == "xlsx"].copy()
    mask = ~xlsx_side["source_method"].isin(EXCLUDED_SOURCE_METHODS)
    return xlsx_side[mask].copy()


def _has_non_empty_abstract(value: object) -> bool:
    """True if `value` is a non-empty, non-NaN string."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return len(str(value).strip()) > 0


def sample_stratum(
    pool: pd.DataFrame,
    strata_name: str,
    n: int,
    rng: random.Random,
    prefer_abstract: bool = False,
) -> pd.DataFrame:
    """Sample `n` rows from the pool matching `strata_name`.

    For strata_name == "include": filter `human_label_ft == "include"`.
    For other strata: filter `exclusion_reason_ft == strata_name` AND
    `human_label_ft == "exclude"`.

    Sampling is deterministic: pool is sorted by covidence_id ASC (or by
    (has_abstract DESC, covidence_id ASC) if prefer_abstract) before
    shuffling with the provided RNG.
    """
    if strata_name == "include":
        subset = pool[pool["human_label_ft"] == "include"].copy()
    else:
        subset = pool[
            (pool["human_label_ft"] == "exclude")
            & (pool["exclusion_reason_ft"] == strata_name)
        ].copy()

    if len(subset) < n:
        raise ValueError(
            f"Pool insuficiente para estrato '{strata_name}': "
            f"{len(subset)} disponibles, {n} requeridos."
        )

    # Soft preference for non-empty abstract (used for includes).
    if prefer_abstract:
        subset["_has_abstract_pref"] = subset["abstract"].apply(_has_non_empty_abstract)
        # Sort: True (1) first via descending. Then covidence_id ASC.
        subset = subset.sort_values(
            by=["_has_abstract_pref", "covidence_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        # Build a deterministic pool then shuffle ONLY within the preferred
        # subset if it already has >= n rows.
        preferred = subset[subset["_has_abstract_pref"]].copy()
        if len(preferred) >= n:
            ordered = preferred.drop(columns=["_has_abstract_pref"]).reset_index(
                drop=True
            )
        else:
            ordered = subset.drop(columns=["_has_abstract_pref"]).reset_index(drop=True)
    else:
        ordered = subset.sort_values(
            by="covidence_id", ascending=True, kind="mergesort"
        ).reset_index(drop=True)

    # Deterministic shuffle using indices and the seeded RNG.
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    picked = indices[:n]
    return ordered.iloc[picked].reset_index(drop=True)


def _normalize_covidence_for_xlsx(covidence_id: str) -> str:
    """Strip leading '#' and whitespace to match xlsx raw IDs.

    The xlsx stores IDs as '#3647'; unified stores them the same way. No
    transformation is currently required, but the helper centralises the
    invariant in case it changes.
    """
    return str(covidence_id).strip()


def compute_has_pdf(
    covidence_id: str,
    xlsx_lookup: Dict[str, List[str]],
    pdfs_existing: set,
) -> bool:
    """True if covidence_id has >=1 xlsx filename present in pdfs_existing."""
    filenames = xlsx_lookup.get(_normalize_covidence_for_xlsx(covidence_id), [])
    for fn in filenames:
        if fn in pdfs_existing:
            return True
    return False


def pick_pdf_filename(
    covidence_id: str,
    xlsx_lookup: Dict[str, List[str]],
    pdfs_existing: set,
) -> Optional[str]:
    """Pick lexicographically-first xlsx filename that exists in pdfs_existing."""
    filenames = sorted(xlsx_lookup.get(_normalize_covidence_for_xlsx(covidence_id), []))
    for fn in filenames:
        if fn in pdfs_existing:
            return fn
    return None


def _build_xlsx_lookup(xlsx_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map covidence_id -> list of unique filenames (non-empty)."""
    lookup: Dict[str, List[str]] = {}
    for _, row in xlsx_df.iterrows():
        cid = str(row["ID del Estudio (Covidence)"]).strip()
        fn = row["Nombre del Archivo Descargado"]
        if fn is None or (isinstance(fn, float) and pd.isna(fn)):
            continue
        fn = str(fn).strip()
        if not fn:
            continue
        bucket = lookup.setdefault(cid, [])
        if fn not in bucket:
            bucket.append(fn)
    return lookup


def build_sample(
    unified_df: pd.DataFrame,
    xlsx_df: pd.DataFrame,
    pdfs_existing: set,
    seed: int = SEED,
) -> pd.DataFrame:
    """Build the 50-row stratified sample (deterministic given inputs+seed)."""
    pool = build_pool(unified_df)
    xlsx_lookup = _build_xlsx_lookup(xlsx_df)

    sampled_parts: List[pd.DataFrame] = []
    # Use one RNG advanced sequentially across strata for full determinism.
    rng = random.Random(seed)

    for strata_name, n in STRATA_COMPOSITION.items():
        prefer_abstract = strata_name == "include"
        picked = sample_stratum(
            pool=pool,
            strata_name=strata_name,
            n=n,
            rng=rng,
            prefer_abstract=prefer_abstract,
        )
        picked = picked.copy()
        picked["strata"] = strata_name
        sampled_parts.append(picked)

    sampled = pd.concat(sampled_parts, ignore_index=True)

    # Derive output columns.
    sampled["has_abstract"] = sampled["abstract"].apply(_has_non_empty_abstract)
    sampled["has_pdf"] = sampled["covidence_id"].apply(
        lambda cid: compute_has_pdf(cid, xlsx_lookup, pdfs_existing)
    )
    sampled["pdf_filename"] = sampled["covidence_id"].apply(
        lambda cid: pick_pdf_filename(cid, xlsx_lookup, pdfs_existing)
    )
    sampled["purpose"] = sampled["strata"].apply(
        lambda s: PURPOSE_INCLUDE if s == "include" else PURPOSE_EXCLUDE
    )

    # Order columns and rows (strata declaration order + covidence_id ASC).
    strata_order = {name: idx for idx, name in enumerate(STRATA_COMPOSITION)}
    sampled["_strata_order"] = sampled["strata"].map(strata_order)
    sampled = sampled.sort_values(
        by=["_strata_order", "covidence_id"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    sampled = sampled.drop(columns=["_strata_order"])

    return sampled[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
def verify_invariants(sample_df: pd.DataFrame, unified_df: pd.DataFrame) -> None:
    """Raise AssertionError if any invariant is violated."""
    # Shape (50, 10).
    assert sample_df.shape == (50, 10), (
        f"Shape esperada (50, 10), obtenida {sample_df.shape}"
    )

    # Distribution strict per strata.
    counts = sample_df["strata"].value_counts().to_dict()
    for strata_name, n in STRATA_COMPOSITION.items():
        got = counts.get(strata_name, 0)
        assert got == n, (
            f"Estrato '{strata_name}': pedido {n}, obtenido {got}"
        )

    # 0 rows with excluded source_method.
    bad = sample_df[sample_df["source_method"].isin(EXCLUDED_SOURCE_METHODS)]
    assert len(bad) == 0, (
        f"{len(bad)} filas con source_method excluido: "
        f"{bad['source_method'].tolist()}"
    )

    # Includes ⊂ real 66 includes.
    real_includes = set(
        unified_df[unified_df["human_label_ft"] == "include"]["covidence_id"].tolist()
    )
    sample_includes = set(
        sample_df[sample_df["strata"] == "include"]["covidence_id"].tolist()
    )
    extra = sample_includes - real_includes
    assert not extra, f"Includes del sample no son includes reales: {extra}"

    # Includes have has_pdf=True (post D.3.bis recovery).
    includes_df = sample_df[sample_df["strata"] == "include"]
    no_pdf = includes_df[~includes_df["has_pdf"]]
    if len(no_pdf) > 0:
        raise AssertionError(
            "Includes sin PDF (invariante critico violado): "
            + str(no_pdf["covidence_id"].tolist())
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _study_label_from_filename(filename: Optional[str]) -> str:
    """Extract a short label from an xlsx filename if parseable."""
    if not filename:
        return ""
    base = filename.rsplit(".", 1)[0]
    # Heuristic: "Author YYYY - Title..." or "Author - YYYY - Title..."
    parts = base.split(" - ")
    if len(parts) >= 2:
        return parts[0].strip()[:60]
    return base.strip()[:60]


def _load_ref_mapping_optional() -> Dict[str, int]:
    """Return covidence_id -> ref_number map if mapping CSV is present."""
    if not REF_MAPPING_CSV.exists():
        return {}
    df = pd.read_csv(REF_MAPPING_CSV)
    out: Dict[str, int] = {}
    for _, row in df.iterrows():
        cid = str(row["covidence_id"]).strip()
        try:
            ref = int(row["ref_number"])
        except (ValueError, TypeError):
            continue
        # Keep the lowest ref_number per covidence_id (since one cid may have many).
        if cid not in out or ref < out[cid]:
            out[cid] = ref
    return out


def write_report(
    sample_df: pd.DataFrame,
    unified_df: pd.DataFrame,
    pool: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate the Spanish markdown report describing the sample."""
    lines: List[str] = []
    lines.append("# Sample micro estratificado (T16.D.4.B)")
    lines.append("")
    lines.append("Sample reproducible de 50 papers extraido del gold standard de")
    lines.append("deporte para evaluar Heroes 1-4 en T16.D.5.")
    lines.append("")

    # 1. Proposito
    lines.append("## 1. Proposito")
    lines.append("")
    lines.append(
        "Sample para evaluar Heroes 1-4 (T16.D.5). Composicion estratificada:"
    )
    lines.append("")
    lines.append("- 10 includes -> cubren los 4 heroes (hero1, hero2, hero3, hero4).")
    lines.append("- 40 excludes -> cubren Heroes 1-2 (hero1, hero2).")
    lines.append("")
    lines.append(
        "Los 8 estratos de exclusion FT estan distribuidos en proporcion al "
        "soporte estimado de cada categoria en el gold de Covidence."
    )
    lines.append("")

    # 2. Composicion declarada vs observada
    lines.append("## 2. Composicion declarada vs observada")
    lines.append("")
    counts = sample_df["strata"].value_counts().to_dict()
    lines.append("| Estrato | N pedido | N obtenido | OK |")
    lines.append("|---|---|---|---|")
    for strata_name, n in STRATA_COMPOSITION.items():
        got = counts.get(strata_name, 0)
        ok = "Si" if got == n else "NO"
        lines.append(f"| {strata_name} | {n} | {got} | {ok} |")
    total_pedido = sum(STRATA_COMPOSITION.values())
    total_obt = int(sample_df.shape[0])
    ok_total = "Si" if total_obt == total_pedido else "NO"
    lines.append(f"| **Total** | **{total_pedido}** | **{total_obt}** | **{ok_total}** |")
    lines.append("")

    # 3. Cobertura de metadata
    lines.append("## 3. Cobertura de metadata en el sample")
    lines.append("")
    lines.append("### 3.1 has_abstract por estrato")
    lines.append("")
    lines.append("| Estrato | Con abstract | Sin abstract |")
    lines.append("|---|---|---|")
    for strata_name in STRATA_COMPOSITION:
        sub = sample_df[sample_df["strata"] == strata_name]
        con = int(sub["has_abstract"].sum())
        sin = int((~sub["has_abstract"]).sum())
        lines.append(f"| {strata_name} | {con} | {sin} |")
    lines.append("")
    lines.append("### 3.2 has_pdf por estrato")
    lines.append("")
    lines.append("| Estrato | Con PDF | Sin PDF |")
    lines.append("|---|---|---|")
    for strata_name in STRATA_COMPOSITION:
        sub = sample_df[sample_df["strata"] == strata_name]
        con = int(sub["has_pdf"].sum())
        sin = int((~sub["has_pdf"]).sum())
        lines.append(f"| {strata_name} | {con} | {sin} |")
    lines.append("")
    lines.append("### 3.3 Distribucion source_method en el sample")
    lines.append("")
    sm_counts = sample_df["source_method"].fillna("(NaN)").value_counts()
    lines.append("| source_method | N |")
    lines.append("|---|---|")
    for sm, c in sm_counts.items():
        lines.append(f"| {sm} | {int(c)} |")
    lines.append("")

    # 4. Pool de origen
    lines.append("## 4. Pool de origen")
    lines.append("")
    xlsx_total = int((unified_df["source"] == "xlsx").sum())
    csv_total = int((unified_df["source"] == "csv").sum())
    low_conf = int(
        (unified_df["source_method"] == "crossref_by_title_low_confidence").sum()
    )
    failed = int((unified_df["source_method"] == "failed").sum())
    eligible = len(pool)
    lines.append(f"- xlsx-side total: {xlsx_total}")
    lines.append(
        f"- xlsx-side eligible (tras filtrar low_confidence={low_conf}, "
        f"failed={failed}): **{eligible}**"
    )
    lines.append(
        f"- csv-side total: {csv_total} (elegibles para pool conceptualmente, "
        "pero no aportan estratos FT)"
    )
    lines.append("")
    lines.append(
        "**Razonamiento:** las 19198 filas csv-side son los `irrelevant` de "
        "Covidence (descartados en T/A), no tienen `human_label_ft` ni "
        "`exclusion_reason_ft`. Como el sample estratifica por estrato FT, "
        "csv-side aporta 0 filas. Todo el sample sale de xlsx-side."
    )
    lines.append("")

    # 5. Seed y reproducibilidad
    lines.append("## 5. Seed y reproducibilidad")
    lines.append("")
    lines.append(f"- `SEED = {SEED}`")
    lines.append("- RNG: `random.Random(SEED)` (un solo RNG avanzado secuencialmente).")
    lines.append(
        "- Pool ordenado por `covidence_id ASC` (e `has_abstract DESC` para "
        "includes) antes del shuffle. No se usa `pandas.DataFrame.sample` "
        "(estado global inestable)."
    )
    lines.append(
        "- El script verifica reproducibilidad byte-a-byte ejecutando "
        "`build_sample()` dos veces y comparando los DataFrames antes de "
        "escribir el CSV."
    )
    lines.append("")

    # 6. Listado completo
    lines.append("## 6. Listado completo de los 50")
    lines.append("")
    ref_map = _load_ref_mapping_optional()
    has_ref_col = bool(ref_map)
    header = (
        "| covidence_id | strata | study_label | has_abstract | has_pdf | "
        "source_method |"
    )
    sep = "|---|---|---|---|---|---|"
    if has_ref_col:
        header = (
            "| covidence_id | ref_number | strata | study_label | has_abstract |"
            " has_pdf | source_method |"
        )
        sep = "|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    # We need study_label: try ref_mapping, else fallback to authors/title.
    ref_label_map: Dict[str, str] = {}
    if REF_MAPPING_CSV.exists():
        try:
            ref_df = pd.read_csv(REF_MAPPING_CSV)
            for _, row in ref_df.iterrows():
                cid = str(row["covidence_id"]).strip()
                lbl = row.get("study_label", "")
                if cid not in ref_label_map and isinstance(lbl, str) and lbl.strip():
                    ref_label_map[cid] = lbl.strip()
        except Exception:
            ref_label_map = {}

    # Build authors/title fallback from unified_df.
    meta_map: Dict[str, Dict[str, str]] = {}
    for _, row in unified_df.iterrows():
        cid = str(row["covidence_id"]).strip()
        if cid in meta_map:
            continue
        meta_map[cid] = {
            "authors": str(row.get("authors") or "").strip(),
            "title": str(row.get("title") or "").strip(),
        }

    for _, row in sample_df.iterrows():
        cid = str(row["covidence_id"]).strip()
        # Prefer ref_mapping label; then xlsx filename heuristic; then unified meta.
        label = ref_label_map.get(cid, "")
        if not label:
            label = _study_label_from_filename(row.get("pdf_filename"))
        if not label:
            meta = meta_map.get(cid, {})
            author = meta.get("authors", "")
            if author:
                label = author[:60]
            else:
                label = meta.get("title", "")[:60]
        label = label.replace("|", "/")  # avoid breaking the markdown table
        sm = row["source_method"] if pd.notna(row["source_method"]) else "(NaN)"
        if has_ref_col:
            ref = ref_map.get(cid, "")
            lines.append(
                f"| {cid} | {ref} | {row['strata']} | {label} | "
                f"{row['has_abstract']} | {row['has_pdf']} | {sm} |"
            )
        else:
            lines.append(
                f"| {cid} | {row['strata']} | {label} | "
                f"{row['has_abstract']} | {row['has_pdf']} | {sm} |"
            )
    lines.append("")

    # 7. Anomalias
    lines.append("## 7. Anomalias")
    lines.append("")
    anomalies: List[str] = []
    inc = sample_df[sample_df["strata"] == "include"]
    inc_no_abs = inc[~inc["has_abstract"]]
    if len(inc_no_abs) > 0:
        anomalies.append(
            f"- {len(inc_no_abs)} include(s) sin abstract: "
            f"{inc_no_abs['covidence_id'].tolist()}"
        )
    inc_no_pdf = inc[~inc["has_pdf"]]
    if len(inc_no_pdf) > 0:
        anomalies.append(
            f"- {len(inc_no_pdf)} include(s) sin PDF: "
            f"{inc_no_pdf['covidence_id'].tolist()}"
        )
    exc = sample_df[sample_df["strata"] != "include"]
    exc_no_pdf_count = int((~exc["has_pdf"]).sum())
    if exc_no_pdf_count > 0:
        anomalies.append(
            f"- {exc_no_pdf_count} exclude(s) sin PDF (informativo, no critico)."
        )
    exc_no_abs_count = int((~exc["has_abstract"]).sum())
    if exc_no_abs_count > 0:
        anomalies.append(
            f"- {exc_no_abs_count} exclude(s) sin abstract (informativo)."
        )
    if not anomalies:
        anomalies.append("- Sin anomalias detectadas.")
    lines.extend(anomalies)
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _scan_pdfs(pdfs_dir: Path) -> set:
    """Return set of filenames present in pdfs_dir (not full paths)."""
    if not pdfs_dir.exists():
        raise FileNotFoundError(f"Directorio de PDFs no encontrado: {pdfs_dir}")
    return {p.name for p in pdfs_dir.iterdir() if p.is_file()}


def main() -> int:
    """Punto de entrada del script."""
    print("[T16.D.4.B] Construyendo sample micro estratificado de 50 papers...")
    print()

    # 1. Load unified v2.
    if not UNIFIED_CSV.exists():
        print(f"[ERROR] No existe el CSV unificado: {UNIFIED_CSV}", file=sys.stderr)
        return 1
    print(f"  Cargando unified v2: {UNIFIED_CSV}")
    unified_df = pd.read_csv(UNIFIED_CSV)
    print(f"  unified shape: {unified_df.shape}")

    # 2. Load xlsx (for filenames).
    if not XLSX_PATH.exists():
        print(f"[ERROR] No existe el xlsx: {XLSX_PATH}", file=sys.stderr)
        return 1
    print(f"  Cargando xlsx: {XLSX_PATH}")
    xlsx_df = pd.read_excel(XLSX_PATH)
    print(f"  xlsx shape: {xlsx_df.shape}")

    # 3. Scan physical PDFs.
    print(f"  Escaneando PDFs en: {PDFS_DIR}")
    pdfs_existing = _scan_pdfs(PDFS_DIR)
    print(f"  PDFs fisicos encontrados: {len(pdfs_existing)}")

    # 4. Build pool.
    pool = build_pool(unified_df)
    print(f"  Pool xlsx-side eligible: {len(pool)} filas")
    print()

    # 5. Build sample twice for byte-a-byte reproducibility check.
    print("  Construyendo sample (1/2)...")
    sample_a = build_sample(unified_df, xlsx_df, pdfs_existing, seed=SEED)
    print("  Construyendo sample (2/2) para verificar reproducibilidad...")
    sample_b = build_sample(unified_df, xlsx_df, pdfs_existing, seed=SEED)
    csv_a = sample_a.to_csv(index=False)
    csv_b = sample_b.to_csv(index=False)
    if csv_a != csv_b:
        print(
            "[ERROR] Reproducibilidad rota: dos ejecuciones generaron CSVs "
            "distintos. Abortando.",
            file=sys.stderr,
        )
        return 2
    print("  Reproducibilidad byte-a-byte: OK")
    print()

    # 6. Verify invariants.
    print("  Verificando invariantes...")
    try:
        verify_invariants(sample_a, unified_df)
    except AssertionError as exc:
        print(f"[ERROR] Invariante violado: {exc}", file=sys.stderr)
        return 3
    print("  Invariantes: OK")
    print()

    # 7. Write outputs.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_a.to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV escrito: {OUTPUT_CSV}")
    write_report(sample_a, unified_df, pool, OUTPUT_REPORT)
    print(f"  Reporte escrito: {OUTPUT_REPORT}")
    print()

    # 8. Summary.
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Sample shape: {sample_a.shape}")
    print()
    print("Composicion por estrato:")
    counts = sample_a["strata"].value_counts().to_dict()
    for strata_name, n in STRATA_COMPOSITION.items():
        got = counts.get(strata_name, 0)
        flag = "OK" if got == n else "FAIL"
        print(f"  {strata_name:<25} pedido={n:>2}  obtenido={got:>2}  [{flag}]")
    print()
    print(f"has_abstract True: {int(sample_a['has_abstract'].sum())}/50")
    print(f"has_pdf      True: {int(sample_a['has_pdf'].sum())}/50")
    inc = sample_a[sample_a["strata"] == "include"]
    print(
        f"Includes con PDF: {int(inc['has_pdf'].sum())}/{len(inc)} (debe ser 10/10)"
    )
    print()
    print("[OK] T16.D.4.B completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
