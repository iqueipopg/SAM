#!/usr/bin/env python3
"""
build_medio_sample.py — Sample medio para los 4 heroes (T16.E.1).

Construye 4 sub-samples reproducibles con composicion fijada por Ignacio
(2026-05-17). Cada hero recibe un sample independiente:

  - sample_medio_a3.csv  (Hero 1)  : 2000 T/A (1000 relevant + 1000 irrelevant)
  - sample_medio_a4.csv  (Hero 2)  : 566 FT (66 includes + 500 excludes stratified)
  - sample_medio_a5.csv  (Hero 3)  : todos los includes con PDF canonico procesable
  - sample_medio_a6.csv  (Hero 4)  : todos los includes con mapping high a gold_rob

Composicion fijada (no improvisar):

  Hero 1 (2000):
    - 1000 relevant (xlsx-side, eligible): proporcional include/exclude FT
    - 1000 irrelevant (csv-side): muestra aleatoria simple

  Hero 2 (566):
    - 66 includes (todos los disponibles)
    - 500 excludes estratificados (proporcional por las 7 razones de Paula,
      minimo 10 por razon; si una razon tiene <10 estudios totales, tomar
      todos los disponibles)

  Hero 3 (~62-64):
    - Todos los includes con `extracted_metadata.csv.pdf_used` != null
    - Esperado N=64 antes de filtrar escaneados/marginales (#3490, #2098 null)

  Hero 4 (62):
    - Todos los includes con mapping high-confidence en
      `ref_to_covidence_mapping.csv` (62 segun T16.D.4)

Reproducibilidad: SEED=20260517. Script ejecuta build dos veces y compara
CSV byte-a-byte; aborta si difieren.

Identificadores y comentarios en ingles. Prints y docstrings en espanol.

Uso:
    ~/meta-analisis-automatizado/venv/bin/python \\
      scripts/build_medio_sample.py
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
EXTRACTED_META_CSV = GOLD_DIR / "unified" / "extracted_metadata.csv"
OUTPUT_DIR = GOLD_DIR / "samples" / "medio"
OUTPUT_A3 = OUTPUT_DIR / "sample_medio_a3.csv"
OUTPUT_A4 = OUTPUT_DIR / "sample_medio_a4.csv"
OUTPUT_A5 = OUTPUT_DIR / "sample_medio_a5.csv"
OUTPUT_A6 = OUTPUT_DIR / "sample_medio_a6.csv"
OUTPUT_REPORT = OUTPUT_DIR / "composition_report.md"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 20260517

EXCLUDED_SOURCE_METHODS = {"crossref_by_title_low_confidence", "failed"}

# Per Ignacio (2026-05-17), Hero 1 totals.
HERO1_RELEVANT_TARGET = 1000
HERO1_IRRELEVANT_TARGET = 1000

# Per Ignacio, Hero 2 totals: 66 includes + 500 excludes.
HERO2_EXCLUDE_TARGET = 500
HERO2_MIN_PER_REASON = 10

# Hero 2 reasons in distribution order (used by composition report).
HERO2_REASONS_ORDERED = [
    "Wrong age",
    "Wrong outcome",
    "Wrong study design",
    "Wrong intervention",
    "Wrong publication type",
    "Wrong population",
    "Sin motivo registrado",
    "Language",
]

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

PURPOSE_HERO1_ONLY = "hero1"
PURPOSE_HERO2_ONLY = "hero2"
PURPOSE_HERO1_PLUS = "hero1,hero2"
PURPOSE_INCLUDE_ALL = "hero1,hero2,hero3,hero4"
PURPOSE_HERO3_ONLY = "hero3"
PURPOSE_HERO4_ONLY = "hero4"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_non_empty_abstract(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return len(str(value).strip()) > 0


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


def _normalize_cid(value: object) -> str:
    return str(value).strip()


def _pick_pdf_filename(
    cid: str,
    xlsx_lookup: Dict[str, List[str]],
    pdfs_existing: set,
) -> str | None:
    filenames = sorted(xlsx_lookup.get(_normalize_cid(cid), []))
    for fn in filenames:
        if fn in pdfs_existing:
            return fn
    return None


def _scan_pdfs(pdfs_dir: Path) -> set:
    if not pdfs_dir.exists():
        raise FileNotFoundError(f"PDFs dir not found: {pdfs_dir}")
    return {p.name for p in pdfs_dir.iterdir() if p.is_file()}


def _shuffled_indices(n: int, rng: random.Random) -> List[int]:
    indices = list(range(n))
    rng.shuffle(indices)
    return indices


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Build pool
# ---------------------------------------------------------------------------
def build_xlsx_pool(unified_df: pd.DataFrame) -> pd.DataFrame:
    """Eligible xlsx-side pool: drop low_confidence and failed."""
    xlsx_side = unified_df[unified_df["source"] == "xlsx"].copy()
    mask = ~xlsx_side["source_method"].isin(EXCLUDED_SOURCE_METHODS)
    return xlsx_side[mask].copy()


def build_csv_pool(unified_df: pd.DataFrame) -> pd.DataFrame:
    """csv-side pool (irrelevant T/A; no FT decision)."""
    return unified_df[unified_df["source"] == "csv"].copy()


# ---------------------------------------------------------------------------
# Hero 2 quotas
# ---------------------------------------------------------------------------
def compute_hero2_exclude_quotas(
    xlsx_pool: pd.DataFrame,
    target: int = HERO2_EXCLUDE_TARGET,
    min_per_reason: int = HERO2_MIN_PER_REASON,
) -> Tuple[Dict[str, int], List[str]]:
    """Return ``(quotas, adjustment_notes)``.

    ``quotas`` maps reason -> N to sample. ``adjustment_notes`` records when
    quotas were forced up (min_per_reason) or down (small categories where
    universe < min_per_reason).
    """
    excludes = xlsx_pool[xlsx_pool["human_label_ft"] == "exclude"]
    counts = (
        excludes["exclusion_reason_ft"].value_counts(dropna=False).to_dict()
    )
    total_excludes = sum(int(v) for v in counts.values())
    notes: List[str] = []
    quotas: Dict[str, int] = {}

    # First pass: proportional.
    raw: Dict[str, float] = {
        reason: (counts.get(reason, 0) / total_excludes) * target
        for reason in HERO2_REASONS_ORDERED
    }
    proportional = {reason: int(round(raw[reason])) for reason in HERO2_REASONS_ORDERED}

    # Second pass: apply minimum and capacity constraints.
    for reason in HERO2_REASONS_ORDERED:
        universe = int(counts.get(reason, 0))
        prop_n = proportional[reason]
        if universe < min_per_reason:
            # Take all available.
            quotas[reason] = universe
            if universe < prop_n:
                notes.append(
                    f"`{reason}`: universo={universe} < min={min_per_reason} → "
                    f"se toman todos los {universe} disponibles."
                )
        else:
            n_capped = max(prop_n, min_per_reason)
            if n_capped > universe:
                n_capped = universe
                notes.append(
                    f"`{reason}`: requerido={prop_n}, universo={universe} → "
                    f"capado a {universe}."
                )
            if n_capped > prop_n:
                notes.append(
                    f"`{reason}`: proporcional={prop_n} < min={min_per_reason} → "
                    f"elevado a {n_capped}."
                )
            quotas[reason] = n_capped

    # Third pass: rebalance to hit exact target.
    total = sum(quotas.values())
    delta = total - target
    if delta != 0:
        # Adjust the largest mutable category (Wrong age usually).
        # A "mutable" reason is one whose quota can move without violating
        # universe or min constraints.
        sorted_reasons = sorted(
            HERO2_REASONS_ORDERED,
            key=lambda r: counts.get(r, 0),
            reverse=True,
        )
        for reason in sorted_reasons:
            if delta == 0:
                break
            universe = int(counts.get(reason, 0))
            curr = quotas[reason]
            if delta > 0:
                # Need to reduce.
                room = curr - min_per_reason
                if room > 0:
                    cut = min(delta, room)
                    quotas[reason] = curr - cut
                    delta -= cut
                    notes.append(
                        f"`{reason}`: rebalance -{cut} para alcanzar total {target}."
                    )
            else:
                # Need to increase.
                room = universe - curr
                if room > 0:
                    add = min(-delta, room)
                    quotas[reason] = curr + add
                    delta += add
                    notes.append(
                        f"`{reason}`: rebalance +{add} para alcanzar total {target}."
                    )
        if delta != 0:
            notes.append(
                f"AVISO: rebalance imposible — total final = {sum(quotas.values())}, "
                f"target = {target}, delta = {delta}."
            )

    return quotas, notes


# ---------------------------------------------------------------------------
# Hero builders
# ---------------------------------------------------------------------------
def build_hero1_sample(
    unified_df: pd.DataFrame,
    xlsx_lookup: Dict[str, List[str]],
    pdfs_existing: set,
    rng: random.Random,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build sample_medio_a3.csv. Returns (df, meta)."""
    xlsx_pool = build_xlsx_pool(unified_df)
    csv_pool = build_csv_pool(unified_df)

    # ── 1000 relevant: proportional include/exclude FT ────────────────────
    rel_includes = xlsx_pool[xlsx_pool["human_label_ft"] == "include"].copy()
    rel_excludes = xlsx_pool[xlsx_pool["human_label_ft"] == "exclude"].copy()
    rel_pool_size = len(rel_includes) + len(rel_excludes)

    if rel_pool_size < HERO1_RELEVANT_TARGET:
        # Take all.
        n_includes = len(rel_includes)
        n_excludes = len(rel_excludes)
    else:
        n_includes = int(round(len(rel_includes) / rel_pool_size * HERO1_RELEVANT_TARGET))
        n_excludes = HERO1_RELEVANT_TARGET - n_includes

    # Cap to availability.
    n_includes = min(n_includes, len(rel_includes))
    n_excludes = min(n_excludes, len(rel_excludes))

    inc_sorted = rel_includes.sort_values(by="covidence_id", kind="mergesort").reset_index(drop=True)
    inc_idx = _shuffled_indices(len(inc_sorted), rng)
    inc_picked = inc_sorted.iloc[inc_idx[:n_includes]].reset_index(drop=True)

    exc_sorted = rel_excludes.sort_values(by="covidence_id", kind="mergesort").reset_index(drop=True)
    exc_idx = _shuffled_indices(len(exc_sorted), rng)
    exc_picked = exc_sorted.iloc[exc_idx[:n_excludes]].reset_index(drop=True)

    # ── 1000 irrelevant: random simple from csv pool ──────────────────────
    irr_target = HERO1_IRRELEVANT_TARGET
    if len(csv_pool) < irr_target:
        irr_target = len(csv_pool)
    irr_sorted = csv_pool.sort_values(by="covidence_id", kind="mergesort").reset_index(drop=True)
    irr_idx = _shuffled_indices(len(irr_sorted), rng)
    irr_picked = irr_sorted.iloc[irr_idx[:irr_target]].reset_index(drop=True)

    # ── Annotate and assemble ─────────────────────────────────────────────
    inc_picked = inc_picked.copy()
    inc_picked["strata"] = "relevant_include_ft"
    inc_picked["purpose"] = PURPOSE_HERO1_ONLY

    exc_picked = exc_picked.copy()
    exc_picked["strata"] = "relevant_exclude_ft"
    exc_picked["purpose"] = PURPOSE_HERO1_ONLY

    irr_picked = irr_picked.copy()
    irr_picked["strata"] = "irrelevant_ta"
    irr_picked["purpose"] = PURPOSE_HERO1_ONLY

    parts = [inc_picked, exc_picked, irr_picked]
    sampled = pd.concat(parts, ignore_index=True)
    sampled = _annotate_pdf_fields(sampled, xlsx_lookup, pdfs_existing)

    # Sort: strata declaration order, then covidence_id ASC.
    strata_order = {
        "relevant_include_ft": 0,
        "relevant_exclude_ft": 1,
        "irrelevant_ta": 2,
    }
    sampled["_strata_order"] = sampled["strata"].map(strata_order)
    sampled = sampled.sort_values(
        by=["_strata_order", "covidence_id"], ascending=[True, True], kind="mergesort"
    ).reset_index(drop=True)
    sampled = sampled.drop(columns=["_strata_order"])

    out = sampled[OUTPUT_COLUMNS]
    meta = {
        "n_total": len(out),
        "n_relevant_include_ft": int(n_includes),
        "n_relevant_exclude_ft": int(n_excludes),
        "n_irrelevant": int(irr_target),
        "rel_pool_size": int(rel_pool_size),
        "csv_pool_size": int(len(csv_pool)),
    }
    return out, meta


def build_hero2_sample(
    unified_df: pd.DataFrame,
    xlsx_lookup: Dict[str, List[str]],
    pdfs_existing: set,
    rng: random.Random,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build sample_medio_a4.csv. Returns (df, meta)."""
    xlsx_pool = build_xlsx_pool(unified_df)

    # All 66 includes (xlsx).
    includes_all = unified_df[
        (unified_df["source"] == "xlsx")
        & (unified_df["human_label_ft"] == "include")
    ].copy()
    # Note: we deliberately use unfiltered includes (low_confidence rows
    # included if present) because we want all 66 includes per spec.
    n_inc_available = len(includes_all)

    # Excludes stratified.
    quotas, adjustment_notes = compute_hero2_exclude_quotas(xlsx_pool)

    exclude_parts: List[pd.DataFrame] = []
    quotas_actual: Dict[str, int] = {}
    excludes_pool = xlsx_pool[xlsx_pool["human_label_ft"] == "exclude"]
    for reason in HERO2_REASONS_ORDERED:
        n_target = quotas.get(reason, 0)
        subset = excludes_pool[
            excludes_pool["exclusion_reason_ft"] == reason
        ].copy()
        subset = subset.sort_values(by="covidence_id", kind="mergesort").reset_index(drop=True)
        idx = _shuffled_indices(len(subset), rng)
        picked = subset.iloc[idx[:n_target]].reset_index(drop=True)
        picked = picked.copy()
        picked["strata"] = reason
        picked["purpose"] = PURPOSE_HERO2_ONLY
        exclude_parts.append(picked)
        quotas_actual[reason] = len(picked)

    # Includes annotated.
    includes_all = includes_all.copy()
    includes_all["strata"] = "include"
    includes_all["purpose"] = PURPOSE_INCLUDE_ALL

    parts = [includes_all] + exclude_parts
    sampled = pd.concat(parts, ignore_index=True)
    sampled = _annotate_pdf_fields(sampled, xlsx_lookup, pdfs_existing)

    # Sort: include first, then reasons in declaration order, then covidence ASC.
    strata_order = {"include": 0}
    for i, reason in enumerate(HERO2_REASONS_ORDERED, start=1):
        strata_order[reason] = i
    sampled["_strata_order"] = sampled["strata"].map(strata_order)
    sampled = sampled.sort_values(
        by=["_strata_order", "covidence_id"], ascending=[True, True], kind="mergesort"
    ).reset_index(drop=True)
    sampled = sampled.drop(columns=["_strata_order"])

    out = sampled[OUTPUT_COLUMNS]
    meta = {
        "n_total": len(out),
        "n_includes": int(n_inc_available),
        "n_excludes_per_reason": quotas_actual,
        "adjustment_notes": adjustment_notes,
    }
    return out, meta


def build_hero3_sample(
    unified_df: pd.DataFrame,
    extracted_meta_df: pd.DataFrame,
    xlsx_lookup: Dict[str, List[str]],
    pdfs_existing: set,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build sample_medio_a5.csv: includes with canonical pdf_used != null.

    Marginal/scanned filtering is deferred to Phase 2 (extractability check);
    here we include every include with a canonical PDF reference.
    """
    inc_ids = set(
        unified_df[unified_df["human_label_ft"] == "include"]["covidence_id"]
        .astype(str)
        .tolist()
    )
    meta_inc = extracted_meta_df[
        extracted_meta_df["covidence_id"].astype(str).isin(inc_ids)
    ].copy()
    eligible = meta_inc[meta_inc["pdf_used"].notna()].copy()
    null_pdf_ids = sorted(
        meta_inc[meta_inc["pdf_used"].isna()]["covidence_id"].astype(str).tolist()
    )

    # Build sample rows from unified (to keep all gold fields available).
    unified_idx = unified_df.set_index("covidence_id")
    rows: List[Dict[str, object]] = []
    canonical_used: Dict[str, str] = {}
    for _, row in eligible.iterrows():
        cid = str(row["covidence_id"])
        canonical = str(row["pdf_used"]).strip()
        gold_row = unified_idx.loc[cid]
        if isinstance(gold_row, pd.DataFrame):
            gold_row = gold_row.iloc[0]
        rows.append({
            "covidence_id": cid,
            "strata": "include",
            "human_label_ta": gold_row.get("human_label_ta"),
            "human_label_ft": gold_row.get("human_label_ft"),
            "exclusion_reason_ft": gold_row.get("exclusion_reason_ft"),
            "has_abstract": _has_non_empty_abstract(gold_row.get("abstract")),
            "has_pdf": canonical in pdfs_existing,
            "source_method": gold_row.get("source_method"),
            "purpose": PURPOSE_INCLUDE_ALL,
            "pdf_filename": canonical,
        })
        canonical_used[cid] = canonical

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out = out.sort_values(by="covidence_id", kind="mergesort").reset_index(drop=True)
    meta = {
        "n_total": len(out),
        "n_includes_with_canonical_pdf": int(len(eligible)),
        "n_includes_no_canonical_pdf": int(len(null_pdf_ids)),
        "covidence_ids_no_canonical_pdf": null_pdf_ids,
        "n_pdf_present_on_disk": int(out["has_pdf"].sum()),
        "n_pdf_missing_on_disk": int((~out["has_pdf"]).sum()),
    }
    return out, meta


def build_hero4_sample(
    unified_df: pd.DataFrame,
    ref_mapping_df: pd.DataFrame,
    extracted_meta_df: pd.DataFrame,
    xlsx_lookup: Dict[str, List[str]],
    pdfs_existing: set,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build sample_medio_a6.csv: includes with high-confidence ref mapping."""
    inc_ids = set(
        unified_df[unified_df["human_label_ft"] == "include"]["covidence_id"]
        .astype(str)
        .tolist()
    )
    high = ref_mapping_df[ref_mapping_df["confidence"] == "high"].copy()
    high_cids = set(high["covidence_id"].dropna().astype(str).tolist())
    eligible_cids = sorted(inc_ids & high_cids)

    # Use canonical pdf_used if available (Hero 4 also extracts text).
    meta_idx = extracted_meta_df.set_index("covidence_id")
    unified_idx = unified_df.set_index("covidence_id")

    rows: List[Dict[str, object]] = []
    missing_canonical: List[str] = []
    for cid in eligible_cids:
        gold_row = unified_idx.loc[cid]
        if isinstance(gold_row, pd.DataFrame):
            gold_row = gold_row.iloc[0]
        canonical = None
        if cid in meta_idx.index:
            meta_row = meta_idx.loc[cid]
            if isinstance(meta_row, pd.DataFrame):
                meta_row = meta_row.iloc[0]
            cand = meta_row.get("pdf_used")
            if isinstance(cand, str) and cand.strip():
                canonical = cand.strip()
        if canonical is None:
            # Fall back to xlsx lookup.
            canonical = _pick_pdf_filename(cid, xlsx_lookup, pdfs_existing)
        if canonical is None:
            missing_canonical.append(cid)
            canonical = ""
        rows.append({
            "covidence_id": cid,
            "strata": "include",
            "human_label_ta": gold_row.get("human_label_ta"),
            "human_label_ft": gold_row.get("human_label_ft"),
            "exclusion_reason_ft": gold_row.get("exclusion_reason_ft"),
            "has_abstract": _has_non_empty_abstract(gold_row.get("abstract")),
            "has_pdf": bool(canonical) and canonical in pdfs_existing,
            "source_method": gold_row.get("source_method"),
            "purpose": PURPOSE_INCLUDE_ALL,
            "pdf_filename": canonical,
        })

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out = out.sort_values(by="covidence_id", kind="mergesort").reset_index(drop=True)
    meta = {
        "n_total": len(out),
        "n_includes_universe": int(len(inc_ids)),
        "n_ref_high_confidence_universe": int(len(high_cids)),
        "n_eligible_intersection": int(len(eligible_cids)),
        "covidence_ids_missing_canonical_pdf": missing_canonical,
        "n_pdf_present_on_disk": int(out["has_pdf"].sum()),
        "n_pdf_missing_on_disk": int((~out["has_pdf"]).sum()),
    }
    return out, meta


# ---------------------------------------------------------------------------
# Shared annotation
# ---------------------------------------------------------------------------
def _annotate_pdf_fields(
    df: pd.DataFrame,
    xlsx_lookup: Dict[str, List[str]],
    pdfs_existing: set,
) -> pd.DataFrame:
    df = df.copy()
    df["has_abstract"] = df["abstract"].apply(_has_non_empty_abstract)
    df["pdf_filename"] = df["covidence_id"].apply(
        lambda cid: _pick_pdf_filename(cid, xlsx_lookup, pdfs_existing) or ""
    )
    df["has_pdf"] = df["pdf_filename"].apply(lambda fn: bool(fn) and fn in pdfs_existing)
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_all(
    unified_df: pd.DataFrame,
    extracted_meta_df: pd.DataFrame,
    ref_mapping_df: pd.DataFrame,
    xlsx_lookup: Dict[str, List[str]],
    pdfs_existing: set,
    seed: int = SEED,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, object]]]:
    """Run all four builders with a single RNG for full determinism."""
    rng = random.Random(seed)
    samples: Dict[str, pd.DataFrame] = {}
    metas: Dict[str, Dict[str, object]] = {}

    # Order matters for determinism (single RNG advanced sequentially).
    h1_df, h1_meta = build_hero1_sample(unified_df, xlsx_lookup, pdfs_existing, rng)
    samples["a3"] = h1_df
    metas["a3"] = h1_meta

    h2_df, h2_meta = build_hero2_sample(unified_df, xlsx_lookup, pdfs_existing, rng)
    samples["a4"] = h2_df
    metas["a4"] = h2_meta

    h3_df, h3_meta = build_hero3_sample(
        unified_df, extracted_meta_df, xlsx_lookup, pdfs_existing
    )
    samples["a5"] = h3_df
    metas["a5"] = h3_meta

    h4_df, h4_meta = build_hero4_sample(
        unified_df, ref_mapping_df, extracted_meta_df, xlsx_lookup, pdfs_existing
    )
    samples["a6"] = h4_df
    metas["a6"] = h4_meta

    return samples, metas


# ---------------------------------------------------------------------------
# Composition report
# ---------------------------------------------------------------------------
def write_composition_report(
    metas: Dict[str, Dict[str, object]],
    output_paths: Dict[str, Path],
    csv_hashes: Dict[str, str],
    seed: int,
) -> None:
    lines: List[str] = []
    lines.append("# Sample medio — composicion (T16.E.1)")
    lines.append("")
    lines.append("Fecha: 2026-05-17. Composicion fijada por Ignacio (sin improvisacion).")
    lines.append(f"Seed: `{seed}`.")
    lines.append("")
    lines.append("## Resumen general")
    lines.append("")
    lines.append("| Hero | Sample | Filas | Hash SHA-256 |")
    lines.append("|---|---|---:|---|")
    for hero_id in ("a3", "a4", "a5", "a6"):
        n = metas[hero_id]["n_total"]
        path = output_paths[hero_id]
        h = csv_hashes[hero_id][:16]
        lines.append(
            f"| Hero {{'a3':1,'a4':2,'a5':3,'a6':4}}[hero_id] | "
            f"`{path.name}` | {n} | `{h}…` |"
        )
    # Replace placeholder mapping inline (Markdown does not support expr).
    lines = [
        ln.replace(
            "{'a3':1,'a4':2,'a5':3,'a6':4}[hero_id]",
            {"a3": "1", "a4": "2", "a5": "3", "a6": "4"}.get(
                ln.split("`")[1].split("_")[2].split(".")[0]
                if "sample_medio_a" in ln
                else "?", "?"
            ),
        )
        for ln in lines
    ]
    # Simpler: rebuild that block cleanly.
    lines = lines[:lines.index("| Hero | Sample | Filas | Hash SHA-256 |") + 2]
    label_map = {"a3": "Hero 1 (A3)", "a4": "Hero 2 (A4)", "a5": "Hero 3 (A5)", "a6": "Hero 4 (A6)"}
    for hero_id in ("a3", "a4", "a5", "a6"):
        n = metas[hero_id]["n_total"]
        path = output_paths[hero_id]
        h = csv_hashes[hero_id][:16]
        lines.append(f"| {label_map[hero_id]} | `{path.name}` | {n} | `{h}…` |")
    lines.append("")

    # Hero 1
    lines.append("## Hero 1 (sample_medio_a3.csv)")
    lines.append("")
    m = metas["a3"]
    lines.append(f"- Total filas: **{m['n_total']}**")
    lines.append(f"- Relevant total: **{int(m['n_relevant_include_ft']) + int(m['n_relevant_exclude_ft'])}**")
    lines.append(f"  - relevant_include_ft: {m['n_relevant_include_ft']}")
    lines.append(f"  - relevant_exclude_ft: {m['n_relevant_exclude_ft']}")
    lines.append(f"- Irrelevant: **{m['n_irrelevant']}** (csv-side aleatorio simple)")
    lines.append("")
    lines.append(
        f"Pool xlsx eligible (post low_confidence/failed filter): "
        f"{m['rel_pool_size']}. Pool csv-side: {m['csv_pool_size']}."
    )
    lines.append("")
    lines.append(
        "**AVISO BLOQUEANTE:** `experiments/eval_hero1_screening.py` linea 216 "
        "hardcodea `gold_label = \"relevant\"` (sample invariante 100% positivo). "
        "El sample medio Hero 1 contiene 1000 relevant + 1000 irrelevant, "
        "incompatible con el evaluador actual sin modificarlo. Ver "
        "`AVISO BLOQUEANTE` al final del informe."
    )
    lines.append("")

    # Hero 2
    lines.append("## Hero 2 (sample_medio_a4.csv)")
    lines.append("")
    m = metas["a4"]
    lines.append(f"- Total filas: **{m['n_total']}** (66 includes + 500 excludes objetivo)")
    lines.append(f"- Includes: {m['n_includes']}")
    lines.append("")
    lines.append("### Distribucion de excludes por razon")
    lines.append("")
    lines.append("| Razon | N en muestra |")
    lines.append("|---|---:|")
    for reason in HERO2_REASONS_ORDERED:
        n = m["n_excludes_per_reason"].get(reason, 0)
        lines.append(f"| {reason} | {n} |")
    lines.append(f"| **Total excludes** | **{sum(m['n_excludes_per_reason'].values())}** |")
    lines.append("")
    if m["adjustment_notes"]:
        lines.append("### Ajustes aplicados")
        lines.append("")
        for note in m["adjustment_notes"]:
            lines.append(f"- {note}")
        lines.append("")

    # Hero 3
    lines.append("## Hero 3 (sample_medio_a5.csv)")
    lines.append("")
    m = metas["a5"]
    lines.append(f"- Total filas: **{m['n_total']}**")
    lines.append(
        f"- Includes con `extracted_metadata.pdf_used` no nulo: "
        f"{m['n_includes_with_canonical_pdf']}"
    )
    lines.append(
        f"- Includes sin PDF canonico: "
        f"{m['n_includes_no_canonical_pdf']} "
        f"(`covidence_ids` = {m['covidence_ids_no_canonical_pdf']})"
    )
    lines.append(
        f"- PDFs presentes en disco: {m['n_pdf_present_on_disk']} / {m['n_total']}"
    )
    if m["n_pdf_missing_on_disk"]:
        lines.append(
            f"- AVISO: {m['n_pdf_missing_on_disk']} PDFs canonicos no estan en "
            f"`pdfs_todos/`. Revisar."
        )
    lines.append("")
    lines.append(
        "Filtrado de escaneados/marginales (#3490, #7759, etc.) se aplica en "
        "Fase 2 (PDF extractability). Aqui se incluyen TODOS los includes con "
        "`pdf_used` no nulo."
    )
    lines.append("")

    # Hero 4
    lines.append("## Hero 4 (sample_medio_a6.csv)")
    lines.append("")
    m = metas["a6"]
    lines.append(f"- Total filas: **{m['n_total']}**")
    lines.append(f"- Includes universo: {m['n_includes_universe']}")
    lines.append(f"- Mapping high-confidence universo: {m['n_ref_high_confidence_universe']}")
    lines.append(f"- Interseccion (incluido AND high mapping): {m['n_eligible_intersection']}")
    lines.append(
        f"- PDFs presentes en disco: {m['n_pdf_present_on_disk']} / {m['n_total']}"
    )
    if m["covidence_ids_missing_canonical_pdf"]:
        lines.append(
            f"- AVISO: {len(m['covidence_ids_missing_canonical_pdf'])} cid(s) "
            f"sin canonical pdf_used: {m['covidence_ids_missing_canonical_pdf']}"
        )
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibilidad")
    lines.append("")
    lines.append(f"- Seed: `{seed}`")
    lines.append("- RNG: `random.Random(seed)` avanzado secuencialmente entre heroes.")
    lines.append("- Build run dos veces; CSV comparados byte-a-byte antes de escribir.")
    lines.append("- Hashes SHA-256 de los 4 CSV en la tabla superior.")
    lines.append("")

    # Hero 1 BLOCKER
    lines.append("---")
    lines.append("")
    lines.append("## AVISO BLOQUEANTE — Hero 1 evaluator incompatible")
    lines.append("")
    lines.append(
        "El evaluador `experiments/eval_hero1_screening.py` esta disenado bajo "
        "el supuesto de que el sample es 100% gold-positive "
        "(`human_label_ta == 'relevant'`). En la linea 216 hardcodea "
        "`gold_label = \"relevant\"  # sample invariant`. Las metricas "
        "calculadas (TP, FN, sensitivity) asumen este invariante."
    )
    lines.append("")
    lines.append(
        "El sample medio Hero 1 contiene 50% relevant + 50% irrelevant per "
        "composicion fijada. Correr el evaluador actual sobre este sample "
        "produciria:"
    )
    lines.append("")
    lines.append(
        "- Matriz de confusion incorrecta: los 1000 irrelevant tendrian "
        "`gold_label='relevant'` (hardcoded), por lo que predicciones correctas "
        "`EXCLUDE` se contarian como FN."
    )
    lines.append(
        "- `sensitivity_overall` calculada sobre 2000 con TP solo de los "
        "verdaderos positivos: deflactada artificialmente."
    )
    lines.append(
        "- `specificity` no se calcula (el evaluador no la implementa)."
    )
    lines.append("")
    lines.append(
        "Resolver requiere: (a) autorizacion del usuario para modificar el "
        "evaluador (~20 LOC para leer `human_label_ta` de la fila en vez de "
        "hardcoding; anadir TN/FP/specificity), o (b) reducir Hero 1 medio a "
        "includes-only (~1500 papers, midiendo solo sensitivity como en micro), "
        "o (c) saltar Hero 1 del run medio."
    )
    lines.append("")
    lines.append(
        "El run overnight Fase 4 ha sido ABORTADO antes de empezar para evitar "
        "consumir ~9.7h de compute con metricas Hero 1 invalidas. Phases 1-2 "
        "completadas; samples medios y check de PDFs listos para retomar el "
        "run en cuanto se decida la via."
    )
    lines.append("")

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("[T16.E.1] Construyendo sample medio (4 heroes)...")
    print()

    if not UNIFIED_CSV.exists():
        print(f"[ERROR] missing: {UNIFIED_CSV}", file=sys.stderr)
        return 1
    if not XLSX_PATH.exists():
        print(f"[ERROR] missing: {XLSX_PATH}", file=sys.stderr)
        return 1
    if not EXTRACTED_META_CSV.exists():
        print(f"[ERROR] missing: {EXTRACTED_META_CSV}", file=sys.stderr)
        return 1
    if not REF_MAPPING_CSV.exists():
        print(f"[ERROR] missing: {REF_MAPPING_CSV}", file=sys.stderr)
        return 1

    print(f"  Cargando unified: {UNIFIED_CSV}")
    unified_df = pd.read_csv(UNIFIED_CSV, dtype={"covidence_id": str})
    print(f"  unified shape: {unified_df.shape}")

    print(f"  Cargando xlsx: {XLSX_PATH}")
    xlsx_df = pd.read_excel(XLSX_PATH)
    print(f"  xlsx shape: {xlsx_df.shape}")

    print(f"  Cargando extracted_metadata: {EXTRACTED_META_CSV}")
    extracted_meta_df = pd.read_csv(EXTRACTED_META_CSV, dtype={"covidence_id": str})

    print(f"  Cargando ref mapping: {REF_MAPPING_CSV}")
    ref_mapping_df = pd.read_csv(REF_MAPPING_CSV, dtype={"covidence_id": str})

    print(f"  Escaneando PDFs en: {PDFS_DIR}")
    pdfs_existing = _scan_pdfs(PDFS_DIR)
    print(f"  PDFs fisicos encontrados: {len(pdfs_existing)}")
    print()

    xlsx_lookup = _build_xlsx_lookup(xlsx_df)

    # Build twice for reproducibility check.
    print("  Construyendo samples (1/2)...")
    samples_a, metas_a = build_all(
        unified_df, extracted_meta_df, ref_mapping_df, xlsx_lookup, pdfs_existing
    )
    print("  Construyendo samples (2/2) para verificar reproducibilidad...")
    samples_b, _ = build_all(
        unified_df, extracted_meta_df, ref_mapping_df, xlsx_lookup, pdfs_existing
    )

    for hero_id in ("a3", "a4", "a5", "a6"):
        csv_a = samples_a[hero_id].to_csv(index=False)
        csv_b = samples_b[hero_id].to_csv(index=False)
        if csv_a != csv_b:
            print(
                f"[ERROR] Reproducibilidad rota para hero {hero_id}: dos builds "
                "produjeron CSVs distintos. Abortando.",
                file=sys.stderr,
            )
            return 2
    print("  Reproducibilidad byte-a-byte: OK")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "a3": OUTPUT_A3,
        "a4": OUTPUT_A4,
        "a5": OUTPUT_A5,
        "a6": OUTPUT_A6,
    }
    csv_hashes: Dict[str, str] = {}
    for hero_id, path in output_paths.items():
        samples_a[hero_id].to_csv(path, index=False)
        csv_hashes[hero_id] = _file_sha256(path)
        print(f"  Escrito: {path}  ({len(samples_a[hero_id])} filas)  sha256={csv_hashes[hero_id][:16]}…")
    print()

    write_composition_report(metas_a, output_paths, csv_hashes, SEED)
    print(f"  Composition report: {OUTPUT_REPORT}")
    print()

    # Final summary stdout.
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for hero_id in ("a3", "a4", "a5", "a6"):
        m = metas_a[hero_id]
        print(f"  sample_medio_{hero_id}.csv: {m['n_total']} filas")
    print()
    print("[OK] Phase 1 (build samples) completada.")
    print(
        "[AVISO] Hero 1 evaluator incompatible con sample mixto — ver "
        "composition_report.md antes de Fase 4."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
