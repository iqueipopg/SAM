# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Cross-reference validation: gold standard sport roster (CSV) vs PDFs on disk.

Reads ``data/gold_standard/deporte/registro_maestro.csv`` and verifies, for each
Covidence ID, which of the listed PDF filenames are physically present under
``data/gold_standard/deporte/`` (recursive search across all subdirectories).
Also identifies orphan PDFs on disk that are not referenced by any roster row.

Output:
    data/gold_standard/deporte/_validation_report.json  (JSON report)
    stdout summary (in Spanish)

The script is idempotent and read-only with respect to the gold standard
directory, except for writing the report file (prefix "_" indicates it is
generated, not source data).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


# --- Constants ---------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = REPO_ROOT / "data" / "gold_standard" / "deporte"
CSV_PATH = GOLD_DIR / "registro_maestro.csv"
REPORT_PATH = GOLD_DIR / "_validation_report.json"

CSV_SEPARATOR = ";"
PDF_LIST_SEPARATOR = " | "
COL_COVIDENCE = "Covidence ID"
COL_CITA = "Cita (Autores)"
COL_PDFS = "Archivos Físicos Extrapolados"
COL_AVISO = "Aviso"


# --- IO helpers --------------------------------------------------------------

def load_roster(csv_path: Path) -> pd.DataFrame:
    """Load the roster CSV with UTF-8, falling back to latin-1 if needed."""
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(csv_path, sep=CSV_SEPARATOR, encoding=encoding, dtype=str)
            return df.fillna("")
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"No se pudo leer {csv_path} con utf-8 ni latin-1")


def scan_pdfs_on_disk(root: Path) -> Dict[str, List[Path]]:
    """Return mapping basename -> list of absolute paths under ``root``.

    Using a list (not a single path) handles the edge case where two
    subdirectories contain a file with the same basename.
    """
    index: Dict[str, List[Path]] = defaultdict(list)
    for pdf in root.rglob("*.pdf"):
        if pdf.is_file():
            index[pdf.name].append(pdf)
    return index


# --- Core validation ---------------------------------------------------------

def parse_expected_pdfs(cell: str) -> List[str]:
    """Split the 'Archivos Físicos Extrapolados' cell into PDF filenames."""
    if not cell or not cell.strip():
        return []
    parts = [p.strip() for p in cell.split(PDF_LIST_SEPARATOR)]
    return [p for p in parts if p]


def classify_studies(
    df: pd.DataFrame,
    disk_index: Dict[str, List[Path]],
) -> Tuple[List[dict], List[dict], List[dict], set]:
    """Classify each study row and collect the set of referenced PDF basenames.

    Returns:
        all_present_rows, partial_rows, zero_rows, referenced_basenames
    """
    all_present: List[dict] = []
    partial: List[dict] = []
    zero: List[dict] = []
    referenced: set = set()

    for _, row in df.iterrows():
        cov_id = str(row.get(COL_COVIDENCE, "")).strip()
        if not cov_id:
            continue
        expected = parse_expected_pdfs(str(row.get(COL_PDFS, "")))
        found: List[str] = []
        missing: List[str] = []
        for pdf_name in expected:
            referenced.add(pdf_name)
            if pdf_name in disk_index:
                found.append(pdf_name)
            else:
                missing.append(pdf_name)

        entry = {
            "covidence_id": cov_id,
            "cita": str(row.get(COL_CITA, "")).strip(),
            "expected_pdfs": expected,
            "found": found,
            "missing": missing,
        }
        if not expected:
            zero.append(entry)
        elif not missing:
            all_present.append(entry)
        elif found and missing:
            partial.append(entry)
        else:
            zero.append(entry)

    return all_present, partial, zero, referenced


def find_orphans(
    disk_index: Dict[str, List[Path]],
    referenced: set,
) -> List[str]:
    """Return repo-relative paths of PDFs on disk not referenced by the CSV."""
    orphans: List[str] = []
    for basename, paths in disk_index.items():
        if basename in referenced:
            continue
        for p in paths:
            try:
                rel = p.relative_to(REPO_ROOT)
            except ValueError:
                rel = p
            orphans.append(str(rel))
    return sorted(orphans)


def detect_anomalies(df: pd.DataFrame) -> List[str]:
    """Detect duplicated or suspicious Covidence IDs (e.g., #B prefix)."""
    anomalies: List[str] = []
    ids = [str(x).strip() for x in df[COL_COVIDENCE].tolist() if str(x).strip()]
    seen: Dict[str, int] = defaultdict(int)
    for cid in ids:
        seen[cid] += 1
    duplicates = sorted([cid for cid, n in seen.items() if n > 1])
    if duplicates:
        anomalies.append(f"IDs duplicados: {duplicates}")
    weird_prefix = sorted([cid for cid in ids if not cid.startswith("#")])
    if weird_prefix:
        anomalies.append(f"IDs sin prefijo '#': {weird_prefix}")
    odd_chars = sorted([cid for cid in ids if cid.startswith("#") and not cid[1:].isdigit()])
    if odd_chars:
        anomalies.append(f"IDs con caracteres no numéricos tras '#': {odd_chars}")
    return anomalies


def _ascii_skeleton(name: str) -> str:
    """Return the ASCII-only skeleton of a filename (drops every non-ASCII byte).

    Useful as a heuristic to pair a roster filename with its mojibake-corrupted
    counterpart on disk: both strings should share the same ASCII prefix/suffix
    and only differ in the bytes representing accented characters.
    """
    return "".join(c for c in name if ord(c) < 128)


def find_suspected_mojibake_matches(
    missing_per_study: List[dict],
    orphans: List[str],
) -> List[dict]:
    """Heuristic: pair each missing PDF with an orphan having the same ASCII skeleton.

    Most missing-but-on-disk cases come from filenames with accented characters
    (é, ö, ü, etc.) that were stored with a corrupted encoding when the PDF
    archive was transferred from Windows. Strict basename match (per spec) flags
    them as missing + orphan; this diagnostic flags the most likely pairing so
    the user can confirm and rename manually if desired.
    """
    orphan_basenames: Dict[str, List[str]] = defaultdict(list)
    for rel in orphans:
        bn = Path(rel).name
        orphan_basenames[_ascii_skeleton(bn)].append(rel)

    matches: List[dict] = []
    for entry in missing_per_study:
        for miss in entry.get("missing", []):
            skel = _ascii_skeleton(miss)
            if not skel:
                continue
            candidates = orphan_basenames.get(skel, [])
            if candidates:
                matches.append({
                    "covidence_id": entry["covidence_id"],
                    "missing_roster_name": miss,
                    "candidate_disk_paths": candidates,
                })
    return matches


# --- Report assembly ---------------------------------------------------------

def build_report(
    df: pd.DataFrame,
    disk_index: Dict[str, List[Path]],
    all_present: List[dict],
    partial: List[dict],
    zero: List[dict],
    orphans: List[str],
    anomalies: List[str],
    mojibake_matches: List[dict],
) -> dict:
    total_pdfs_on_disk = sum(len(v) for v in disk_index.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(CSV_PATH.relative_to(REPO_ROOT)),
        "total_studies_in_roster": int(len(df)),
        "total_pdfs_on_disk": total_pdfs_on_disk,
        "summary": {
            "all_present": len(all_present),
            "partial": len(partial),
            "zero": len(zero),
        },
        "studies_zero": [
            {"covidence_id": s["covidence_id"], "expected_pdfs": s["expected_pdfs"]}
            for s in zero
        ],
        "studies_partial": [
            {
                "covidence_id": s["covidence_id"],
                "expected_pdfs": s["expected_pdfs"],
                "found": s["found"],
                "missing": s["missing"],
            }
            for s in partial
        ],
        "orphan_pdfs": orphans,
        "orphan_count": len(orphans),
        "anomalies": anomalies,
        "suspected_mojibake_matches": mojibake_matches,
    }


def print_summary(report: dict) -> None:
    print("=" * 70)
    print("VALIDACIÓN GOLD STANDARD DEPORTE — Roster CSV vs PDFs en disco")
    print("=" * 70)
    print(f"CSV:                 {report['csv_path']}")
    print(f"Generado:            {report['generated_at']}")
    print(f"Estudios en roster:  {report['total_studies_in_roster']}")
    print(f"PDFs en disco:       {report['total_pdfs_on_disk']}")
    print()
    s = report["summary"]
    print("Clasificación de estudios:")
    print(f"  all_present (todos los PDFs presentes): {s['all_present']}")
    print(f"  partial     (algunos PDFs faltan):      {s['partial']}")
    print(f"  zero        (ningún PDF en disco):      {s['zero']}")
    print()
    if report["studies_zero"]:
        print("Estudios SIN ningún PDF en disco (BLOCKER):")
        for entry in report["studies_zero"]:
            print(f"  - {entry['covidence_id']}  esperados={len(entry['expected_pdfs'])}")
    else:
        print("Estudios con cero PDFs: ninguno.")
    print()
    print(f"PDFs huérfanos (en disco, no en CSV): {report['orphan_count']}")
    if report["anomalies"]:
        print()
        print("Anomalías detectadas:")
        for a in report["anomalies"]:
            print(f"  - {a}")
    mojibake = report.get("suspected_mojibake_matches", [])
    if mojibake:
        print()
        print(f"Posibles emparejamientos por mojibake (encoding corrupto): {len(mojibake)}")
        print("  (los PDFs existen en disco pero con caracteres acentuados corruptos)")
        for m in mojibake:
            print(f"  - {m['covidence_id']}: {m['missing_roster_name']}")
    print("=" * 70)


# --- Main --------------------------------------------------------------------

def main(argv: Iterable[str] = ()) -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: no existe el CSV: {CSV_PATH}", file=sys.stderr)
        return 2
    if not GOLD_DIR.is_dir():
        print(f"ERROR: no existe el directorio gold: {GOLD_DIR}", file=sys.stderr)
        return 2

    df = load_roster(CSV_PATH)
    # Sanity check on column names.
    required_cols = {COL_COVIDENCE, COL_PDFS}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(
            f"ERROR: columnas faltantes en CSV: {sorted(missing_cols)}. "
            f"Columnas encontradas: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2

    disk_index = scan_pdfs_on_disk(GOLD_DIR)
    all_present, partial, zero, referenced = classify_studies(df, disk_index)
    orphans = find_orphans(disk_index, referenced)
    anomalies = detect_anomalies(df)
    mojibake_matches = find_suspected_mojibake_matches(partial + zero, orphans)

    report = build_report(
        df=df,
        disk_index=disk_index,
        all_present=all_present,
        partial=partial,
        zero=zero,
        orphans=orphans,
        anomalies=anomalies,
        mojibake_matches=mojibake_matches,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_summary(report)
    print(f"\nInforme JSON guardado en: {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
