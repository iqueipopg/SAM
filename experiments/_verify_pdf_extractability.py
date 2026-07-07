# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
Ad-hoc PDF extractability verifier for T16.D.6.A.bis Paso 2.

For each ``strata == 'include'`` row in
``data/gold_standard/deporte/samples/micro/sample_50.csv``:

  - Locate the PDF in ``data/gold_standard/deporte/pdfs_todos/``
  - Run three independent backends in cascade: pypdf, pdfminer, pdfplumber
  - Pick the backend with the largest extracted-char count
  - Flag is_extractable when chars_extracted > 100

Output: ``outputs/eval/micro/pdf_extractability_check.json``

This script does NOT touch agents/, config/, .env, or any other production code.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "samples" / "micro" / "sample_50.csv"
PDFS_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_todos"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "eval" / "micro" / "pdf_extractability_check.json"

CHAR_THRESHOLD = 100


def _load_includes() -> list[dict]:
    """Return the rows of sample_50.csv whose strata == 'include'."""
    if not SAMPLE_CSV.is_file():
        raise FileNotFoundError(f"sample_50.csv not found: {SAMPLE_CSV}")
    rows: list[dict] = []
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("strata") == "include":
                rows.append(row)
    return rows


def _extract_pypdf(path: Path) -> tuple[int, int, str | None]:
    """Return (pages_total, chars_extracted, error)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages = len(reader.pages)
        chars = 0
        for pg in reader.pages:
            try:
                txt = pg.extract_text() or ""
            except Exception:
                txt = ""
            chars += len(txt.strip())
        return pages, chars, None
    except Exception as e:
        return 0, 0, f"{type(e).__name__}: {e}"


def _extract_pdfminer(path: Path) -> tuple[int, int, str | None]:
    """Return (pages_total, chars_extracted, error). pdfminer doesn't expose
    pages_total cheaply, so we return 0 for it (the pypdf count is authoritative)."""
    try:
        from pdfminer.high_level import extract_text
        txt = extract_text(str(path)) or ""
        return 0, len(txt.strip()), None
    except Exception as e:
        return 0, 0, f"{type(e).__name__}: {e}"


def _extract_pdfplumber(path: Path) -> tuple[int, int, str | None]:
    """Return (pages_total, chars_extracted, error)."""
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = len(pdf.pages)
            chars = 0
            for pg in pdf.pages:
                try:
                    txt = pg.extract_text() or ""
                except Exception:
                    txt = ""
                chars += len(txt.strip())
        return pages, chars, None
    except Exception as e:
        return 0, 0, f"{type(e).__name__}: {e}"


def _verify_one(row: dict) -> dict:
    cid = row.get("covidence_id", "")
    pdf_filename = row.get("pdf_filename", "")
    pdf_path = PDFS_DIR / pdf_filename
    record: dict = {
        "covidence_id": cid,
        "pdf_filename": pdf_filename,
        "pdf_exists": pdf_path.is_file(),
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.is_file() else 0,
        "backends": {},
        "pages_total": 0,
        "chars_extracted": 0,
        "is_extractable": False,
        "backend_used": None,
        "elapsed_seconds": 0.0,
    }
    if not pdf_path.is_file():
        record["error"] = "pdf_not_found"
        return record

    t0 = time.time()
    backends = (
        ("pypdf", _extract_pypdf),
        ("pdfminer", _extract_pdfminer),
        ("pdfplumber", _extract_pdfplumber),
    )
    best_chars = 0
    best_backend: str | None = None
    pages_total = 0
    for name, fn in backends:
        pages, chars, err = fn(pdf_path)
        record["backends"][name] = {
            "pages": pages,
            "chars": chars,
            "error": err,
        }
        if pages > pages_total:
            pages_total = pages
        if chars > best_chars:
            best_chars = chars
            best_backend = name

    record["pages_total"] = pages_total
    record["chars_extracted"] = best_chars
    record["backend_used"] = best_backend
    record["is_extractable"] = best_chars > CHAR_THRESHOLD
    record["elapsed_seconds"] = round(time.time() - t0, 2)
    return record


def main() -> int:
    warnings.filterwarnings("ignore")
    print("=" * 72)
    print("Verificación extractabilidad PDFs - T16.D.6.A.bis Paso 2")
    print("=" * 72)
    print(f"sample_50 : {SAMPLE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"pdfs_dir  : {PDFS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"threshold : > {CHAR_THRESHOLD} chars => extractable")
    print()

    includes = _load_includes()
    print(f"Includes encontrados: {len(includes)}\n")

    records: list[dict] = []
    t_start = time.time()
    for i, row in enumerate(includes, 1):
        cid = row.get("covidence_id", "")
        fn = row.get("pdf_filename", "")
        print(f"[{i:2d}/{len(includes)}] {cid:8} ({fn[:60]}...)")
        rec = _verify_one(row)
        records.append(rec)
        mark = "OK" if rec["is_extractable"] else "FAIL"
        print(
            f"           pages={rec['pages_total']:3d}  "
            f"chars={rec['chars_extracted']:7d}  "
            f"backend={rec['backend_used'] or '-':10s}  "
            f"-> {mark}"
        )

    n_extr = sum(1 for r in records if r["is_extractable"])
    n_total = len(records)
    n_fail = n_total - n_extr

    envelope = {
        "task": "T16.D.6.A.bis Paso 2 - PDF extractability check",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample_csv": str(SAMPLE_CSV.relative_to(PROJECT_ROOT)),
        "pdfs_dir": str(PDFS_DIR.relative_to(PROJECT_ROOT)),
        "char_threshold": CHAR_THRESHOLD,
        "backends_used": ["pypdf", "pdfminer", "pdfplumber"],
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "n_includes_total": n_total,
        "n_extractable": n_extr,
        "n_non_extractable": n_fail,
        "non_extractable_ids": [
            r["covidence_id"] for r in records if not r["is_extractable"]
        ],
        "results": records,
        "elapsed_seconds_total": round(time.time() - t_start, 2),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 72)
    print("RESUMEN")
    print("=" * 72)
    print(f"Extractables   : {n_extr}/{n_total}")
    print(f"No-extractables: {n_fail}")
    if n_fail:
        print("\nNo-extractables (chars <= threshold):")
        for r in records:
            if not r["is_extractable"]:
                reason_bits = []
                for bname, bdata in r["backends"].items():
                    reason_bits.append(
                        f"{bname}={bdata['chars']}c"
                        + (f" err={bdata['error']}" if bdata["error"] else "")
                    )
                print(f"  {r['covidence_id']:8} {r['pdf_filename'][:60]:60} -> "
                      + ", ".join(reason_bits))
    print(f"\nElapsed total : {envelope['elapsed_seconds_total']}s")
    print(f"Output JSON   : {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
