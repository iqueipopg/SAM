# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
PDF extractability verifier for the medio samples (T16.E.1 Fase 2).

Reuses the cascade backends (pypdf -> pdfminer -> pdfplumber) of the micro
version. Operates over the union of unique PDFs referenced by
``sample_medio_a4.csv``, ``sample_medio_a5.csv`` and ``sample_medio_a6.csv``
(sample_medio_a3.csv is T/A-only so it does not need PDFs).

Thresholds:
  - >  1000 chars   ->  OK
  - 100-1000 chars  ->  marginal
  - <= 100 chars    ->  non_extractable (scanned/encrypted)

Outputs:
  - outputs/eval/medio/pdf_extractability_check.json
  - outputs/eval/medio/pdf_extractability_report.md

GUARDRAIL (caller responsibility):
  >20% non-extractable in any of the three samples => caller aborts.
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
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLES_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "samples" / "medio"
PDFS_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte" / "pdfs_todos"
OUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "medio"
OUT_JSON = OUT_DIR / "pdf_extractability_check.json"
OUT_MD = OUT_DIR / "pdf_extractability_report.md"

MARGINAL_THRESHOLD = 100
OK_THRESHOLD = 1000

SAMPLES_NEEDING_PDF = (
    ("a4", "sample_medio_a4.csv"),
    ("a5", "sample_medio_a5.csv"),
    ("a6", "sample_medio_a6.csv"),
)


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _extract_pypdf(path: Path) -> tuple[int, int, str | None]:
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
    try:
        from pdfminer.high_level import extract_text
        txt = extract_text(str(path)) or ""
        return 0, len(txt.strip()), None
    except Exception as e:
        return 0, 0, f"{type(e).__name__}: {e}"


def _extract_pdfplumber(path: Path) -> tuple[int, int, str | None]:
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


def _verify_one(pdf_filename: str) -> dict:
    pdf_path = PDFS_DIR / pdf_filename
    rec: dict = {
        "pdf_filename": pdf_filename,
        "pdf_exists": pdf_path.is_file(),
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.is_file() else 0,
        "backends": {},
        "pages_total": 0,
        "chars_extracted": 0,
        "best_backend": None,
        "classification": "missing",
        "elapsed_seconds": 0.0,
    }
    if not pdf_path.is_file():
        rec["error"] = "pdf_not_found"
        return rec
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
        rec["backends"][name] = {"pages": pages, "chars": chars, "error": err}
        if pages > pages_total:
            pages_total = pages
        if chars > best_chars:
            best_chars = chars
            best_backend = name
    rec["pages_total"] = pages_total
    rec["chars_extracted"] = best_chars
    rec["best_backend"] = best_backend
    if best_chars >= OK_THRESHOLD:
        rec["classification"] = "ok"
    elif best_chars >= MARGINAL_THRESHOLD:
        rec["classification"] = "marginal"
    else:
        rec["classification"] = "non_extractable"
    rec["elapsed_seconds"] = round(time.time() - t0, 2)
    return rec


def main() -> int:
    warnings.filterwarnings("ignore")
    print("=" * 72)
    print("PDF extractability check — medio samples (T16.E.1 Fase 2)")
    print("=" * 72)
    print(f"samples dir : {SAMPLES_DIR.relative_to(PROJECT_ROOT)}")
    print(f"pdfs dir    : {PDFS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"thresholds  : ok >={OK_THRESHOLD}c, marginal {MARGINAL_THRESHOLD}-{OK_THRESHOLD}c, "
          f"non_extractable <{MARGINAL_THRESHOLD}c")
    print()

    # Build union of PDFs referenced.
    per_sample_pdfs: Dict[str, List[str]] = {}
    union: Dict[str, List[str]] = {}  # pdf_filename -> list of samples that reference it
    sample_loads = {}
    for hero_id, fname in SAMPLES_NEEDING_PDF:
        rows = _load_rows(SAMPLES_DIR / fname)
        pdfs: List[str] = []
        for r in rows:
            pdf = (r.get("pdf_filename") or "").strip()
            if pdf:
                pdfs.append(pdf)
                union.setdefault(pdf, []).append(hero_id)
        per_sample_pdfs[hero_id] = pdfs
        sample_loads[hero_id] = len(rows)
        print(f"  sample {hero_id}: {len(rows)} rows, {len(pdfs)} pdf_filename non-empty")
    unique_pdfs = sorted(union.keys())
    print(f"  union unique PDFs : {len(unique_pdfs)}")
    print()

    # Verify all.
    results: List[dict] = []
    t_start = time.time()
    for i, pdf in enumerate(unique_pdfs, 1):
        rec = _verify_one(pdf)
        rec["used_by"] = union.get(pdf, [])
        results.append(rec)
        cls = rec["classification"]
        mark = {"ok": "OK", "marginal": "MARG", "non_extractable": "FAIL", "missing": "MISS"}[cls]
        if i <= 10 or i % 50 == 0:
            print(
                f"  [{i:4d}/{len(unique_pdfs)}] {pdf[:65]:65} "
                f"chars={rec['chars_extracted']:>6d}  -> {mark}"
            )

    # Aggregate.
    by_class = {"ok": 0, "marginal": 0, "non_extractable": 0, "missing": 0}
    by_class_per_sample: Dict[str, Dict[str, int]] = {
        h: {"ok": 0, "marginal": 0, "non_extractable": 0, "missing": 0}
        for h, _ in SAMPLES_NEEDING_PDF
    }
    for rec in results:
        cls = rec["classification"]
        by_class[cls] += 1
        for h in rec["used_by"]:
            by_class_per_sample[h][cls] += 1

    # Per-sample totals (note: rows count includes empty pdf_filename rows
    # which we ignore for ratios but we surface them).
    per_sample_totals_pdfs: Dict[str, int] = {
        h: len(per_sample_pdfs[h]) for h, _ in SAMPLES_NEEDING_PDF
    }
    # Per-sample non-extractable ratio.
    per_sample_ratio: Dict[str, float] = {}
    guardrail_breaches: List[str] = []
    for h, _ in SAMPLES_NEEDING_PDF:
        denom = per_sample_totals_pdfs[h]
        # Count non_extractable + missing as effective failures for ratio.
        bad = by_class_per_sample[h]["non_extractable"] + by_class_per_sample[h]["missing"]
        ratio = bad / denom if denom > 0 else 0.0
        per_sample_ratio[h] = round(ratio, 4)
        if ratio > 0.20:
            guardrail_breaches.append(
                f"sample {h}: {bad}/{denom} ({ratio*100:.1f}%) > 20%"
            )

    envelope = {
        "task": "T16.E.1 Fase 2 - PDF extractability medio",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "thresholds": {
            "marginal_chars": MARGINAL_THRESHOLD,
            "ok_chars": OK_THRESHOLD,
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "n_unique_pdfs": len(unique_pdfs),
        "by_classification_global": by_class,
        "by_classification_per_sample": by_class_per_sample,
        "per_sample_rows_with_pdf": per_sample_totals_pdfs,
        "per_sample_nonextractable_ratio": per_sample_ratio,
        "guardrail_threshold_pct": 20.0,
        "guardrail_breaches": guardrail_breaches,
        "non_extractable_files": [
            {
                "pdf_filename": r["pdf_filename"],
                "chars_extracted": r["chars_extracted"],
                "used_by": r["used_by"],
                "classification": r["classification"],
            }
            for r in results
            if r["classification"] in ("non_extractable", "missing")
        ],
        "marginal_files": [
            {
                "pdf_filename": r["pdf_filename"],
                "chars_extracted": r["chars_extracted"],
                "used_by": r["used_by"],
            }
            for r in results
            if r["classification"] == "marginal"
        ],
        "results": results,
        "elapsed_seconds_total": round(time.time() - t_start, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False)

    # Markdown report.
    md_lines: List[str] = []
    md_lines.append("# PDF extractability — sample medio (T16.E.1 Fase 2)")
    md_lines.append("")
    md_lines.append(f"Fecha UTC: `{envelope['timestamp_utc']}`")
    md_lines.append(
        f"Umbrales: OK >={OK_THRESHOLD} chars, marginal {MARGINAL_THRESHOLD}-{OK_THRESHOLD}, "
        f"non_extractable <{MARGINAL_THRESHOLD}."
    )
    md_lines.append(f"Tiempo total: {envelope['elapsed_seconds_total']}s.")
    md_lines.append("")
    md_lines.append("## Distribucion global")
    md_lines.append("")
    md_lines.append(f"- PDFs unicos verificados: **{envelope['n_unique_pdfs']}**")
    md_lines.append(f"- OK         : {by_class['ok']}")
    md_lines.append(f"- Marginales : {by_class['marginal']}")
    md_lines.append(f"- No extraibles : {by_class['non_extractable']}")
    md_lines.append(f"- Ausentes en disco: {by_class['missing']}")
    md_lines.append("")
    md_lines.append("## Por sample")
    md_lines.append("")
    md_lines.append("| Sample | Filas con PDF | OK | Marg | NonExtr | Missing | % bad |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for hero_id, _ in SAMPLES_NEEDING_PDF:
        c = by_class_per_sample[hero_id]
        denom = per_sample_totals_pdfs[hero_id]
        ratio = per_sample_ratio[hero_id] * 100
        md_lines.append(
            f"| medio_{hero_id} | {denom} | {c['ok']} | {c['marginal']} | "
            f"{c['non_extractable']} | {c['missing']} | {ratio:.2f}% |"
        )
    md_lines.append("")
    if guardrail_breaches:
        md_lines.append("## GUARDRAIL DISPARADO (>20% non-extractable + missing)")
        md_lines.append("")
        for line in guardrail_breaches:
            md_lines.append(f"- {line}")
        md_lines.append("")
    else:
        md_lines.append("## GUARDRAIL")
        md_lines.append("")
        md_lines.append("Sin breaches; todos los samples bajo 20% non-extractable.")
        md_lines.append("")

    if envelope["non_extractable_files"]:
        md_lines.append("## PDFs no extraibles / ausentes")
        md_lines.append("")
        md_lines.append("| PDF | chars | clas. | used_by |")
        md_lines.append("|---|---:|---|---|")
        for f in envelope["non_extractable_files"]:
            md_lines.append(
                f"| `{f['pdf_filename'][:80]}` | {f['chars_extracted']} | "
                f"{f['classification']} | {','.join(f['used_by'])} |"
            )
        md_lines.append("")

    if envelope["marginal_files"]:
        md_lines.append("## PDFs marginales (100-1000 chars)")
        md_lines.append("")
        md_lines.append("| PDF | chars | used_by |")
        md_lines.append("|---|---:|---|")
        for f in envelope["marginal_files"]:
            md_lines.append(
                f"| `{f['pdf_filename'][:80]}` | {f['chars_extracted']} | "
                f"{','.join(f['used_by'])} |"
            )
        md_lines.append("")

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print()
    print("=" * 72)
    print("RESUMEN")
    print("=" * 72)
    print(f"PDFs unicos       : {envelope['n_unique_pdfs']}")
    print(f"OK                : {by_class['ok']}")
    print(f"Marginales        : {by_class['marginal']}")
    print(f"No extraibles     : {by_class['non_extractable']}")
    print(f"Ausentes en disco : {by_class['missing']}")
    print()
    for h, _ in SAMPLES_NEEDING_PDF:
        c = by_class_per_sample[h]
        denom = per_sample_totals_pdfs[h]
        ratio = per_sample_ratio[h] * 100
        print(f"  medio_{h}: {denom} pdfs, OK={c['ok']} marg={c['marginal']} "
              f"nonExtr={c['non_extractable']} miss={c['missing']} ({ratio:.2f}% bad)")
    print()
    if guardrail_breaches:
        print("[GUARDRAIL] >20% non-extractable. ABORTAR antes de Fase 4.")
        for line in guardrail_breaches:
            print(f"  - {line}")
    else:
        print("[GUARDRAIL] OK: <20% non-extractable en todos los samples.")
    print(f"Output JSON: {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Output MD  : {OUT_MD.relative_to(PROJECT_ROOT)}")

    return 0 if not guardrail_breaches else 1


if __name__ == "__main__":
    sys.exit(main())
