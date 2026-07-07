# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Parser: extracts Risk-of-Bias table from sup2.pdf (Promoting PA / HPR 2025).

Source PDF:
    data/gold_standard/deporte/promoting_PA_HPR_2025/sup2.pdf

The PDF contains "Table 1. Outcomes' characteristics and risk of bias per study"
spanning pages 1-5 (page 5 also starts Table 2 which is out of scope). Each row
describes one outcome of one study with columns:
    Author, year, reference # | # outcomes | Outcomes | Obj/sf | Assessment tool | RoB

RoB values are L (low), SC (some concerns) or H (high).

Approach
--------
1. pdfplumber.extract_tables() returns 0 tables (the PDF uses whitespace, no rules),
   so we use extract_words() and bucket every word into one of six columns by its
   x-coordinate. Column boundaries derived from header positions:
        author      x in [ 80, 220)
        n_outcomes  x in [220, 235)
        outcome     x in [235, 370)
        obj_sf      x in [370, 425)
        tool        x in [425, 540)
        rob         x in [540, 600)
2. Words are grouped into visual lines by `top` (rounded to ~6pt buckets). A
   "logical row" of the table can span several visual lines because the assessment
   tool, the author block, or the RoB cell may wrap. We anchor logical rows on the
   RoB cell (always exactly one of {L, SC, H}); every preceding line without a RoB
   token is merged into the next line that has one.
3. The author column propagates: continuation rows of the same study leave the
   author cell blank, so we forward-fill once we see a non-empty author string
   that contains a reference number (e.g. "[1]" or "[36,37]").

Note on covidence_id
--------------------
sup2.pdf does NOT include Covidence IDs (those came from sup1 / Paula's spreadsheet
processed by C.3). The closest identifier available here is the author-year string
plus the reference number in brackets (e.g. "Hartman et al. (2016) [1]"). The CSV
emits an empty covidence_id column for compatibility with the requested schema
plus a study_ref + ref_number that downstream code can join against gold_metrics.

Output
------
    data/gold_standard_normalized/gold_rob.csv

Run
---
    ~/meta-analisis-automatizado/venv/bin/python scripts/parse_gold_rob.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = (
    REPO_ROOT
    / "data"
    / "gold_standard"
    / "deporte"
    / "promoting_PA_HPR_2025"
    / "sup2.pdf"
)
OUT_DIR = REPO_ROOT / "data" / "gold_standard_normalized"
OUT_CSV = OUT_DIR / "gold_rob.csv"

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
# (lo, hi) x-coordinate boundaries per column, derived from header positions.
COLUMNS: dict[str, tuple[float, float]] = {
    "author": (80.0, 220.0),
    "n_outcomes": (220.0, 235.0),
    "outcome": (235.0, 370.0),
    "obj_sf": (370.0, 425.0),
    "tool": (425.0, 540.0),
    "rob": (540.0, 600.0),
}

# Table 1 lives on pages 1..5 (Table 2 starts mid-page-5 with a header line).
ROB_PAGES = range(0, 5)  # 0-indexed → pages 1..5 inclusive

# RoB cell tokens; anything else is treated as continuation noise.
ROB_TOKENS = {"L", "SC", "H"}

# Lines below this are page footers / Table-2 header on page 5.
TOP_BODY_MAX = 760.0

# Per-page "top" cutoffs (anything ABOVE this y is the page header / table-title
# and gets skipped). Page 1 has the document title + the column-header row, so
# its body starts noticeably lower; pages 2-5 plunge straight into table rows.
PAGE_TOP_BODY_MIN: dict[int, float] = {
    0: 165.0,  # page 1: skip "Supplementary Material 2" + "Table 1." + col headers
    1: 80.0,   # page 2
    2: 80.0,   # page 3
    3: 80.0,   # page 4
    4: 80.0,   # page 5 (Table 2 boundary handled separately)
}

# Reference-number bracket pattern (e.g. "[1]", "[36,37]", "[3–5]")
REF_BRACKET_RE = re.compile(r"\[([\d,–\-]+)\]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def column_of(x: float) -> str | None:
    for name, (lo, hi) in COLUMNS.items():
        if lo <= x < hi:
            return name
    return None


def bucket_top(top: float) -> int:
    """Round top coordinate into a 6pt bucket so words on the same visual line group together."""
    return int(round(top / 6.0))


def extract_visual_lines(page, top_min: float) -> list[dict[str, str]]:
    """Return a list of visual lines (one per y-bucket) with text per column."""
    buckets: dict[int, dict[str, list[tuple[float, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for w in page.extract_words():
        top = w["top"]
        if top < top_min or top > TOP_BODY_MAX:
            continue
        col = column_of(w["x0"])
        if col is None:
            continue
        # Detect end-of-Table-1 marker on page 5:
        text = w["text"]
        if text.startswith("Table") and col == "author":
            # "Table 2." header — stop processing this page.
            return _finalize_lines(buckets, stop_top_bucket=bucket_top(top))
        # Detect the "Note." footer that closes Table 1 (skip footnotes).
        if text == "Note." and col == "author":
            return _finalize_lines(buckets, stop_top_bucket=bucket_top(top))
        buckets[bucket_top(top)][col].append((w["x0"], text))

    return _finalize_lines(buckets, stop_top_bucket=None)


def _finalize_lines(
    buckets: dict[int, dict[str, list[tuple[float, str]]]],
    stop_top_bucket: int | None,
) -> list[dict[str, str]]:
    lines: list[dict[str, str]] = []
    for key in sorted(buckets.keys()):
        if stop_top_bucket is not None and key >= stop_top_bucket:
            break
        row = buckets[key]
        line: dict[str, str] = {}
        for col in COLUMNS:
            words = sorted(row.get(col, []))
            line[col] = " ".join(t for _, t in words).strip()
        line["_top"] = key  # type: ignore[assignment]
        lines.append(line)
    return lines


def consolidate_into_logical_rows(visual_lines: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge visual lines into logical table rows, anchored on the RoB cell.

    Multi-line cells use the following heuristic, derived from observed y-spacing
    in sup2.pdf (anchored rows are 12pt apart; intra-cell wraps land 6pt away):

    * **Post-anchor tool wrap**: a tool-only visual line that sits ~6pt below a
      just-flushed row → append to that previous row's ``tool``.
    * **Pre-anchor wraps** (author or tool or outcome wraps that appear BEFORE
      the next anchor) accumulate naturally into the buffer of the next row.
    * Other cells (author wraps, outcome wraps, etc.) accumulate into the current
      buffer until a RoB token closes the row.
    """
    logical: list[dict[str, str]] = []
    buffer: dict[str, list[str]] = defaultdict(list)
    # Track the y-bucket of the most recently flushed line so we can decide
    # whether a tool-only continuation belongs to it (post-anchor wrap).
    last_flushed_top: int | None = None
    # Half-line tolerance: a continuation that lands within this many y-bucket
    # units of the flushed line is treated as a post-anchor wrap. Buckets are
    # 6pt wide and the y-key is the result of round(top/6), so Δ=1 (~6pt) is a
    # half-line wrap and Δ=2 (~12pt) is the next data row.
    POST_ANCHOR_GAP = 1

    def flush_buffer(rob_token: str, top: int) -> None:
        nonlocal last_flushed_top
        row = {col: " ".join(buffer[col]).strip() for col in COLUMNS}
        row["rob"] = rob_token
        # Skip empty rows: a RoB anchor with no outcome/tool content is a
        # stray duplicate left over by a vertically-merged RoB cell in the
        # source PDF — drop it but keep the y-anchor so subsequent wrap
        # detection still works.
        # A row needs concrete outcome content; otherwise it's the stray top
        # of a vertically-merged RoB cell, and we drop it. We still update
        # last_flushed_top so subsequent wrap detection has a y-anchor.
        has_content = bool(row["outcome"] or row["tool"])
        if has_content:
            logical.append(row)
        last_flushed_top = top
        buffer.clear()

    AUTHOR_REF_ONLY_RE = re.compile(r"^\[[\d,–\-]+\]$")

    def is_post_anchor_wrap(vline: dict[str, str]) -> bool:
        """A line that wraps content for the JUST-flushed row.

        Heuristic: only the ``tool`` column (and optionally a bracketed ref like
        ``[36,37]`` in the author column, which is just a hanging reference of
        the previous study) has content. A fresh outcome + obj_sf would signal a
        NEW data row whose RoB lands on a sub-line below — do not merge.
        """
        author_is_ref_wrap = (
            not vline["author"]
            or bool(AUTHOR_REF_ONLY_RE.match(vline["author"].strip()))
        )
        return (
            bool(vline["tool"] or AUTHOR_REF_ONLY_RE.match(vline["author"].strip() or ""))
            and author_is_ref_wrap
            and not vline["n_outcomes"]
            and not vline["outcome"]
            and not vline["obj_sf"]
            and not vline["rob"].strip()
        )

    for vline in visual_lines:
        top = int(vline["_top"])  # type: ignore[arg-type]
        rob_raw = vline["rob"].strip()
        rob_tokens_in_cell = [t for t in rob_raw.split() if t]
        rob_clean = next((t for t in rob_tokens_in_cell if t in ROB_TOKENS), None)

        # Is this a post-anchor wrap of the just-flushed row? We require:
        #   * a small y-gap (1 bucket = 6pt) to the previous anchor
        #   * the buffer for the upcoming row has not started yet
        buffer_empty = not any(buffer[c] for c in COLUMNS)
        if (
            is_post_anchor_wrap(vline)
            and logical
            and buffer_empty
            and last_flushed_top is not None
            and (top - last_flushed_top) <= POST_ANCHOR_GAP
        ):
            prev = logical[-1]
            for col in ("outcome", "obj_sf", "tool"):
                if vline[col]:
                    prev[col] = (prev[col] + " " + vline[col]).strip()
            # Author ref wraps (e.g. "[36,37]") attach to the previous author.
            if vline["author"] and AUTHOR_REF_ONLY_RE.match(vline["author"].strip()):
                prev["author"] = (prev["author"] + " " + vline["author"]).strip()
            continue

        for col in ("author", "n_outcomes", "outcome", "obj_sf", "tool"):
            if vline[col]:
                buffer[col].append(vline[col])

        if rob_clean is not None:
            flush_buffer(rob_clean, top)

    # Buffer with unflushed content = stray footer text → discard silently.
    return logical


def forward_fill_author(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Propagate the last non-empty author/study reference to continuation rows."""
    last_author = ""
    for row in rows:
        if row["author"]:
            last_author = row["author"]
        else:
            row["author"] = last_author
    return rows


def extract_ref_number(author: str) -> str:
    m = REF_BRACKET_RE.search(author)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Quality assessment helpers
# ---------------------------------------------------------------------------
_DUAL_OBJSF_RE = re.compile(
    r"^(Objective|Self-report) +(Objective|Self-report)$"
)


def assess_quality(rows: list[dict[str, str]]) -> tuple[bool, dict[str, object]]:
    """Return (is_clean, stats). Clean = >=80% rows with all 4 mandatory fields."""
    total = len(rows)
    complete_count = 0
    rob_counter: Counter[str] = Counter()
    anomalies: list[str] = []
    for i, r in enumerate(rows, start=2):  # +2 to align with 1-indexed CSV including header
        rob_counter[r["rob_judgment"]] += 1
        if all([r["study_ref"], r["outcome"], r["instrument_human"], r["rob_judgment"]]):
            complete_count += 1
        # Flag rows whose obj_sf column shows TWO labels — these are residual
        # cases where the source PDF visually merged a RoB cell across two
        # adjacent data rows; we merged them into one CSV row.
        if _DUAL_OBJSF_RE.match(r["obj_sf"].strip()):
            anomalies.append(
                f"row {i}: dual-outcome merge in '{r['study_ref']}' "
                f"(outcome='{r['outcome']}')"
            )
    pct_complete = (complete_count / total) if total else 0.0
    pct_known_rob = (
        sum(rob_counter[k] for k in ROB_TOKENS) / total if total else 0.0
    )
    stats = {
        "total_rows": total,
        "complete_rows": complete_count,
        "pct_complete": round(pct_complete * 100, 1),
        "rob_breakdown": dict(rob_counter),
        "pct_known_rob": round(pct_known_rob * 100, 1),
        "anomalies": anomalies,
    }
    is_clean = pct_complete >= 0.80 and pct_known_rob >= 0.80
    return is_clean, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse() -> list[dict[str, str]]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Source PDF not found: {PDF_PATH}")

    rows_out: list[dict[str, str]] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        all_logical: list[dict[str, str]] = []
        for page_idx in ROB_PAGES:
            if page_idx >= len(pdf.pages):
                break
            page = pdf.pages[page_idx]
            top_min = PAGE_TOP_BODY_MIN.get(page_idx, 80.0)
            visual_lines = extract_visual_lines(page, top_min)
            logical = consolidate_into_logical_rows(visual_lines)
            all_logical.extend(logical)

    all_logical = forward_fill_author(all_logical)

    for r in all_logical:
        rows_out.append(
            {
                "covidence_id": "",  # not present in sup2.pdf
                "study_ref": r["author"],
                "ref_number": extract_ref_number(r["author"]),
                "outcome": r["outcome"],
                "obj_sf": r["obj_sf"],
                "instrument_human": r["tool"],
                "rob_judgment": r["rob"],
            }
        )
    return rows_out


def write_csv(rows: list[dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "covidence_id",
        "study_ref",
        "ref_number",
        "outcome",
        "obj_sf",
        "instrument_human",
        "rob_judgment",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    print(f"[parse_gold_rob] Reading {PDF_PATH}")
    try:
        rows = parse()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    is_clean, stats = assess_quality(rows)
    write_csv(rows)

    print(f"[parse_gold_rob] Extracted {stats['total_rows']} rows -> {OUT_CSV}")
    print(f"[parse_gold_rob] Complete rows: {stats['complete_rows']} "
          f"({stats['pct_complete']}%)")
    print(f"[parse_gold_rob] RoB breakdown: {stats['rob_breakdown']} "
          f"(known {stats['pct_known_rob']}%)")
    anomalies = stats["anomalies"]  # type: ignore[assignment]
    if anomalies:
        print(f"[parse_gold_rob] Anomalies ({len(anomalies)}):")
        for a in anomalies:
            print(f"  - {a}")
    if is_clean:
        print("[parse_gold_rob] VERDICT: CLEAN extraction (>=80% rows complete).")
    else:
        print("[parse_gold_rob] VERDICT: DIRTY extraction (<80% rows complete). "
              "Review CSV and/or flag as deuda.")
    return 0 if is_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
