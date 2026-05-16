#!/usr/bin/env python3
"""
recover_failed_metadata.py — T16.D.3.bis

Recover the 78 deporte gold-standard studies that ended up with
`source_method == "failed"` after T16.D.3.A + T16.D.3.A.bis. Two focused
methods are layered on top of the existing extractor output, without
touching the 1474 successful rows:

    Method 0.5 — Mojibake repair in the filename manifest. The PDFs whose
        xlsx filenames contain non-ASCII characters were saved on disk with
        their UTF-8 bytes re-decoded under an 8-bit code page (CP866 on
        this Spanish/Russian-DOS environment, CP1252 / CP850 on other
        layouts). The standard NFKD+drop-combining skeleton in
        `extract_metadata_from_pdfs.py` doesn't bridge this gap because the
        on-disk characters (e.g. "├й") are out-of-script and get dropped to
        nothing while the xlsx side keeps them as "e". This method walks
        the PDF directory, tries `encode(cp).decode("utf-8")` over a
        cascade of code pages, and matches repaired skeletons against the
        xlsx skeletons. If a PDF is found, the original extraction
        pipeline (Methods 1-5) is re-run on it.

    Method 6 — Crossref by title. For every failed study still without
        title/DOI after Method 0.5, parse `author / year / title` from the
        xlsx filename, query `https://api.crossref.org/works` with
        `query.title` + `query.author`, and accept the top hit when
        `score >= 80` AND |response_year - filename_year| <= 1.
        - Accepted hits become `source_method = "crossref_by_title"`.
        - Rejected hits become `source_method = "crossref_by_title_low_confidence"`
          and stash the candidate metadata in `warnings` (no main field
          population).

The CSV schema is unchanged: 10 columns, only two new values are added to
the `source_method` enum. The 1474 non-failed rows are written through
untouched.

Identifiers and code comments in English. Logs in Spanish per the project rules
rule #12.
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# Make `scripts/extract_metadata_from_pdfs.py` importable when this file is
# executed directly (PYTHONPATH may not include the project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_metadata_from_pdfs import (  # noqa: E402
    OUTPUT_COLUMNS,
    OUTPUT_CSV,
    PDFS_DIR,
    XLSX_COL_FILENAME,
    XLSX_COL_ID,
    XLSX_COL_STATUS,
    XLSX_PATH,
    _CROSSREF_SLEEP,
    _USER_AGENT,
    _ascii_skeleton,
    _join_warnings,
    fetch_abstract_semantic_scholar,
    process_pdf,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Code page cascade for Method 0.5. CP866 covers our actual data
# (Spanish characters saved under Russian DOS rendering); CP1252 and CP850
# are kept first because the task spec mentions them and they are common
# Windows / Western DOS layouts.
MOJIBAKE_CODEPAGES = ("cp1252", "cp850", "cp866", "cp437", "cp1251")

# Crossref-by-title (Method 6) thresholds
CROSSREF_SEARCH_BASE = "https://api.crossref.org/works"
CROSSREF_TITLE_MIN_SCORE = 80.0
CROSSREF_TITLE_YEAR_TOL = 1
CROSSREF_TITLE_TIMEOUT = 20

YEAR_REGEX = re.compile(r"(19[5-9]\d|20[0-2]\d)")
# Strip trailing `_NN.pdf` index suffix and the `.pdf` extension
INDEX_SUFFIX_REGEX = re.compile(r"_\d+\.pdf$", re.IGNORECASE)
PDF_SUFFIX_REGEX = re.compile(r"\.pdf$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Method 0.5 — Mojibake repair
# ---------------------------------------------------------------------------
def repair_mojibake_filename(disk_name: str) -> Optional[str]:
    """Try to reverse double-decoding of `disk_name`.

    Returns the repaired string (decoded as UTF-8 after re-encoding via a
    candidate 8-bit code page) or None if no code page round-trips.
    """
    for cp in MOJIBAKE_CODEPAGES:
        try:
            return disk_name.encode(cp, errors="strict").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return None


def build_repaired_skeleton_index(
    pdfs_dir: Path,
) -> dict[str, str]:
    """Walk `pdfs_dir`; for each file try mojibake repair, then keep a map
    from the repaired ASCII skeleton to the actual on-disk filename.

    Both the original and the repaired skeletons are stored so that the
    standard match path stays a no-op for ASCII-clean files (their original
    skeleton already gets registered by the base extractor's manifest).
    """
    repaired_index: dict[str, str] = {}
    for path in pdfs_dir.iterdir():
        if not path.is_file():
            continue
        disk_name = path.name
        if all(ord(c) < 128 for c in disk_name):
            # Pure ASCII filename: no mojibake possible, the base manifest
            # already matched this byte-exact. Skip to keep the index small.
            continue
        repaired = repair_mojibake_filename(disk_name)
        if not repaired:
            continue
        rep_skel = _ascii_skeleton(repaired)
        if not rep_skel:
            continue
        # Keep the first occurrence deterministically.
        repaired_index.setdefault(rep_skel, disk_name)
    return repaired_index


def find_pdf_via_mojibake(
    xlsx_filename: str,
    repaired_index: dict[str, str],
) -> Optional[str]:
    """Return the actual on-disk filename matching `xlsx_filename` via the
    mojibake repair index, or None."""
    if not xlsx_filename:
        return None
    target_skel = _ascii_skeleton(xlsx_filename)
    if not target_skel:
        return None
    return repaired_index.get(target_skel)


# ---------------------------------------------------------------------------
# Method 6 — Crossref by title (filename parsing + Crossref search)
# ---------------------------------------------------------------------------
def parse_filename(xlsx_filename: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Parse `Author year - Title.pdf` style filenames into (author, year, title).

    Supported formats observed in the gold-standard xlsx include:
        "Bennett JA et al. - 2007 - Motivational interviewing to increase...pdf"
        "Bennett-2007-Motivational interviewing to increase...pdf"
        "2007 - Motivational interviewing to increase...pdf"
        "Bennett et al. - 2007 - Motivational interviewing to...pdf"

    Returns (author_or_None, year_or_None, title_or_None).
    """
    if not xlsx_filename or not isinstance(xlsx_filename, str):
        return None, None, None

    name = xlsx_filename.strip()
    # Strip `_NN.pdf` index suffix, then `.pdf`
    name = INDEX_SUFFIX_REGEX.sub("", name)
    name = PDF_SUFFIX_REGEX.sub("", name)
    # Normalize underscores to spaces (T16.D.3.A.bis convention) but keep
    # hyphens because hyphenated separators carry meaning here.
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    # Locate the year
    year_match = YEAR_REGEX.search(name)
    if not year_match:
        return None, None, None
    year = int(year_match.group(0))
    pre = name[: year_match.start()].strip(" -")
    post = name[year_match.end() :].strip(" -")

    # Author: everything before the year. Drop "et al." and trailing
    # initials. Take only the leading surname.
    author: Optional[str] = None
    if pre:
        # Remove "et al." (case insensitive) and trailing initials/letters
        author_raw = re.sub(r"\bet\s+al\.?", "", pre, flags=re.IGNORECASE).strip()
        author_raw = author_raw.rstrip(" -,")
        if author_raw:
            # Take the first whitespace token (handles hyphenated surnames
            # like "Jansson-Frojmark" because hyphens are kept inside the
            # token; we split on whitespace only).
            first_token = author_raw.split()[0]
            # Strip trailing punctuation
            first_token = first_token.strip(" .,;:")
            if len(first_token) >= 2:
                author = first_token

    # Title: everything after the year. Drop leading separators.
    title: Optional[str] = post if post else None
    if title:
        title = title.strip(" -:")
        # Some titles end with "...pdf" debris if the extension regex missed
        # (defensive only).
        title = PDF_SUFFIX_REGEX.sub("", title).strip()
        # Replace stray underscores already handled, but trailing trailing
        # ellipsis from truncated filenames is fine for fuzzy search.
        if len(title) < 5:
            title = None

    return author, year, title


class CrossrefTitleRetryable(Exception):
    """Retryable Crossref-by-title failure."""


def _is_retryable_ct(exc: BaseException) -> bool:
    return isinstance(exc, CrossrefTitleRetryable)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_retryable_ct),
)
def _crossref_search_title(title: str, author: Optional[str]) -> dict:
    """GET `/works?query.title=...&query.author=...&rows=1`.

    Raises CrossrefTitleRetryable on 429/5xx/timeout/connection errors.
    Returns the JSON body on 200.
    """
    headers = {"User-Agent": _USER_AGENT}
    params: dict[str, str] = {"query.title": title, "rows": "1"}
    if author:
        params["query.author"] = author
    try:
        resp = requests.get(
            CROSSREF_SEARCH_BASE,
            params=params,
            headers=headers,
            timeout=CROSSREF_TITLE_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise CrossrefTitleRetryable(str(exc)) from exc

    if resp.status_code in (429,) or 500 <= resp.status_code < 600:
        raise CrossrefTitleRetryable(f"crossref-search HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise CrossrefTitleRetryable(
            f"crossref-search unexpected HTTP {resp.status_code}"
        )
    try:
        return resp.json() or {}
    except ValueError as exc:
        raise CrossrefTitleRetryable(f"crossref-search JSON parse: {exc}") from exc


def _format_author(author: dict) -> Optional[str]:
    """Format `{family, given}` -> 'Family AB' style (mirror of extractor)."""
    family = (author.get("family") or "").strip()
    given = (author.get("given") or "").strip()
    if not family and not given:
        name = (author.get("name") or "").strip()
        return name or None
    initials = []
    for token in re.split(r"[\s\.\-]+", given):
        token = token.strip()
        if not token:
            continue
        initials.append(token[0].upper())
    initials_str = "".join(initials)
    if family and initials_str:
        return f"{family} {initials_str}"
    return family or initials_str or None


def _strip_jats(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _year_from_item(item: dict) -> Optional[int]:
    """Extract a year integer from a Crossref item, trying common date keys
    in the order requested by the task spec."""
    for key in (
        "published",
        "published-online",
        "published-print",
        "created",
        "issued",
    ):
        block = item.get(key)
        if not block:
            continue
        date_parts = block.get("date-parts") if isinstance(block, dict) else None
        if date_parts and date_parts[0]:
            try:
                year = int(date_parts[0][0])
            except (TypeError, ValueError):
                continue
            if 1900 <= year <= 2100:
                return year
    return None


def _normalize_doi(doi: str) -> str:
    """Lowercase + strip URL prefix variants, mirroring the extractor."""
    d = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix) :]
            break
    return d.rstrip(".,;)")


def crossref_by_title(
    title: str,
    author: Optional[str],
    year_filename: int,
) -> tuple[Optional[dict], str]:
    """Run a Crossref title search and decide acceptance.

    Returns (result_dict_or_None, decision_str). `decision_str` is one of:
        "accepted"
        "low_confidence" — result returned but failed score/year check
        "no_result"      — Crossref returned 200 with empty items list
        "unreachable: <details>" — retries exhausted
    The result dict shape (when accepted or low-confidence):
        {
            "title": str,
            "authors": str | None,
            "abstract": str | None,
            "abstract_source": "crossref" | None,
            "doi": str | None,
            "year": int | None,
            "score": float,
            "year_diff": int | None,
        }
    Sleeps `_CROSSREF_SLEEP` after every call.
    """
    try:
        data = _crossref_search_title(title, author)
    except CrossrefTitleRetryable as exc:
        time.sleep(_CROSSREF_SLEEP)
        return None, f"unreachable: {exc}"
    except RetryError as exc:
        time.sleep(_CROSSREF_SLEEP)
        return None, f"unreachable: {exc}"
    except Exception as exc:  # noqa: BLE001
        time.sleep(_CROSSREF_SLEEP)
        return None, f"unreachable: {type(exc).__name__}: {exc}"

    time.sleep(_CROSSREF_SLEEP)

    message = (data or {}).get("message", {})
    items = message.get("items") or []
    if not items:
        return None, "no_result"
    item = items[0]

    score = float(item.get("score") or 0.0)
    cr_year = _year_from_item(item)
    year_diff: Optional[int] = (
        abs(cr_year - year_filename) if cr_year is not None else None
    )

    # Build result payload
    cr_titles = item.get("title") or []
    cr_title: Optional[str] = None
    if isinstance(cr_titles, list) and cr_titles:
        cr_title = str(cr_titles[0]).strip() or None
    elif isinstance(cr_titles, str):
        cr_title = cr_titles.strip() or None

    authors_data = item.get("author") or []
    formatted = [_format_author(a) for a in authors_data if isinstance(a, dict)]
    formatted = [f for f in formatted if f]
    cr_authors = "; ".join(formatted) if formatted else None

    cr_doi = item.get("DOI")
    cr_doi = _normalize_doi(str(cr_doi)) if cr_doi else None

    cr_abstract: Optional[str] = None
    abs_source: Optional[str] = None
    raw_abs = item.get("abstract")
    if raw_abs:
        cleaned = _strip_jats(str(raw_abs))
        if cleaned:
            cr_abstract = cleaned
            abs_source = "crossref"

    payload = {
        "title": cr_title,
        "authors": cr_authors,
        "abstract": cr_abstract,
        "abstract_source": abs_source,
        "doi": cr_doi,
        "year": cr_year,
        "score": score,
        "year_diff": year_diff,
    }

    accepted = (
        score >= CROSSREF_TITLE_MIN_SCORE
        and year_diff is not None
        and year_diff <= CROSSREF_TITLE_YEAR_TOL
        and bool(cr_title)
    )
    if accepted:
        return payload, "accepted"
    return payload, "low_confidence"


# ---------------------------------------------------------------------------
# Per-study recovery orchestration
# ---------------------------------------------------------------------------
def _xlsx_filenames_for_study(
    df_xlsx: pd.DataFrame, covidence_id: str
) -> list[str]:
    """Return ordered xlsx filenames for one covidence_id."""
    mask = df_xlsx[XLSX_COL_ID].astype(str).str.strip() == covidence_id
    out: list[str] = []
    for fname in df_xlsx.loc[mask, XLSX_COL_FILENAME].tolist():
        if isinstance(fname, str) and fname.strip():
            out.append(fname.strip())
    return out


def _row_warnings_list(existing: object) -> list[str]:
    """Return existing warnings split into list, dedup-friendly."""
    if not isinstance(existing, str) or not existing.strip():
        return []
    return [p.strip() for p in existing.split(";") if p.strip()]


def _result_to_row_update(
    result,
    new_warnings: list[str],
    used_filename: Optional[str],
) -> dict:
    """Convert an ExtractionResult (from process_pdf) into the row update
    dict that gets written into the CSV."""
    return {
        "title": result.title,
        "authors": result.authors,
        "abstract": result.abstract,
        "doi": result.doi,
        "year": result.year,
        "source_method": result.source_method,
        "source_abstract": result.source_abstract,
        "pdf_used": used_filename,
        "warnings": _join_warnings(new_warnings),
    }


# ---------------------------------------------------------------------------
# Main recovery driver
# ---------------------------------------------------------------------------
def main() -> int:
    t_start = time.time()

    if not OUTPUT_CSV.exists():
        print(f"[T16.D.3.bis] ERROR: no existe {OUTPUT_CSV}")
        return 2
    if not XLSX_PATH.exists():
        print(f"[T16.D.3.bis] ERROR: no existe {XLSX_PATH}")
        return 2
    if not PDFS_DIR.exists():
        print(f"[T16.D.3.bis] ERROR: no existe {PDFS_DIR}")
        return 2

    print(f"[T16.D.3.bis] Leyendo CSV: {OUTPUT_CSV}")
    df = pd.read_csv(OUTPUT_CSV, dtype={"covidence_id": str})
    initial_shape = df.shape
    print(f"[T16.D.3.bis] shape inicial: {initial_shape}")

    df_xlsx = pd.read_excel(XLSX_PATH)

    # Identify the 78 failed studies
    failed_mask = df["source_method"] == "failed"
    failed_idx = df.index[failed_mask].tolist()
    n_failed = len(failed_idx)
    print(f"[T16.D.3.bis] failed iniciales: {n_failed}")

    # Sub-split for reporting
    n_no_pdf = int(
        df.loc[failed_mask, "warnings"]
        .astype(str)
        .str.contains("no PDF found", na=False)
        .sum()
    )
    n_all_failed = int(
        df.loc[failed_mask, "warnings"]
        .astype(str)
        .str.contains("all PDFs failed", na=False)
        .sum()
    )

    print("[T16.D.3.bis] Construyendo indice de repair mojibake...")
    repaired_index = build_repaired_skeleton_index(PDFS_DIR)
    print(f"[T16.D.3.bis]   filenames reparados en disco: {len(repaired_index)}")

    # Counters
    m05_pdfs_found = 0
    m05_extraction_ok = 0
    m05_still_failed = 0  # PDF found but every method still failed
    m6_attempted = 0
    m6_accepted = 0
    m6_low_conf = 0
    m6_unparseable = 0
    m6_no_result = 0
    m6_unreachable = 0

    # Snapshot the original failed reason text for each id to preserve in
    # warnings when we recover.
    progress_every = 10

    for i, idx in enumerate(failed_idx, start=1):
        cov_id = df.at[idx, "covidence_id"]
        xlsx_filenames = _xlsx_filenames_for_study(df_xlsx, cov_id)
        original_warning_list = _row_warnings_list(df.at[idx, "warnings"])
        had_no_pdf = any("no PDF found" in w for w in original_warning_list)

        # -----------------------------------------------------------------
        # Method 0.5 — Mojibake repair (only for "no PDF found" rows)
        # -----------------------------------------------------------------
        recovered_via_repair = False
        method05_match_filename: Optional[str] = None
        if had_no_pdf:
            for xfn in xlsx_filenames:
                disk_match = find_pdf_via_mojibake(xfn, repaired_index)
                if disk_match is not None:
                    method05_match_filename = disk_match
                    break

            if method05_match_filename is not None:
                m05_pdfs_found += 1
                pdf_path = PDFS_DIR / method05_match_filename
                result = process_pdf(pdf_path)

                # Method 5 (SS abstract enrichment) — mirror process_study()
                # behaviour: only invoke when source_method is set, DOI is
                # present, and abstract is missing.
                if (
                    result.source_method
                    and result.has_doi()
                    and not result.has_abstract()
                ):
                    ss_abs, ss_err = fetch_abstract_semantic_scholar(result.doi)
                    if ss_abs:
                        result.abstract = ss_abs
                        result.source_abstract = "semantic_scholar"
                    if ss_err:
                        result.warnings.append(ss_err)

                if result.source_method:
                    m05_extraction_ok += 1
                    new_warnings: list[str] = []
                    new_warnings.extend(original_warning_list)
                    new_warnings.append("filename repaired via mojibake fix")
                    for w in result.warnings:
                        if w.startswith(("crossref", "SS ")):
                            new_warnings.append(w)
                    update = _result_to_row_update(
                        result,
                        new_warnings,
                        used_filename=method05_match_filename,
                    )
                    for col, val in update.items():
                        df.at[idx, col] = val
                    recovered_via_repair = True
                else:
                    # PDF found but extraction still failed — fall through
                    # to Method 6 below. Keep a note so we can stash it
                    # together with the eventual Crossref outcome.
                    m05_still_failed += 1

        if recovered_via_repair:
            if i % progress_every == 0 or i == n_failed:
                elapsed = (time.time() - t_start) / 60.0
                print(
                    f"[T16.D.3.bis]   {i}/{n_failed} "
                    f"M0.5 found: {m05_pdfs_found} "
                    f"M0.5 ok: {m05_extraction_ok} "
                    f"M6 acc: {m6_accepted} "
                    f"M6 low: {m6_low_conf} "
                    f"elapsed: {elapsed:.1f} min"
                )
            continue

        # -----------------------------------------------------------------
        # Method 6 — Crossref by title (for all remaining failures)
        # -----------------------------------------------------------------
        # Parse filename for query: try each xlsx filename until one parses.
        parsed: Optional[tuple[Optional[str], int, str]] = None
        for xfn in xlsx_filenames:
            author, year_fn, title_fn = parse_filename(xfn)
            if year_fn is not None and title_fn:
                parsed = (author, year_fn, title_fn)
                break

        if parsed is None:
            m6_unparseable += 1
            new_warnings = list(original_warning_list)
            if m05_match_failed_attempted := (
                had_no_pdf and method05_match_filename is not None
            ):
                new_warnings.append(
                    "filename repaired via mojibake fix but extraction failed"
                )
            new_warnings.append("filename unparseable for crossref query")
            df.at[idx, "warnings"] = _join_warnings(new_warnings)
            # source_method stays "failed"
            if i % progress_every == 0 or i == n_failed:
                elapsed = (time.time() - t_start) / 60.0
                print(
                    f"[T16.D.3.bis]   {i}/{n_failed} "
                    f"M0.5 found: {m05_pdfs_found} "
                    f"M0.5 ok: {m05_extraction_ok} "
                    f"M6 acc: {m6_accepted} "
                    f"M6 low: {m6_low_conf} "
                    f"elapsed: {elapsed:.1f} min"
                )
            continue

        author, year_fn, title_fn = parsed
        m6_attempted += 1
        payload, decision = crossref_by_title(title_fn, author, year_fn)

        new_warnings = list(original_warning_list)
        if had_no_pdf and method05_match_filename is not None:
            new_warnings.append(
                "filename repaired via mojibake fix but extraction failed"
            )

        if decision == "accepted":
            m6_accepted += 1
            assert payload is not None
            # If Crossref didn't supply an abstract but the DOI is known,
            # try Method 5 (Semantic Scholar) to enrich.
            abstract_val = payload["abstract"]
            abstract_src = payload["abstract_source"]
            if not abstract_val and payload["doi"]:
                ss_abs, ss_err = fetch_abstract_semantic_scholar(payload["doi"])
                if ss_abs:
                    abstract_val = ss_abs
                    abstract_src = "semantic_scholar"
                if ss_err:
                    new_warnings.append(ss_err)

            df.at[idx, "title"] = payload["title"]
            df.at[idx, "authors"] = payload["authors"]
            df.at[idx, "abstract"] = abstract_val
            df.at[idx, "doi"] = payload["doi"]
            df.at[idx, "year"] = payload["year"]
            df.at[idx, "source_method"] = "crossref_by_title"
            df.at[idx, "source_abstract"] = abstract_src
            # pdf_used stays as the (failed) name we have, or None
            new_warnings.append(
                f"recovered via crossref by title "
                f"(score={int(payload['score'])}, "
                f"year_diff={payload['year_diff']})"
            )
            df.at[idx, "warnings"] = _join_warnings(new_warnings)

        elif decision == "low_confidence":
            m6_low_conf += 1
            assert payload is not None
            # Do NOT populate main fields. Stash candidate as a structured
            # warning instead.
            df.at[idx, "source_method"] = "crossref_by_title_low_confidence"
            candidate_repr = (
                f"low_confidence_candidate: title={payload['title']!r} "
                f"doi={payload['doi']} year={payload['year']} "
                f"score={int(payload['score'])}"
            )
            new_warnings.append(
                f"crossref by title low confidence: "
                f"score={int(payload['score'])}, "
                f"year_diff={payload['year_diff']}"
            )
            new_warnings.append(candidate_repr)
            df.at[idx, "warnings"] = _join_warnings(new_warnings)

        elif decision == "no_result":
            m6_no_result += 1
            new_warnings.append("crossref by title no result")
            df.at[idx, "source_method"] = "crossref_by_title_low_confidence"
            df.at[idx, "warnings"] = _join_warnings(new_warnings)

        else:
            # unreachable: <details>
            m6_unreachable += 1
            new_warnings.append(f"crossref by title unreachable: {decision[12:]}")
            # source_method stays "failed"
            df.at[idx, "warnings"] = _join_warnings(new_warnings)

        if i % progress_every == 0 or i == n_failed:
            elapsed = (time.time() - t_start) / 60.0
            print(
                f"[T16.D.3.bis]   {i}/{n_failed} "
                f"M0.5 found: {m05_pdfs_found} "
                f"M0.5 ok: {m05_extraction_ok} "
                f"M6 acc: {m6_accepted} "
                f"M6 low: {m6_low_conf} "
                f"elapsed: {elapsed:.1f} min"
            )

    # ---------------------------------------------------------------------
    # Persist
    # ---------------------------------------------------------------------
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLUMNS]
    # year may end up float NaN; coerce to nullable Int64 as the base script
    # does.
    try:
        df["year"] = df["year"].astype("Int64")
    except (TypeError, ValueError):
        # Mixed types after our updates; round-trip via to_numeric
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[T16.D.3.bis] CSV reescrito: {OUTPUT_CSV}  shape={df.shape}")

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------
    elapsed_min = (time.time() - t_start) / 60.0
    final_failed = int((df["source_method"] == "failed").sum())
    final_low_conf = int(
        (df["source_method"] == "crossref_by_title_low_confidence").sum()
    )
    final_by_title = int((df["source_method"] == "crossref_by_title").sum())
    total = len(df)
    success_methods = (
        "pdf_metadata",
        "doi_crossref",
        "pdfplumber_fallback",
        "crossref_by_title",
    )
    final_success = int(df["source_method"].isin(success_methods).sum())
    initial_success = 1474  # 1552 - 78 from previous pass

    recovered = final_by_title  # those flipped from failed to crossref_by_title
    # "still failed" = originally failed AND still failed (or low conf,
    # which is not a recovery)
    still_failed_among_orig = final_failed
    low_conf_among_orig = final_low_conf

    print()
    print("=== T16.D.3.bis summary ===")
    print(f"failed originals: {n_failed}")
    print(f"  no PDF found: {n_no_pdf}")
    print(f"  all PDFs failed: {n_all_failed}")
    print()
    print("Method 0.5 (mojibake repair):")
    print(f"  PDFs encontrados: {m05_pdfs_found}")
    print(f"  Extraccion exitosa tras encontrar PDF: {m05_extraction_ok}")
    print(
        f"  Aun failed (PDF localizado pero extraccion fallo): {m05_still_failed}"
    )
    print()
    print("Method 6 (crossref by title):")
    print(f"  intentados: {m6_attempted}")
    print(f"  aceptados (score>=80 AND year+/-1): {m6_accepted}")
    print(f"  low_confidence: {m6_low_conf + m6_no_result}")
    print(f"  no parseables: {m6_unparseable}")
    print(f"  no result: {m6_no_result}")
    print(f"  unreachable: {m6_unreachable}")
    print()
    print(f"Estado final (de los {n_failed} originales):")
    print(f"  recovered (no failed): {m05_extraction_ok + m6_accepted}")
    print(f"  still failed: {still_failed_among_orig}")
    print(f"  low_confidence: {low_conf_among_orig}")
    print()
    print(f"Tasa de exito GLOBAL ({total} estudios):")
    pct_before = (initial_success / total * 100) if total else 0.0
    pct_after = (final_success / total * 100) if total else 0.0
    print(f"  before: {initial_success} ({pct_before:.1f}%)")
    print(f"  after:  {final_success} ({pct_after:.1f}%)")
    print()
    print(f"elapsed total: {elapsed_min:.1f} min")

    return 0


if __name__ == "__main__":
    sys.exit(main())
