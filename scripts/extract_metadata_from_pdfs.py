#!/usr/bin/env python3
# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""
extract_metadata_from_pdfs.py — T16.D.3.A (+ T16.D.3.A.bis Method 5)

Extract bibliographic metadata for the 1552 deporte gold-standard studies
whose unified CSV rows have title/authors/abstract/doi/year all NaN (xlsx
source). The xlsx (`Resumen_Total_PDFs.xlsx`) only records covidence IDs +
filenames + screening decisions; metadata must be recovered from the PDFs
themselves stored in `data/gold_standard/deporte/pdfs_todos/`.

Strategy per study (covidence_id):
    Iterate over its PDFs (xlsx order). For each PDF try
        Method 1: pypdf metadata + XMP
        Method 2: DOI regex on first-page text (pdfplumber)
        Method 3: Crossref lookup (only if Method 2 found a DOI)
        Method 5: Semantic Scholar by DOI (abstract enrichment only; only
                  invoked when the study has a DOI but Crossref didn't
                  provide an abstract). Does NOT change source_method.
        Method 4: pdfplumber heuristic fallback (largest-font title,
                  adjacent author lines, Abstract block)
    Stop on the first PDF that yields at least a title. If all PDFs fail,
    record `source_method="failed"`. `#32107` has no PDF (Falló HTTP) and is
    recorded as failed with a specific warning.

Method 5 (T16.D.3.A.bis) — Semantic Scholar abstract enrichment:
    GET https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract
    Used to enrich abstracts when pdf_metadata/crossref/pdfplumber_fallback
    failed to supply one. Only invoked when DOI is present. Populates a new
    `source_abstract` column without altering the principal `source_method`.

Architecture decision (T16.D.3.A.bis): Option A — Method 5 is integrated
natively into the main flow. A `--abstracts-only` CLI flag re-reads an
existing CSV (skipping PDF re-extraction) and only runs Method 5 on rows
that are eligible (success row, DOI present, abstract missing). This keeps
schema authoritative in a single script.

Output:
    data/gold_standard/deporte/unified/extracted_metadata.csv

Columns (10):
    covidence_id, title, authors, abstract, doi, year,
    source_method, source_abstract, pdf_used, warnings

`source_abstract` values:
    "pdf_metadata"        — abstract from Method 1
    "crossref"            — abstract from Method 3
    "pdfplumber_fallback" — abstract from Method 4
    "semantic_scholar"    — abstract from Method 5
    None                  — no abstract available

Identifiers and code comments are in English. Logs and stdout summary are in
Spanish, per project rule #12.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import pypdf
import pdfplumber
import requests
import yaml
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPORTE_DIR = PROJECT_ROOT / "data" / "gold_standard" / "deporte"
XLSX_PATH = DEPORTE_DIR / "Resumen_Total_PDFs.xlsx"
PDFS_DIR = DEPORTE_DIR / "pdfs_todos"
OUTPUT_DIR = DEPORTE_DIR / "unified"
OUTPUT_CSV = OUTPUT_DIR / "extracted_metadata.csv"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

XLSX_COL_ID = "ID del Estudio (Covidence)"
XLSX_COL_FILENAME = "Nombre del Archivo Descargado"
XLSX_COL_STATUS = "Estado"

STATUS_FAILED_HTTP = "Falló HTTP"

MISMATCH_ABORT_FRACTION = 0.05  # abort if >5% filenames missing

OUTPUT_COLUMNS = [
    "covidence_id",
    "title",
    "authors",
    "abstract",
    "doi",
    "year",
    "source_method",
    "source_abstract",
    "pdf_used",
    "warnings",
]

# DOI regex (case-insensitive). Stops at whitespace/closing punctuation.
DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

# pdfplumber title-fallback constants
MAX_TITLE_CHARS = 250
ABSTRACT_HEADERS = ("ABSTRACT", "SUMMARY", "Abstract", "Summary")
ABSTRACT_MAX_CHARS = 2000
YEAR_REGEX = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")

CROSSREF_BASE = "https://api.crossref.org/works/"

# Semantic Scholar (Method 5)
SS_BASE = "https://api.semanticscholar.org/graph/v1/paper/DOI:"
SS_FIELDS = "abstract"
SS_SLEEP_SECONDS = 3.0  # ~20 req/min, well below 100 req/5min unauthenticated
SS_TIMEOUT_SECONDS = 15
SS_PROGRESS_EVERY = 50


# ---------------------------------------------------------------------------
# Config-driven values
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    """Load config.yaml (best-effort)."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


_CONFIG = _load_config()
_RETRIEVAL_CFG = _CONFIG.get("retrieval", {}) or {}
_CROSSREF_SLEEP = float(_RETRIEVAL_CFG.get("crossref_seconds", 0.5))
_POLITE_EMAIL = (
    _RETRIEVAL_CFG.get("unpaywall_email")
    or _CONFIG.get("entrez_email")
    or "meta-analysis@example.com"
)
_USER_AGENT = f"SAM/1.0 (mailto:{_POLITE_EMAIL})"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """Per-PDF extraction outcome."""
    title: Optional[str] = None
    authors: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    source_method: Optional[str] = None  # set when accepted
    source_abstract: Optional[str] = None  # provenance of `abstract` value
    warnings: list = field(default_factory=list)

    def has_title(self) -> bool:
        return bool(self.title and self.title.strip())

    def has_doi(self) -> bool:
        return bool(self.doi and self.doi.strip())

    def has_abstract(self) -> bool:
        return bool(self.abstract and self.abstract.strip())


# ---------------------------------------------------------------------------
# Filename normalization helpers
# ---------------------------------------------------------------------------
def _ascii_skeleton(s: str) -> str:
    """Return an ASCII-only, lowercase, whitespace-collapsed skeleton.

    Used for tolerant filename matching. Handles:
      - NFC vs NFD diacritic differences (NFKD + drop combining marks).
      - Mojibake on the FS (e.g. UTF-8 filename re-decoded as CP850/CP437):
        any non-ASCII char in the cleaned skeleton is dropped entirely so
        that "INTERVENCIÓN_156" and "INTERVENCI├УN_156" both collapse to
        "intervencin_156".
      - Mixed casing and runs of whitespace.
    """
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    no_diacritics = "".join(
        c for c in nfkd if not unicodedata.combining(c)
    )
    # Drop any remaining non-ASCII (mojibake bytes get squashed)
    ascii_only = no_diacritics.encode("ascii", "ignore").decode("ascii")
    lower = ascii_only.lower()
    collapsed = re.sub(r"\s+", " ", lower).strip()
    return collapsed


def build_filename_index(pdfs_dir: Path) -> tuple[set[str], dict[str, str]]:
    """Return (byte-exact set, skeleton -> filename map) for files in dir."""
    files = [p.name for p in pdfs_dir.iterdir() if p.is_file()]
    exact = set(files)
    skeleton: dict[str, str] = {}
    for fname in files:
        key = _ascii_skeleton(fname)
        # If two files share a skeleton, keep the first (deterministic by order
        # from iterdir which is FS-defined). Skeleton clashes are extremely rare.
        skeleton.setdefault(key, fname)
    return exact, skeleton


def resolve_filename(
    xlsx_filename: str,
    exact: set[str],
    skeleton: dict[str, str],
) -> tuple[Optional[str], bool]:
    """Resolve an xlsx filename to a real file on disk.

    Returns (resolved_filename | None, used_skeleton_match).
    """
    if xlsx_filename in exact:
        return xlsx_filename, False
    key = _ascii_skeleton(xlsx_filename)
    if key in skeleton:
        return skeleton[key], True
    return None, False


# ---------------------------------------------------------------------------
# Method 1 — pypdf metadata + XMP
# ---------------------------------------------------------------------------
def _clean_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


# Placeholder/junk values commonly written by PDF generators that should
# not be accepted as real metadata.
_PLACEHOLDER_TITLES = {
    "untitled",
    "microsoft word",
    "no title",
    "title",
    "document",
    "unknown",
    "default",
    "manuscript",
    "article",
    "paper",
    "draft",
}
_PLACEHOLDER_AUTHORS = {
    "unknown",
    "author",
    "anonymous",
    "user",
    "default",
    "owner",
    "admin",
}


def _is_placeholder_title(text: Optional[str]) -> bool:
    if not text:
        return True
    norm = text.strip().lower()
    if not norm:
        return True
    if norm in _PLACEHOLDER_TITLES:
        return True
    # 'Microsoft Word - document.doc' style filenames
    if norm.startswith("microsoft word -") or norm.startswith("microsoft word"):
        return True
    # Reject very short titles (1-2 words, < 10 chars)
    if len(text.strip()) < 10:
        return True
    return False


def _sanitize_author(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    norm = text.strip().lower()
    if norm in _PLACEHOLDER_AUTHORS:
        return None
    return text.strip() or None


def _parse_pdf_creation_year(value) -> Optional[int]:
    """Parse `D:YYYYMMDD...` or XMP `YYYY-MM-DD...` into a year integer."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.startswith("D:"):
        s = s[2:]
    # First 4 digits considered year
    m = re.match(r"(\d{4})", s)
    if not m:
        return None
    year = int(m.group(1))
    if 1950 <= year <= 2030:
        return year
    return None


def extract_pdf_metadata(pdf_path: Path) -> ExtractionResult:
    """Try Method 1: pypdf metadata + XMP."""
    result = ExtractionResult()
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as exc:
        result.warnings.append(f"pypdf open failed: {type(exc).__name__}")
        return result

    meta = None
    try:
        meta = reader.metadata
    except Exception:
        meta = None

    if meta:
        result.title = _clean_text(meta.get("/Title"))
        result.authors = _clean_text(meta.get("/Author"))
        # DOI may live in Subject/Keywords/custom
        haystack_parts = [
            meta.get("/Subject"),
            meta.get("/Keywords"),
            meta.get("/doi"),
            meta.get("/DOI"),
        ]
        for part in haystack_parts:
            if part is None:
                continue
            match = DOI_REGEX.search(str(part))
            if match:
                result.doi = match.group(0)
                break
        result.year = _parse_pdf_creation_year(meta.get("/CreationDate"))

    # XMP metadata (best-effort)
    try:
        xmp = reader.xmp_metadata
    except Exception:
        xmp = None
    if xmp is not None:
        try:
            if not result.title:
                dc_title = getattr(xmp, "dc_title", None)
                if dc_title:
                    # dc_title may be dict-like or list
                    if isinstance(dc_title, dict):
                        # take first language value
                        for v in dc_title.values():
                            result.title = _clean_text(v)
                            if result.title:
                                break
                    elif isinstance(dc_title, list) and dc_title:
                        result.title = _clean_text(dc_title[0])
                    else:
                        result.title = _clean_text(dc_title)
            if not result.authors:
                dc_creator = getattr(xmp, "dc_creator", None)
                if dc_creator:
                    if isinstance(dc_creator, list):
                        result.authors = "; ".join(
                            _clean_text(c) or "" for c in dc_creator if c
                        ).strip("; ")
                    else:
                        result.authors = _clean_text(dc_creator)
            if not result.doi:
                for attr in ("dc_identifier", "doi"):
                    val = getattr(xmp, attr, None)
                    if val:
                        match = DOI_REGEX.search(str(val))
                        if match:
                            result.doi = match.group(0)
                            break
            if not result.year:
                for attr in ("xmp_createDate", "xmp_modifyDate"):
                    val = getattr(xmp, attr, None)
                    if val:
                        result.year = _parse_pdf_creation_year(str(val))
                        if result.year:
                            break
        except Exception:
            pass

    # Sanitize trailing punctuation on DOI (regex may grab trailing periods)
    if result.doi:
        result.doi = result.doi.rstrip(".,;)")

    # Reject placeholder titles/authors written by PDF generators
    if _is_placeholder_title(result.title):
        result.title = None
    result.authors = _sanitize_author(result.authors)

    if result.has_title() or result.has_doi():
        result.source_method = "pdf_metadata"
    return result


# ---------------------------------------------------------------------------
# Method 2 — DOI regex on first page (pdfplumber)
# ---------------------------------------------------------------------------
def extract_doi_first_page(pdf_path: Path) -> Optional[str]:
    """Open with pdfplumber and search first-page text for a DOI."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if not pdf.pages:
                return None
            text = pdf.pages[0].extract_text() or ""
    except Exception:
        return None
    match = DOI_REGEX.search(text)
    if match:
        return match.group(0).rstrip(".,;)")
    return None


# ---------------------------------------------------------------------------
# Method 3 — Crossref
# ---------------------------------------------------------------------------
class CrossrefNotFound(Exception):
    """Crossref returned 404 for this DOI."""


class CrossrefRetryable(Exception):
    """Retryable Crossref failure."""


def _is_retryable_crossref(exc: BaseException) -> bool:
    return isinstance(exc, CrossrefRetryable)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_retryable_crossref),
)
def _crossref_get(doi: str) -> dict:
    """GET crossref. Raises CrossrefNotFound on 404, CrossrefRetryable on
    429/5xx/timeout/connection errors."""
    headers = {"User-Agent": _USER_AGENT}
    try:
        resp = requests.get(
            CROSSREF_BASE + doi,
            headers=headers,
            timeout=15,
        )
    except requests.RequestException as exc:
        raise CrossrefRetryable(str(exc)) from exc

    if resp.status_code == 404:
        raise CrossrefNotFound(f"crossref 404 for DOI={doi}")
    if resp.status_code in (429,) or 500 <= resp.status_code < 600:
        raise CrossrefRetryable(f"crossref HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise CrossrefRetryable(f"crossref unexpected HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError as exc:
        raise CrossrefRetryable(f"crossref JSON parse: {exc}") from exc


def _format_author(author: dict) -> Optional[str]:
    """Format `{family, given}` -> 'Family AB' style (initials, no dot)."""
    family = (author.get("family") or "").strip()
    given = (author.get("given") or "").strip()
    if not family and not given:
        # 'name' field for organizations
        name = (author.get("name") or "").strip()
        return name or None
    # Build initials from given name tokens
    initials = []
    for token in re.split(r"[\s\.\-]+", given):
        token = token.strip()
        if not token:
            continue
        # Take first letter, uppercase
        initials.append(token[0].upper())
    initials_str = "".join(initials)
    if family and initials_str:
        return f"{family} {initials_str}"
    return family or initials_str or None


def _strip_jats(text: str) -> str:
    """Strip JATS/HTML tags from an abstract."""
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _year_from_message(message: dict) -> Optional[int]:
    """Extract year from a crossref `message`."""
    for key in ("published", "published-online", "published-print", "issued"):
        block = message.get(key)
        if not block:
            continue
        date_parts = block.get("date-parts")
        if date_parts and date_parts[0]:
            try:
                year = int(date_parts[0][0])
            except (TypeError, ValueError):
                continue
            if 1900 <= year <= 2100:
                return year
    return None


def fetch_crossref(doi: str) -> tuple[Optional[ExtractionResult], Optional[str]]:
    """Fetch and parse Crossref. Returns (result|None, warning|None)."""
    try:
        data = _crossref_get(doi)
    except CrossrefNotFound as exc:
        return None, str(exc)
    except CrossrefRetryable as exc:
        return None, f"crossref unreachable: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"crossref unreachable: {type(exc).__name__}: {exc}"
    finally:
        time.sleep(_CROSSREF_SLEEP)

    message = (data or {}).get("message", {})
    if not message:
        return None, "crossref empty message"

    result = ExtractionResult()
    titles = message.get("title") or []
    if isinstance(titles, list) and titles:
        result.title = _clean_text(titles[0])
    elif isinstance(titles, str):
        result.title = _clean_text(titles)

    authors_data = message.get("author") or []
    formatted = [
        _format_author(a) for a in authors_data if isinstance(a, dict)
    ]
    formatted = [f for f in formatted if f]
    if formatted:
        result.authors = "; ".join(formatted)

    abstract = message.get("abstract")
    if abstract:
        cleaned_abstract = _strip_jats(str(abstract)) or None
        if cleaned_abstract:
            result.abstract = cleaned_abstract
            result.source_abstract = "crossref"

    result.doi = doi
    result.year = _year_from_message(message)
    result.source_method = "doi_crossref"

    if not result.has_title():
        # Crossref returned but no title — treat as failure
        return None, "crossref returned no title"
    return result, None


# ---------------------------------------------------------------------------
# Method 5 — Semantic Scholar by DOI (abstract enrichment only)
# ---------------------------------------------------------------------------
class SemanticScholarNotFound(Exception):
    """Semantic Scholar returned 404 for this DOI."""


class SemanticScholarRetryable(Exception):
    """Retryable Semantic Scholar failure (429/5xx/timeout/conn error)."""


def _is_retryable_ss(exc: BaseException) -> bool:
    return isinstance(exc, SemanticScholarRetryable)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_retryable_ss),
)
def _ss_get(doi: str) -> dict:
    """GET Semantic Scholar by DOI. Raises SemanticScholarNotFound on 404,
    SemanticScholarRetryable on 429/5xx/timeout/connection errors."""
    headers = {"User-Agent": _USER_AGENT}
    url = f"{SS_BASE}{doi}?fields={SS_FIELDS}"
    try:
        resp = requests.get(url, headers=headers, timeout=SS_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise SemanticScholarRetryable(str(exc)) from exc

    if resp.status_code == 404:
        raise SemanticScholarNotFound(f"SS 404 for DOI={doi}")
    if resp.status_code == 429 or 500 <= resp.status_code < 600:
        raise SemanticScholarRetryable(f"SS HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise SemanticScholarRetryable(f"SS unexpected HTTP {resp.status_code}")
    try:
        return resp.json() or {}
    except ValueError as exc:
        raise SemanticScholarRetryable(f"SS JSON parse: {exc}") from exc


def fetch_abstract_semantic_scholar(
    doi: str,
) -> tuple[Optional[str], Optional[str]]:
    """Fetch an abstract from Semantic Scholar by DOI.

    Returns (abstract|None, error_or_404_str|None). The second element is
    None on success (even if abstract is None because SS lacks it), a "SS
    no record for DOI=..." string on 404, or a short error description on
    other failures after retries. Sleeps SS_SLEEP_SECONDS at the end of
    every call to enforce the polite rate limit.
    """
    try:
        data = _ss_get(doi)
    except SemanticScholarNotFound:
        return None, f"SS no record for DOI={doi}"
    except SemanticScholarRetryable as exc:
        return None, f"SS abstract retrieval failed: {exc}"
    except RetryError as exc:
        return None, f"SS abstract retrieval failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"SS abstract retrieval failed: {type(exc).__name__}: {exc}"
    finally:
        time.sleep(SS_SLEEP_SECONDS)

    abstract = data.get("abstract")
    if not abstract:
        # 200 with no abstract: not a failure, just not indexed text
        return None, None
    cleaned = str(abstract).strip()
    if not cleaned:
        return None, None
    return cleaned, None


# ---------------------------------------------------------------------------
# Method 4 — pdfplumber heuristic fallback
# ---------------------------------------------------------------------------
_AUTHOR_LINE_REGEX = re.compile(r"[A-Z][A-Za-z'\-]+")


def _looks_like_author_line(line: str) -> bool:
    """Heuristic to detect an author line: capital-initial tokens, possible
    commas/initials, digits should not dominate."""
    line = line.strip()
    if not line or len(line) < 3:
        return False
    n_digits = sum(c.isdigit() for c in line)
    if n_digits / max(len(line), 1) > 0.5:
        return False
    has_comma = "," in line
    has_initial = bool(re.search(r"\b[A-Z]\.", line))
    cap_words = len(_AUTHOR_LINE_REGEX.findall(line))
    return cap_words >= 2 and (has_comma or has_initial or cap_words >= 3)


def _extract_title_largest_font(page) -> Optional[str]:
    """Group characters by font size; pick contiguous run of largest size
    within the first ~30% lines, return joined text trimmed to MAX_TITLE_CHARS.
    """
    chars = page.chars or []
    if not chars:
        return None
    # Group characters by approximate y-position (line) and font size.
    # Compute the maximum size present in the upper region.
    page_height = page.height or 1
    upper_limit = page_height * 0.5  # be tolerant
    candidates = [c for c in chars if c.get("top", 0) <= upper_limit]
    if not candidates:
        candidates = chars
    sizes = [round(c.get("size", 0), 1) for c in candidates if c.get("size")]
    if not sizes:
        return None
    max_size = max(sizes)
    # Cluster characters into lines by y-top (within +/-2 pts)
    lines = []
    current_top = None
    current_chars = []
    for c in sorted(candidates, key=lambda x: (x.get("top", 0), x.get("x0", 0))):
        top = c.get("top", 0)
        if current_top is None or abs(top - current_top) <= 2.5:
            current_chars.append(c)
            if current_top is None:
                current_top = top
        else:
            if current_chars:
                lines.append(current_chars)
            current_chars = [c]
            current_top = top
    if current_chars:
        lines.append(current_chars)
    # Find first line whose max size equals the page max size; collect
    # consecutive lines that share this top size.
    title_lines: list[str] = []
    capture = False
    for line_chars in lines:
        line_sizes = [round(c.get("size", 0), 1) for c in line_chars]
        if not line_sizes:
            if capture:
                break
            continue
        line_max = max(line_sizes)
        text = "".join(c.get("text", "") for c in line_chars).strip()
        if not text:
            if capture:
                break
            continue
        if abs(line_max - max_size) < 0.1:
            title_lines.append(text)
            capture = True
        elif capture:
            break
    if not title_lines:
        return None
    title = " ".join(title_lines)
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rstrip() + "..."
    if len(title) < 10:
        return None
    return title


def _extract_authors_after_title(page_text: str, title: str) -> Optional[str]:
    """Pull up to 3 author-like lines following the title in `page_text`."""
    if not title or not page_text:
        return None
    lines = [ln.strip() for ln in page_text.splitlines()]
    # Locate title position by token overlap (title may be on multiple lines).
    title_tokens = [t for t in re.split(r"\s+", title) if t]
    if not title_tokens:
        return None
    first_token = title_tokens[0]
    start_idx = None
    for idx, ln in enumerate(lines):
        if ln.startswith(first_token[:20]):
            start_idx = idx
            break
    if start_idx is None:
        # Fall back to first non-empty line + 1
        for idx, ln in enumerate(lines):
            if ln:
                start_idx = idx
                break
    if start_idx is None:
        return None
    # Skip lines that are still part of title (heuristic: contiguous title tokens)
    cursor = start_idx
    while cursor < len(lines):
        if not lines[cursor]:
            cursor += 1
            continue
        # Stop title segment when line stops matching title text.
        # Simple check: if this line is in the title string itself, skip it.
        if lines[cursor] and lines[cursor] in title:
            cursor += 1
            continue
        break
    # Collect up to 3 candidate author lines.
    candidates = []
    examined = 0
    while cursor < len(lines) and examined < 6 and len(candidates) < 3:
        ln = lines[cursor]
        cursor += 1
        examined += 1
        if not ln:
            if candidates:
                break
            continue
        if _looks_like_author_line(ln):
            candidates.append(ln)
        elif candidates:
            break
    if not candidates:
        return None
    return " ".join(candidates).strip() or None


def _extract_abstract_block(pages_text: list[str]) -> Optional[str]:
    """Locate Abstract/Summary header within first 2 pages and capture body."""
    combined = "\n".join(pages_text)
    # Match header at line start
    header_pat = re.compile(
        r"(?m)^\s*(ABSTRACT|SUMMARY|Abstract|Summary)\b[:\.\s]*"
    )
    m = header_pat.search(combined)
    if not m:
        return None
    start = m.end()
    body = combined[start:]
    # Next section header: line in ALL CAPS or known section names
    next_header = re.search(
        r"(?m)^\s*("
        r"INTRODUCTION|BACKGROUND|METHODS|MATERIALS AND METHODS|RESULTS|"
        r"DISCUSSION|CONCLUSION|KEYWORDS|KEY WORDS|REFERENCES|"
        r"Introduction|Background|Methods|Results|Discussion|Conclusion|"
        r"Keywords|Key Words|References"
        r")\b",
        body,
    )
    if next_header:
        body = body[: next_header.start()]
    body = body[:ABSTRACT_MAX_CHARS]
    body = re.sub(r"\s+", " ", body).strip()
    if not body or len(body) < 20:
        return None
    return body


def _extract_year_near_top(page_text: str) -> Optional[int]:
    """Find the smallest plausible 4-digit year in the first-page text."""
    if not page_text:
        return None
    snippet = page_text[:1500]
    matches = YEAR_REGEX.findall(snippet)
    if not matches:
        return None
    years = sorted({int(m) for m in matches})
    for y in years:
        if 1950 <= y <= 2030:
            return y
    return None


def extract_pdfplumber_fallback(pdf_path: Path) -> ExtractionResult:
    """Method 4: heuristic extraction using pdfplumber."""
    result = ExtractionResult()
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if not pdf.pages:
                return result
            first_page = pdf.pages[0]
            first_text = first_page.extract_text() or ""
            title = _extract_title_largest_font(first_page)
            result.title = title
            result.authors = _extract_authors_after_title(first_text, title or "")
            pages_text = [first_text]
            if len(pdf.pages) > 1:
                pages_text.append(pdf.pages[1].extract_text() or "")
            abstract_text = _extract_abstract_block(pages_text)
            if abstract_text:
                result.abstract = abstract_text
                result.source_abstract = "pdfplumber_fallback"
            result.year = _extract_year_near_top(first_text)
    except Exception as exc:
        result.warnings.append(f"pdfplumber failed: {type(exc).__name__}")
        return result

    if result.has_title():
        result.source_method = "pdfplumber_fallback"
    return result


# ---------------------------------------------------------------------------
# Per-study orchestration
# ---------------------------------------------------------------------------
def process_pdf(pdf_path: Path) -> ExtractionResult:
    """Run Method 1 -> 2 -> 3 -> 4 for a single PDF."""
    final = ExtractionResult()
    final_warnings: list[str] = []

    # Method 1
    m1 = extract_pdf_metadata(pdf_path)
    final_warnings.extend(m1.warnings)
    if m1.source_method:
        # Method 1 accepted (title or DOI present)
        # If we have only DOI but no title, prefer to refine via Crossref.
        if m1.has_title():
            # Method 1 never populates abstract today, but keep provenance
            # consistent if a future change ever does.
            src_abs = "pdf_metadata" if m1.has_abstract() else None
            return ExtractionResult(
                title=m1.title,
                authors=m1.authors,
                abstract=m1.abstract,
                doi=m1.doi,
                year=m1.year,
                source_method="pdf_metadata",
                source_abstract=src_abs,
                warnings=final_warnings,
            )
        # only DOI -> try Crossref to fill the rest
        if m1.has_doi():
            cr, warn = fetch_crossref(m1.doi)
            if warn:
                final_warnings.append(warn)
            if cr is not None:
                cr.warnings = final_warnings
                return cr
            # Crossref failed; fall through to other methods with DOI preserved
            final.doi = m1.doi

    # Method 2 — first-page DOI
    doi = extract_doi_first_page(pdf_path)
    if doi:
        if not final.doi:
            final.doi = doi
        # Method 3 — Crossref (skip if we just tried this DOI in Method 1)
        already_tried = (m1.has_doi() and m1.doi == doi)
        if not already_tried:
            cr, warn = fetch_crossref(doi)
            if warn:
                final_warnings.append(warn)
            if cr is not None:
                cr.warnings = final_warnings
                return cr

    # Method 4 — heuristic
    m4 = extract_pdfplumber_fallback(pdf_path)
    final_warnings.extend(m4.warnings)
    if m4.source_method:
        return ExtractionResult(
            title=m4.title,
            authors=m4.authors,
            abstract=m4.abstract,
            doi=final.doi or m4.doi,
            year=m4.year,
            source_method="pdfplumber_fallback",
            source_abstract=m4.source_abstract,
            warnings=final_warnings,
        )

    # All methods failed for this PDF
    final.warnings = final_warnings
    return final  # source_method=None means failed


def process_study(
    covidence_id: str,
    pdf_entries: list[tuple[str, Optional[Path], bool, str]],
) -> dict:
    """Run extraction for one study.

    pdf_entries: ordered list of (xlsx_filename, resolved_path|None,
    used_skeleton_match, status_from_xlsx).
    Returns the row dict for the CSV.
    """
    # Special case: study has only one xlsx row with status 'Falló HTTP'
    # and no resolved PDFs.
    resolved_entries = [e for e in pdf_entries if e[1] is not None]
    if not resolved_entries:
        # Study has no PDF on disk
        warnings = []
        if any(e[3] == STATUS_FAILED_HTTP for e in pdf_entries):
            warnings.append("no PDF found (Falló HTTP)")
        else:
            warnings.append("no PDF found")
        return _empty_row(covidence_id, warnings)

    skeleton_warning_added = False
    failed_filenames: list[str] = []
    crossref_warnings: list[str] = []

    for idx, (xlsx_fname, path, used_skeleton, _status) in enumerate(
        resolved_entries, start=1
    ):
        result = process_pdf(path)
        # Collect warnings from this attempt that we want to surface
        # (only crossref-related ones; ignore generic open errors unless final fail).
        for w in result.warnings:
            if w.startswith("crossref"):
                crossref_warnings.append(w)
        if result.source_method:
            warnings: list[str] = []
            if used_skeleton:
                warnings.append("filename matched via ASCII skeleton")
                skeleton_warning_added = True
            if len(resolved_entries) > 1 and idx > 1:
                warnings.append(
                    f"used PDF {idx} of {len(resolved_entries)}"
                )
            # Surface unique crossref warnings collected so far
            for w in crossref_warnings:
                if w not in warnings:
                    warnings.append(w)
            # Method 5 — Semantic Scholar abstract enrichment.
            # Only invoked when we have a DOI but no abstract yet. Does not
            # alter source_method; only fills `abstract` + `source_abstract`.
            if (
                not result.has_abstract()
                and result.has_doi()
            ):
                ss_abs, ss_err = fetch_abstract_semantic_scholar(result.doi)
                if ss_abs:
                    result.abstract = ss_abs
                    result.source_abstract = "semantic_scholar"
                if ss_err:
                    warnings.append(ss_err)
            return {
                "covidence_id": covidence_id,
                "title": result.title,
                "authors": result.authors,
                "abstract": result.abstract,
                "doi": result.doi,
                "year": result.year,
                "source_method": result.source_method,
                "source_abstract": result.source_abstract,
                "pdf_used": path.name,
                "warnings": _join_warnings(warnings),
            }
        # PDF failed; remember filename
        failed_filenames.append(path.name)

    # All PDFs failed
    warnings: list[str] = []
    if any(used_skel for _, _, used_skel, _ in resolved_entries):
        warnings.append("filename matched via ASCII skeleton")
    warnings.append("all PDFs failed: " + ", ".join(failed_filenames))
    for w in crossref_warnings:
        if w not in warnings:
            warnings.append(w)
    return _empty_row(covidence_id, warnings)


def _empty_row(covidence_id: str, warnings: list[str]) -> dict:
    return {
        "covidence_id": covidence_id,
        "title": None,
        "authors": None,
        "abstract": None,
        "doi": None,
        "year": None,
        "source_method": "failed",
        "source_abstract": None,
        "pdf_used": None,
        "warnings": _join_warnings(warnings),
    }


def _join_warnings(warnings: list[str]) -> Optional[str]:
    cleaned = [w for w in warnings if w]
    if not cleaned:
        return None
    # Deduplicate while preserving order
    seen = set()
    out = []
    for w in cleaned:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
    return "; ".join(out)


# ---------------------------------------------------------------------------
# Manifest check (Fase 0)
# ---------------------------------------------------------------------------
def build_manifest() -> tuple[
    pd.DataFrame,
    dict[str, list[tuple[str, Optional[Path], bool, str]]],
    dict,
]:
    """Read xlsx, build per-study ordered PDF entries, run manifest checks.

    Returns (xlsx_df, per_study_entries, stats_dict). Aborts the script if
    >5% of filenames can't be resolved.
    """
    print(f"[T16.D.3.A] Leyendo xlsx: {XLSX_PATH}")
    df = pd.read_excel(XLSX_PATH)
    if XLSX_COL_ID not in df.columns or XLSX_COL_FILENAME not in df.columns:
        raise SystemExit(
            f"xlsx missing required columns: {XLSX_COL_ID!r}, {XLSX_COL_FILENAME!r}"
        )

    print(f"[T16.D.3.A] Indexando PDFs en disco: {PDFS_DIR}")
    exact, skeleton = build_filename_index(PDFS_DIR)
    print(f"[T16.D.3.A]   total PDFs en disco: {len(exact)}")

    byte_exact = 0
    skel_matches = 0
    mismatches: list[tuple[str, str]] = []  # (covidence_id, filename)

    per_study: dict[str, list[tuple[str, Optional[Path], bool, str]]] = {}
    # Preserve xlsx order for grouping
    for _, row in df.iterrows():
        cov_id = str(row[XLSX_COL_ID]).strip()
        xlsx_fname = row[XLSX_COL_FILENAME]
        status = (
            str(row[XLSX_COL_STATUS]).strip()
            if XLSX_COL_STATUS in df.columns
            else ""
        )
        if not isinstance(xlsx_fname, str) or not xlsx_fname.strip():
            per_study.setdefault(cov_id, []).append((None, None, False, status))
            continue
        xlsx_fname = xlsx_fname.strip()
        resolved, used_skel = resolve_filename(xlsx_fname, exact, skeleton)
        if resolved is None:
            mismatches.append((cov_id, xlsx_fname))
            per_study.setdefault(cov_id, []).append(
                (xlsx_fname, None, False, status)
            )
            continue
        if used_skel:
            skel_matches += 1
        else:
            byte_exact += 1
        per_study.setdefault(cov_id, []).append(
            (xlsx_fname, PDFS_DIR / resolved, used_skel, status)
        )

    total_filenames = byte_exact + skel_matches + len(mismatches)
    print(f"[T16.D.3.A]   filenames totales en xlsx: {total_filenames}")
    print(f"[T16.D.3.A]   byte-exact matches:        {byte_exact}")
    print(f"[T16.D.3.A]   ASCII skeleton matches:    {skel_matches}")
    print(f"[T16.D.3.A]   mismatches:                {len(mismatches)}")

    stats = {
        "total_filenames": total_filenames,
        "byte_exact": byte_exact,
        "skeleton": skel_matches,
        "mismatches": mismatches,
    }

    if total_filenames > 0:
        miss_frac = len(mismatches) / total_filenames
        if miss_frac > MISMATCH_ABORT_FRACTION:
            print(
                f"[T16.D.3.A] ERROR: {len(mismatches)} mismatches "
                f"({miss_frac*100:.1f}%) supera el umbral del "
                f"{MISMATCH_ABORT_FRACTION*100:.0f}%. Abortando."
            )
            for cov_id, fname in mismatches[:50]:
                print(f"    {cov_id}: {fname!r}")
            raise SystemExit(2)

    return df, per_study, stats


# ---------------------------------------------------------------------------
# Abstract-source distribution helper
# ---------------------------------------------------------------------------
def _print_abstract_distribution(df: pd.DataFrame) -> None:
    """Print abstract coverage among successful rows and the source_abstract
    breakdown. Spanish-friendly output."""
    successes = df[df["source_method"] != "failed"]
    total_succ = len(successes)
    with_abs = int(successes["abstract"].notna().sum())
    pct = (with_abs / total_succ * 100) if total_succ else 0.0
    print()
    print("Abstract coverage AFTER enrichment:")
    print(f"  total successes: {total_succ}")
    print(f"  with abstract: {with_abs} ({pct:.1f}%)")
    print("  source_abstract distribution:")
    expected = (
        "pdf_metadata",
        "crossref",
        "pdfplumber_fallback",
        "semantic_scholar",
    )
    counts = successes["source_abstract"].value_counts(dropna=False).to_dict()
    for label in expected:
        print(f"    {label}: {int(counts.get(label, 0))}")
    none_count = int(successes["source_abstract"].isna().sum())
    print(f"    None: {none_count}")


# ---------------------------------------------------------------------------
# T16.D.3.A.bis — abstracts-only enrichment pass
# ---------------------------------------------------------------------------
def _backfill_source_abstract_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `source_abstract` column to a legacy CSV (9 cols, no provenance)
    by mapping existing abstracts back to the method that produced them.

    Mapping rules:
      - source_method=="pdf_metadata"        & abstract not null -> "pdf_metadata"
      - source_method=="doi_crossref"        & abstract not null -> "crossref"
      - source_method=="pdfplumber_fallback" & abstract not null -> "pdfplumber_fallback"
      - otherwise                                                -> None
    """
    if "source_abstract" not in df.columns:
        df["source_abstract"] = None

    def _classify(row: pd.Series) -> Optional[str]:
        existing = row.get("source_abstract")
        if isinstance(existing, str) and existing.strip():
            return existing
        abstract = row.get("abstract")
        if not isinstance(abstract, str) or not abstract.strip():
            return None
        method = row.get("source_method")
        if method == "pdf_metadata":
            return "pdf_metadata"
        if method == "doi_crossref":
            return "crossref"
        if method == "pdfplumber_fallback":
            return "pdfplumber_fallback"
        return None

    df["source_abstract"] = df.apply(_classify, axis=1)
    return df


def _append_warning(existing: object, new_warning: str) -> str:
    """Concatenate a new warning to an existing warnings cell (semicolon
    separator), deduplicating."""
    if not new_warning:
        return existing if isinstance(existing, str) else ""
    if isinstance(existing, str) and existing.strip():
        parts = [p.strip() for p in existing.split(";") if p.strip()]
        if new_warning not in parts:
            parts.append(new_warning)
        return "; ".join(parts)
    return new_warning


def run_abstracts_only_enrichment() -> int:
    """Read the existing CSV and run only Method 5 (Semantic Scholar) for
    studies that are successes with a DOI but no abstract. Writes the CSV
    in-place with the 10-column schema.
    """
    t_start = time.time()
    if not OUTPUT_CSV.exists():
        print(
            f"[T16.D.3.A.bis] ERROR: no existe {OUTPUT_CSV}. "
            "Ejecuta primero la extraccion completa."
        )
        return 2

    print(f"[T16.D.3.A.bis] Leyendo CSV existente: {OUTPUT_CSV}")
    df = pd.read_csv(OUTPUT_CSV, dtype={"covidence_id": str})
    print(f"[T16.D.3.A.bis] shape inicial: {df.shape}")

    # Backfill source_abstract from existing data (no-op if column already
    # populated coherently).
    df = _backfill_source_abstract_column(df)

    # Identify eligible rows.
    eligible_mask = (
        (df["source_method"] != "failed")
        & df["abstract"].isna()
        & df["doi"].notna()
        & df["doi"].astype(str).str.strip().ne("")
    )
    eligible_idx = df.index[eligible_mask].tolist()
    n_eligible = len(eligible_idx)
    print(
        f"[T16.D.3.A.bis] candidatos SS (DOI + sin abstract): {n_eligible}"
    )

    if "warnings" not in df.columns:
        df["warnings"] = None

    n_success = 0
    n_404 = 0
    n_failed = 0

    for i, idx in enumerate(eligible_idx, start=1):
        doi = str(df.at[idx, "doi"]).strip()
        if not doi:
            continue
        ss_abs, ss_err = fetch_abstract_semantic_scholar(doi)
        if ss_abs:
            df.at[idx, "abstract"] = ss_abs
            df.at[idx, "source_abstract"] = "semantic_scholar"
            n_success += 1
        if ss_err:
            existing = df.at[idx, "warnings"]
            existing_str = existing if isinstance(existing, str) else ""
            df.at[idx, "warnings"] = _append_warning(existing_str, ss_err)
            if ss_err.startswith("SS no record"):
                n_404 += 1
            else:
                n_failed += 1

        if i % SS_PROGRESS_EVERY == 0 or i == n_eligible:
            elapsed = time.time() - t_start
            print(
                f"[T16.D.3.A.bis]   {i}/{n_eligible} "
                f"exito: {n_success} 404: {n_404} fallos: {n_failed} "
                f"elapsed: {elapsed/60:.1f} min"
            )

    # Ensure all OUTPUT_COLUMNS exist and write in canonical order.
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLUMNS]
    df["year"] = df["year"].astype("Int64", errors="ignore")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[T16.D.3.A.bis] CSV reescrito: {OUTPUT_CSV}  shape={df.shape}")

    # Final summary
    elapsed_min = (time.time() - t_start) / 60.0
    print()
    print("=== T16.D.3.A.bis summary ===")
    print(f"SS candidates (DOI + no abstract): {n_eligible}")
    print(f"SS calls attempted: {n_eligible}")
    pct_succ = (n_success / n_eligible * 100) if n_eligible else 0.0
    print(
        f"SS calls successful (abstract retrieved): {n_success} "
        f"({pct_succ:.1f}%)"
    )
    print(f"SS calls 404 (paper not indexed): {n_404}")
    print(f"SS calls failed (other errors): {n_failed}")

    _print_abstract_distribution(df)

    print()
    print(f"elapsed SS pass: {elapsed_min:.1f} min")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract bibliographic metadata from the deporte gold-standard "
            "PDFs (T16.D.3.A) and, optionally, enrich abstracts via "
            "Semantic Scholar (T16.D.3.A.bis Method 5)."
        )
    )
    parser.add_argument(
        "--abstracts-only",
        action="store_true",
        help=(
            "Skip PDF re-extraction. Read the existing CSV and run only "
            "Method 5 (Semantic Scholar by DOI) on rows with a DOI but no "
            "abstract. Writes the 10-column schema in-place."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.abstracts_only:
        return run_abstracts_only_enrichment()

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_xlsx, per_study, manifest_stats = build_manifest()

    # Ordered list of unique covidence IDs preserving xlsx order
    ordered_ids: list[str] = []
    seen = set()
    for cov_id in df_xlsx[XLSX_COL_ID].astype(str).str.strip().tolist():
        if cov_id not in seen:
            seen.add(cov_id)
            ordered_ids.append(cov_id)

    print(f"[T16.D.3.A] Procesando {len(ordered_ids)} estudios...")
    rows: list[dict] = []
    method_counts = {
        "pdf_metadata": 0,
        "doi_crossref": 0,
        "pdfplumber_fallback": 0,
        "failed": 0,
    }
    progress_every = 50

    for i, cov_id in enumerate(ordered_ids, start=1):
        entries = per_study.get(cov_id, [])
        row = process_study(cov_id, entries)
        rows.append(row)
        method = row["source_method"] or "failed"
        method_counts[method] = method_counts.get(method, 0) + 1
        if i % progress_every == 0 or i == len(ordered_ids):
            done = i
            succ = sum(
                method_counts[m]
                for m in (
                    "pdf_metadata",
                    "doi_crossref",
                    "pdfplumber_fallback",
                )
            )
            elapsed = time.time() - t_start
            print(
                f"[T16.D.3.A]   {done}/{len(ordered_ids)} "
                f"exito acumulado: {succ} "
                f"({succ/done*100:.1f}%) "
                f"elapsed: {elapsed/60:.1f} min"
            )

    df_out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    # Ensure year is integer-or-null
    df_out["year"] = df_out["year"].astype("Int64", errors="ignore")
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[T16.D.3.A] CSV escrito: {OUTPUT_CSV}  shape={df_out.shape}")

    # Summary block (stdout)
    total = len(df_out)
    success = total - method_counts["failed"]
    print()
    print("=== T16.D.3.A summary ===")
    print(f"total studies: {total}")
    pct = lambda n: (n / total * 100) if total else 0.0
    print(f"success: {success} ({pct(success):.1f}%)")
    print(
        f"  pdf_metadata: {method_counts['pdf_metadata']} "
        f"({pct(method_counts['pdf_metadata']):.1f}%)"
    )
    print(
        f"  doi_crossref: {method_counts['doi_crossref']} "
        f"({pct(method_counts['doi_crossref']):.1f}%)"
    )
    print(
        f"  pdfplumber_fallback: {method_counts['pdfplumber_fallback']} "
        f"({pct(method_counts['pdfplumber_fallback']):.1f}%)"
    )
    print(
        f"failed: {method_counts['failed']} ({pct(method_counts['failed']):.1f}%)"
    )
    print(f"elapsed total: {(time.time() - t_start)/60:.1f} min")
    print(
        f"manifest: byte-exact={manifest_stats['byte_exact']}, "
        f"skeleton={manifest_stats['skeleton']}, "
        f"mismatches={len(manifest_stats['mismatches'])}"
    )

    # Abstract coverage breakdown (T16.D.3.A.bis schema)
    _print_abstract_distribution(df_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
