# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""MeSH validator — canonical NCBI MeSH descriptor lookup for A1 queries.

T17.A1.1 / Fase E.2 implementation. Locked by pre-registration
`docs/iterations/T17.A1.1_hypothesis.md` (tag `pre-T17.A1.1`).

Public API (frozen by the plan ``elegant-tumbling-turtle.md``):
    - ``load_catalog(path)``        → ``set[str]``  (lowercase descriptors)
    - ``download_catalog(target)``  → ``None``      (fetches d{year}.bin)
    - ``validate_query(query, …)``  → ``(str, int, list[dict])``

Behavior:
- Extract each ``"phrase"[MeSH]`` token using ``r'"([^"]+)"\\[MeSH\\]'``.
- Case-insensitive lookup against the catalog set (already lowercased).
- Hit  → keep the ``[MeSH]`` tag unchanged.
- Miss → replace ``[MeSH]`` with ``[Title/Abstract]`` for that phrase.
- For each miss, record an informational ``difflib.get_close_matches`` near-miss.

Determinism: given a fixed catalog and a fixed query, the output is invariant.
No sampling, no LLM, no network during ``validate_query``.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import requests

from agents.retry import retry_with_backoff

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_PATH = Path("data/mesh_catalog.json")

# Same pattern A1 uses internally (agent1_protocol.py:188).
_MESH_TAG_RE = re.compile(r'"([^"]+)"\[MeSH\]')

# Candidate annual ASCII MeSH files, ordered most-recent first.
# NLM layout (verified 2026-05): files live under per-year subdirs:
#   https://nlmpubs.nlm.nih.gov/projects/mesh/{year}/asciimesh/d{year}.bin
_NCBI_MESH_BASE_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/"
_NCBI_MESH_FILES = [
    "2025/asciimesh/d2025.bin",
    "2024/asciimesh/d2024.bin",
    "2023/asciimesh/d2023.bin",
]

# Module-level cache: (mtime, catalog_set, descriptors_sorted_list).
_CATALOG_CACHE: dict[str, object] = {
    "path": None,
    "mtime": None,
    "set": None,
    "sorted_list": None,
}


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def load_catalog(catalog_path: Path = DEFAULT_CATALOG_PATH) -> set[str]:
    """Load the canonical NCBI MeSH descriptor set (lowercase) from disk.

    Cached at module level keyed by file mtime: subsequent calls without
    on-disk changes are O(1). If the catalog file is replaced on disk
    (mtime advances), the cache is invalidated and the file re-read.

    Args:
        catalog_path: JSON file produced by ``download_catalog``.

    Returns:
        ``set[str]`` of lowercased descriptors.
    """
    path = Path(catalog_path)
    try:
        mtime = os.stat(path).st_mtime
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"MeSH catalog not found at {path}. Run "
            "`python -c 'from agents.mesh_validator import download_catalog; "
            "download_catalog()'` first."
        ) from e

    cached_path = _CATALOG_CACHE.get("path")
    cached_mtime = _CATALOG_CACHE.get("mtime")
    cached_set = _CATALOG_CACHE.get("set")
    if (
        cached_path == str(path)
        and cached_mtime == mtime
        and isinstance(cached_set, set)
    ):
        return cached_set

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Prefer the explicitly lowercased field if present; otherwise lowercase
    # the canonical descriptors on the fly.
    if "descriptors_lowercase" in data and isinstance(
        data["descriptors_lowercase"], list
    ):
        lowered = [str(d).strip().lower() for d in data["descriptors_lowercase"]]
    else:
        lowered = [str(d).strip().lower() for d in data.get("descriptors", [])]

    catalog_set = {d for d in lowered if d}
    sorted_list = sorted(catalog_set)

    _CATALOG_CACHE.update(
        {
            "path": str(path),
            "mtime": mtime,
            "set": catalog_set,
            "sorted_list": sorted_list,
        }
    )
    return catalog_set


def _catalog_sorted_list(catalog: set[str]) -> list[str]:
    """Return a sorted list view of ``catalog`` (cached when possible)."""
    cached_set = _CATALOG_CACHE.get("set")
    cached_list = _CATALOG_CACHE.get("sorted_list")
    if cached_set is catalog and isinstance(cached_list, list):
        return cached_list
    return sorted(catalog)


# ---------------------------------------------------------------------------
# Catalog download
# ---------------------------------------------------------------------------


def _parse_mh_lines(content: str) -> list[str]:
    """Parse all ``MH = ...`` lines from an NCBI ASCII MeSH file."""
    descriptors: list[str] = []
    for line in content.splitlines():
        # Format: "MH = Descriptor Name" (note the spaces around '=').
        if line.startswith("MH = "):
            name = line[5:].strip()
            if name:
                descriptors.append(name)
    return descriptors


def _http_get(url: str) -> requests.Response:
    """Single GET wrapped by retry_with_backoff. Raises on 4xx/5xx."""
    def _do_get() -> requests.Response:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp

    return retry_with_backoff(
        _do_get,
        retryable_exceptions=(requests.exceptions.RequestException,),
    )


def _try_pypi_fallback() -> Optional[list[str]]:
    """Last-resort fallback: try installed Python packages if available."""
    # We do NOT add new dependencies. Only use if already in venv.
    for modname in ("pymed", "pymesh", "mesh"):
        try:
            mod = __import__(modname)
        except ImportError:
            continue
        # No standardized API across these libs; best-effort introspection.
        for attr in ("descriptors", "get_descriptors", "all_descriptors"):
            fn = getattr(mod, attr, None)
            if callable(fn):
                try:
                    result = fn()
                    if isinstance(result, (list, set, tuple)):
                        out = [str(x).strip() for x in result if str(x).strip()]
                        if out:
                            return out
                except Exception:
                    continue
    return None


def _write_blocked_report(
    blocked_path: Path,
    attempts: list[dict],
) -> None:
    """Write a structured BLOCKED markdown report describing what failed."""
    blocked_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BLOCKED — MeSH catalog acquisition failed",
        "",
        f"Timestamp (UTC): {_dt.datetime.now(_dt.timezone.utc).isoformat()}",
        "",
        "## Attempts",
        "",
    ]
    for a in attempts:
        lines.append(f"- **URL**: `{a.get('url', '?')}`")
        lines.append(f"  - status_code: `{a.get('status_code')}`")
        lines.append(f"  - error: `{a.get('error')}`")
    lines.append("")
    lines.append(
        "Sprint T17.A1.1 cannot proceed without the canonical MeSH catalog."
    )
    blocked_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_catalog(target_path: Path = DEFAULT_CATALOG_PATH) -> None:
    """Download the latest NCBI MeSH ASCII descriptor file, parse, persist.

    Cascade:
        1. Primary URLs in ``_NCBI_MESH_FILES`` (d2025 → d2024 → d2023).
        2. PyPI fallback if a relevant package is installed in the venv.
        3. Total failure → write ``outputs/eval/T17.A1/BLOCKED_mesh.md``
           and raise ``RuntimeError``.

    The persisted JSON includes both canonical-case descriptors and a
    pre-lowercased list to avoid a second pass at load time.
    """
    target_path = Path(target_path)
    attempts: list[dict] = []
    descriptors: Optional[list[str]] = None
    source_file: Optional[str] = None
    source_url: Optional[str] = None

    for fname in _NCBI_MESH_FILES:
        url = _NCBI_MESH_BASE_URL + fname
        attempt: dict = {"url": url, "status_code": None, "error": None}
        try:
            resp = _http_get(url)
            attempt["status_code"] = resp.status_code
            content = resp.text
            parsed = _parse_mh_lines(content)
            if not parsed:
                attempt["error"] = "no MH lines parsed (empty payload?)"
                attempts.append(attempt)
                continue
            descriptors = parsed
            # ``source_file`` records just the basename for readability;
            # ``source_url`` keeps the fully-qualified URL.
            source_file = fname.rsplit("/", 1)[-1]
            source_url = url
            attempts.append(attempt)
            logger.info(
                "[mesh_validator] Downloaded %s (%d descriptors) from %s",
                source_file,
                len(parsed),
                url,
            )
            break
        except requests.exceptions.HTTPError as e:
            attempt["status_code"] = getattr(e.response, "status_code", None)
            attempt["error"] = str(e)
            attempts.append(attempt)
            continue
        except Exception as e:  # pragma: no cover (network-level errors)
            attempt["error"] = repr(e)
            attempts.append(attempt)
            continue

    if descriptors is None:
        # Try Python-package fallback (only if pre-installed).
        pypi_result = _try_pypi_fallback()
        if pypi_result:
            descriptors = pypi_result
            source_file = "pypi-fallback"
            source_url = "pypi"
            attempts.append(
                {
                    "url": "pypi://pymed|pymesh|mesh",
                    "status_code": None,
                    "error": None,
                }
            )

    if descriptors is None:
        blocked_path = Path("outputs/eval/T17.A1/BLOCKED_mesh.md")
        _write_blocked_report(blocked_path, attempts)
        raise RuntimeError(
            "MeSH catalog download failed for all candidates. "
            f"See {blocked_path} for details."
        )

    # Normalize: dedupe by lowercased form, preserve canonical casing of
    # the first occurrence.
    seen: set[str] = set()
    canonical: list[str] = []
    lowered: list[str] = []
    for d in descriptors:
        d_clean = d.strip()
        if not d_clean:
            continue
        key = d_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        canonical.append(d_clean)
        lowered.append(key)

    payload = {
        "descriptors": canonical,
        "descriptors_lowercase": lowered,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_file": source_file,
        "source_url": source_url,
        "count": len(canonical),
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, target_path)

    # Invalidate cache so next load_catalog() reads fresh content.
    _CATALOG_CACHE.update(
        {"path": None, "mtime": None, "set": None, "sorted_list": None}
    )
    logger.info(
        "[mesh_validator] Catalog persisted to %s (%d descriptors, source=%s)",
        target_path,
        len(canonical),
        source_file,
    )


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------


def validate_query(
    query: str,
    catalog: Optional[set[str]] = None,
) -> tuple[str, int, list[dict]]:
    """Validate ``[MeSH]`` tags in an A1-generated PubMed Boolean query.

    Args:
        query: The Boolean PubMed query string emitted by A1.
        catalog: Optional pre-loaded descriptor set (lowercased). If None,
            ``load_catalog()`` is called (module-cached).

    Returns:
        Tuple of:
        - ``modified_query`` (``str``): the query with non-canonical
          ``[MeSH]`` tags downgraded to ``[Title/Abstract]``.
        - ``modifications_count`` (``int``): number of downgrades.
        - ``near_misses`` (``list[dict]``): one entry per Miss with keys
          ``phrase`` (original casing) and ``nearest_canonical``
          (best ``difflib`` match at cutoff 0.7, or ``None``).
          Informational only — does not influence the substitution.
    """
    if not query:
        return query, 0, []

    if catalog is None:
        catalog = load_catalog()

    sorted_catalog = _catalog_sorted_list(catalog)

    phrases = _MESH_TAG_RE.findall(query)
    if not phrases:
        return query, 0, []

    modified = query
    modifications = 0
    near_misses: list[dict] = []
    # Track which (lowercased) phrases we already downgraded to avoid
    # repeating the substitution on subsequent occurrences (re.sub already
    # replaces all, but we count once per *distinct* phrase miss).
    handled_misses: set[str] = set()

    for phrase in phrases:
        phrase_lower = phrase.strip().lower()
        if phrase_lower in catalog:
            # Hit — keep the [MeSH] tag.
            continue

        # Miss — downgrade.
        if phrase_lower in handled_misses:
            continue
        handled_misses.add(phrase_lower)

        # Escape the phrase for use in a regex pattern (handles parens,
        # brackets, etc. that A1 might emit).
        escaped = re.escape(phrase)
        pattern = r'"' + escaped + r'"\[MeSH\]'
        replacement = f'"{phrase}"[Title/Abstract]'
        new_modified, n_subs = re.subn(pattern, replacement, modified)
        if n_subs > 0:
            modified = new_modified
            modifications += 1

        # Diagnostic near-miss (informational only).
        matches = difflib.get_close_matches(
            phrase_lower, sorted_catalog, n=1, cutoff=0.7
        )
        near_misses.append(
            {
                "phrase": phrase,
                "nearest_canonical": matches[0] if matches else None,
            }
        )

    return modified, modifications, near_misses
