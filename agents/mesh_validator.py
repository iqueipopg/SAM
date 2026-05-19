"""MeSH validator — canonical NCBI MeSH descriptor lookup for A1 queries.

T17.A1 Fase D.2 stub. The real implementation lands in Fase E.2 / T17.A1.1.

In this stub, ``validate_query`` is a passthrough (no modifications). This
allows the paired-evaluation plumbing in ``experiments/eval_hero_a1.py`` to be
exercised end-to-end during D.2 testing without changing query semantics. The
substantive intervention behavior will be added — and unit-tested separately —
in Fase E.2 via ``tests/test_mesh_validator.py``.

Public API (frozen by the plan ``elegant-tumbling-turtle.md``):
    - ``load_catalog(path)``        → ``set[str]``  (E.2: parses NCBI ascii MeSH)
    - ``download_catalog(target)``  → ``None``      (E.2: fetches d{year}.bin)
    - ``validate_query(query, …)``  → ``(str, int, list[dict])``
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_CATALOG_PATH = Path("data/mesh_catalog.json")


def load_catalog(catalog_path: Path = DEFAULT_CATALOG_PATH) -> set[str]:
    """Load the canonical NCBI MeSH descriptor set from disk.

    Raises ``NotImplementedError`` in this D.2 stub; the real implementation
    lands in T17.A1.1 / Fase E.2 alongside the catalog acquisition pipeline.
    """
    raise NotImplementedError(
        "mesh_validator.load_catalog will be implemented in T17.A1.1 / Fase E.2"
    )


def download_catalog(target_path: Path = DEFAULT_CATALOG_PATH) -> None:
    """Download the NCBI MeSH ASCII catalog and cache it as JSON.

    Raises ``NotImplementedError`` in this D.2 stub; the real implementation
    lands in T17.A1.1 / Fase E.2 (uses ``agents.retry.retry_with_backoff``).
    """
    raise NotImplementedError(
        "mesh_validator.download_catalog will be implemented in T17.A1.1 / Fase E.2"
    )


def validate_query(
    query: str,
    catalog: set[str] | None = None,
) -> tuple[str, int, list[dict]]:
    """Validate ``[MeSH]`` tags in an A1-generated PubMed Boolean query.

    Stub for D.2: passthrough. Returns the query unchanged, ``modifications=0``,
    and an empty ``near_misses`` list. The real implementation in Fase E.2 will:
      - Match each ``"phrase"[MeSH]`` against ``catalog`` (case-insensitive).
      - Downgrade non-canonical tags to ``[Title/Abstract]``.
      - Log near-misses via ``difflib.get_close_matches`` (informational only).

    Args:
        query: The Boolean PubMed query string emitted by A1.
        catalog: Optional pre-loaded descriptor set; if ``None``, the real
            implementation will call ``load_catalog()`` internally.

    Returns:
        ``(modified_query, modifications_count, near_misses)`` where
        ``near_misses`` is a list of ``{"phrase": str, "nearest_canonical": str | None}``.
    """
    return query, 0, []
