# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Per-paper _index.json operations — idempotent append + atomic rewrite.

Each per-paper agent dir holds an _index.json with a list of IndexEntry
records: { slug, paper_id, timestamp }. Operations on this file MUST be
idempotent: committing the same slug twice does NOT append a duplicate.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .manifest import utc_now_iso
from .paths import AGENT_MODES, RunPaths


@dataclass(frozen=True)
class IndexEntry:
    """One entry in a per-paper agent's _index.json."""

    slug: str
    paper_id: str
    timestamp: str

    @classmethod
    def from_dict(cls, d: dict) -> "IndexEntry":
        return cls(
            slug=str(d["slug"]),
            paper_id=str(d["paper_id"]),
            timestamp=str(d["timestamp"]),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _index_file(agent_dir: Path) -> Path:
    return agent_dir / "_index.json"


def read_index(agent_dir: Path) -> List[IndexEntry]:
    """Read _index.json from the agent dir. Returns [] if missing."""
    f = _index_file(agent_dir)
    if not f.exists():
        return []
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Treat unreadable index as empty to allow recovery via audit.
        return []
    if not isinstance(raw, list):
        return []
    out: List[IndexEntry] = []
    for item in raw:
        if isinstance(item, dict) and {"slug", "paper_id", "timestamp"} <= set(item):
            try:
                out.append(IndexEntry.from_dict(item))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def index_contains(agent_dir: Path, slug: str) -> bool:
    return any(e.slug == slug for e in read_index(agent_dir))


def _write_index_atomic(agent_dir: Path, entries: Iterable[IndexEntry]) -> None:
    """Rewrite _index.json atomically (tmp + os.replace)."""
    agent_dir.mkdir(parents=True, exist_ok=True)
    target = _index_file(agent_dir)
    payload = json.dumps(
        [e.to_dict() for e in entries], indent=2, ensure_ascii=False
    )
    fd, tmp_name = tempfile.mkstemp(
        prefix="._index_", suffix=".tmp", dir=str(agent_dir)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass
        raise


def append_index_unique(
    agent_dir: Path, entry: IndexEntry
) -> bool:
    """Append `entry` to _index.json if its slug is not already present.

    Returns True if a new entry was appended, False if it was a no-op
    (slug already present).
    """
    existing = read_index(agent_dir)
    if any(e.slug == entry.slug for e in existing):
        return False
    existing.append(entry)
    _write_index_atomic(agent_dir, existing)
    return True


def index_slugs(agent_dir: Path) -> set[str]:
    return {e.slug for e in read_index(agent_dir)}


def find_duplicates(agent_dir: Path) -> List[str]:
    """Return slugs that appear more than once in _index.json (should be empty)."""
    counts: dict[str, int] = {}
    for e in read_index(agent_dir):
        counts[e.slug] = counts.get(e.slug, 0) + 1
    return [slug for slug, n in counts.items() if n > 1]


# Convenience helper used by commit.py
def make_entry(slug: str, paper_id: str, timestamp: Optional[str] = None) -> IndexEntry:
    return IndexEntry(
        slug=slug,
        paper_id=paper_id,
        timestamp=timestamp or utc_now_iso(),
    )


def assert_per_paper(run_id: str, agent_id: str) -> Path:
    """Helper: validate agent_id is per-paper, return its dir for `run_id`."""
    if AGENT_MODES.get(agent_id) != "per_paper":
        raise ValueError(f"{agent_id} is not a per-paper agent")
    return RunPaths(run_id).agent_dir(agent_id)
