"""A8 manuscript schema (single) — narrative manuscript output."""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import AgentResultBase


class ManuscriptSections(BaseModel):
    """The 6 canonical IMRAD+ sections expected for the manuscript."""

    model_config = ConfigDict(extra="forbid")

    abstract: str
    introduction: str
    methods: str
    results: str
    discussion: str
    conclusion: str


class ManuscriptOutput(AgentResultBase):
    """Output for sam_a8_manuscript/manuscript.json."""

    sections: ManuscriptSections
    references: List[Any] = Field(
        ..., description="Reference list (raw refs; structure handled by exporter)."
    )
    pdf_path: Optional[str] = Field(
        default=None, description="Path to compiled PDF; null if compilation skipped."
    )
