"""A9 explainer schema (single) — Excel + Spanish markdown traceability report."""
from __future__ import annotations

from pydantic import Field

from .common import AgentResultBase


class ExplainerOutput(AgentResultBase):
    """Output for sam_a9_explainer/explainer.json."""

    excel_path: str = Field(..., description="Path to papers_raw.xlsx (27-column trace).")
    report_path: str = Field(..., description="Path to explainability.md (Spanish).")
    n_total: int = Field(..., ge=0)
    n_included: int = Field(..., ge=0)
    n_excluded: int = Field(..., ge=0)
