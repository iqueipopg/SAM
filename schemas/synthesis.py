"""A7 synthesis schema (single) — R/metafor outputs are rich and varied.

Per mapping doc, A7 uses `extra='allow'` so R metafor analyses, plots,
GRADE blocks, sensitivity panels, etc. are not constrained. The only
structural guarantee is that `analyses` and `plots` exist.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from .common import AgentResultBase


class SynthesisOutput(AgentResultBase):
    """Output for sam_a7_synthesis/results.json.

    Permissive on purpose: R/metafor emits an open-ended dictionary of
    analyses and plot paths. Structural check is limited to presence of
    `analyses` and `plots`.
    """

    model_config = ConfigDict(extra="allow")

    analyses: Dict[str, Any] = Field(
        ..., description="Per-outcome meta-analysis results (REML, I², etc.)."
    )
    plots: Dict[str, str] = Field(
        ..., description="Map of plot key -> file path (forest/funnel/etc.)."
    )
    r_script_path: Optional[str] = Field(
        default=None, description="Path to exported R script (meta_analysis.R)."
    )
