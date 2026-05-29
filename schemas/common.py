"""Common base schema for all SAM agent outputs.

T17.C.0 — anchored to Qwen real output (see docs/iterations/T17.C.0_qwen_field_mapping.md).

D8 — `tokens_in` / `tokens_out` are Optional[int] = None.
NEVER 0. None means "not measured"; 0 would falsely claim "zero tokens" and break
cost metrics in C.1.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Backend = Literal["claude", "qwen"]


class AgentResultBase(BaseModel):
    """Common header fields every agent JSON must carry.

    Concrete agent schemas inherit from this and add their own fields.
    Subclasses default to ConfigDict(extra="forbid") unless they override
    (e.g. A7 uses extra="allow" because R metafor outputs are rich).
    """

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(..., description="Subagent identifier, e.g. 'sam_a1_protocol'.")
    backend: Backend = Field(..., description="Execution backend: 'claude' or 'qwen'.")
    timestamp: str = Field(..., description="ISO 8601 timestamp of agent completion.")
    tokens_in: Optional[int] = Field(
        default=None,
        description="Input tokens consumed. None = not measured (NEVER set to 0).",
    )
    tokens_out: Optional[int] = Field(
        default=None,
        description="Output tokens produced. None = not measured (NEVER set to 0).",
    )
