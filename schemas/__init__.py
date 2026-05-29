"""SAM Pydantic schemas (T17.C.0).

Anchored to Qwen real output (see docs/iterations/T17.C.0_qwen_field_mapping.md).
All per-paper agent schemas (A3..A6) carry a `paper_id` field that mirrors
`pmid` for orchestrator compatibility.

Public re-exports:
- common: AgentResultBase, Backend
- protocol: ProtocolOutput, ProtocolBody, Framework
- papers: PapersRawOutput, Paper, SearchStats
- screening: ScreeningOutput, VoteEntry, ScreeningDecision
- fulltext: FulltextOutput, FulltextVoteEntry, FulltextDecision
- extraction: ExtractionOutput, StudyCharacteristics, InterventionExposure,
              Outcome, NGroups, Concordance, ExtractionQuality,
              ExtractionConfidence, EffectSizeType
- quality: QualityOutput, QualityAssessment, DomainJudgment,
           StudyDesignDetected, QualityTool
- synthesis: SynthesisOutput
- manuscript: ManuscriptOutput, ManuscriptSections
- explainer: ExplainerOutput
- manifest: Manifest, RunConfig, AgentStatus, AgentStatusStr, SubsetSpec
"""
from __future__ import annotations

from .common import AgentResultBase, Backend
from .explainer import ExplainerOutput
from .extraction import (
    Concordance,
    EffectSizeType,
    ExtractionConfidence,
    ExtractionOutput,
    ExtractionQuality,
    InterventionExposure,
    NGroups,
    Outcome,
    StudyCharacteristics,
)
from .fulltext import FulltextDecision, FulltextOutput, FulltextVoteEntry
from .manifest import (
    AgentStatus,
    AgentStatusStr,
    Manifest,
    RunConfig,
    SubsetSpec,
)
from .manuscript import ManuscriptOutput, ManuscriptSections
from .papers import Paper, PapersRawOutput, SearchStats
from .protocol import Framework, ProtocolBody, ProtocolOutput
from .quality import (
    DomainJudgment,
    QualityAssessment,
    QualityOutput,
    QualityTool,
    StudyDesignDetected,
)
from .screening import ScreeningDecision, ScreeningOutput, VoteEntry
from .synthesis import SynthesisOutput

__all__ = [
    # common
    "AgentResultBase",
    "Backend",
    # protocol
    "ProtocolOutput",
    "ProtocolBody",
    "Framework",
    # papers
    "PapersRawOutput",
    "Paper",
    "SearchStats",
    # screening
    "ScreeningOutput",
    "VoteEntry",
    "ScreeningDecision",
    # fulltext
    "FulltextOutput",
    "FulltextVoteEntry",
    "FulltextDecision",
    # extraction
    "ExtractionOutput",
    "StudyCharacteristics",
    "InterventionExposure",
    "Outcome",
    "NGroups",
    "Concordance",
    "ExtractionQuality",
    "ExtractionConfidence",
    "EffectSizeType",
    # quality
    "QualityOutput",
    "QualityAssessment",
    "DomainJudgment",
    "StudyDesignDetected",
    "QualityTool",
    # synthesis
    "SynthesisOutput",
    # manuscript
    "ManuscriptOutput",
    "ManuscriptSections",
    # explainer
    "ExplainerOutput",
    # manifest
    "Manifest",
    "RunConfig",
    "AgentStatus",
    "AgentStatusStr",
    "SubsetSpec",
]
