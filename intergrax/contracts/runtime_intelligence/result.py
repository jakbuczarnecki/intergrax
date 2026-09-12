# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Versioned analyzer output envelope (W6-B)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.runtime_intelligence.evidence import IntelligenceEvidence
from intergrax.contracts.runtime_intelligence.recommendation import IntelligenceRecommendation

RUNTIME_INTELLIGENCE_RESULT_SCHEMA_VERSION = "runtime_intelligence_result.v1"


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceResult:
    """
    Analysis product owned by the Runtime Intelligence plane.

    Does not mutate execution state; recommendations remain advisory.
    """

    analysis_summary: str
    confidence: float
    evidence: tuple[IntelligenceEvidence, ...]
    recommendations: tuple[IntelligenceRecommendation, ...]
    analyzer_id: str
    analyzer_version: str
    schema_version: str = RUNTIME_INTELLIGENCE_RESULT_SCHEMA_VERSION
    degraded: bool = False

    def __post_init__(self) -> None:
        if not self.analysis_summary.strip():
            raise ValueError("analysis_summary must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        if not self.evidence:
            raise ValueError("evidence must be non-empty")
        if not self.analyzer_id.strip() or not self.analyzer_version.strip():
            raise ValueError("analyzer identity required")
        if not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty")


__all__ = [
    "RUNTIME_INTELLIGENCE_RESULT_SCHEMA_VERSION",
    "RuntimeIntelligenceResult",
]
