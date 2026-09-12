# © Artur Czarnecki. All rights reserved.

"""Capability extractor plugin contract (DS-E2E-15J-L3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    BehavioralComparisonResult,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline.contracts import (
    CapabilityDimensionId,
    CapabilityObservation,
    LimitationObservation,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)


@dataclass(frozen=True, slots=True)
class CapabilityExtractorResult:
    capabilities: tuple[CapabilityObservation, ...]
    limitations: tuple[LimitationObservation, ...]


class CapabilityExtractor(Protocol):
    """Pluggable factual capability/limitation extraction for one dimension."""

    @property
    def extractor_id(self) -> str: ...

    @property
    def dimension_id(self) -> CapabilityDimensionId: ...

    def extract(
        self,
        *,
        profile_key: str,
        outcome: ModelQualificationOutcome | None,
        behavioral_comparison: BehavioralComparisonResult | None,
    ) -> CapabilityExtractorResult: ...


__all__ = ["CapabilityExtractor", "CapabilityExtractorResult"]
