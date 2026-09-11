# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive analyzer SPI — candidates only, no persistence or actions (PREVENTIVE R6)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.preventive.analyzer_descriptor import PreventiveAnalyzerDescriptor
from intergrax.contracts.preventive.context import PreventiveAnalysisInput
from intergrax.contracts.preventive.recommendation import PreventiveRecommendationCandidate


@runtime_checkable
class PreventiveAnalyzer(Protocol):
    @property
    def analyzer_id(self) -> str: ...

    @property
    def analyzer_namespace(self) -> str: ...

    @property
    def priority(self) -> int: ...

    @property
    def analyzer_version(self) -> str: ...

    @property
    def descriptor(self) -> PreventiveAnalyzerDescriptor: ...

    def analyze(
        self,
        analysis_input: PreventiveAnalysisInput,
    ) -> tuple[PreventiveRecommendationCandidate, ...]:
        """
        Pure recommendation reasoning.

        Must not persist recommendations, create Problems, or execute remediation.
        """


__all__ = ["PreventiveAnalyzer"]
