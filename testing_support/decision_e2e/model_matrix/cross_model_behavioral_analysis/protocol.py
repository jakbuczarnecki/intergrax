# © Artur Czarnecki. All rights reserved.

"""Behavior analyzer plugin contract (DS-E2E-15J-L2)."""

from __future__ import annotations

from typing import Protocol

from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    ComparisonArea,
    ComparisonAreaFinding,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)


class BehaviorAnalyzer(Protocol):
    """Pluggable comparison for one behavioral dimension across model outcomes."""

    @property
    def analyzer_id(self) -> str: ...

    @property
    def comparison_area(self) -> ComparisonArea: ...

    def analyze(
        self,
        outcomes: tuple[ModelQualificationOutcome, ...],
    ) -> ComparisonAreaFinding: ...


__all__ = ["BehaviorAnalyzer"]
