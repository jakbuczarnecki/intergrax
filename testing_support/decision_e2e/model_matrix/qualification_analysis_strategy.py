# © Artur Czarnecki. All rights reserved.

"""Analysis strategy contract for multi-model qualification (R6)."""

from __future__ import annotations

from typing import Protocol

from testing_support.decision_e2e.local_qualification_session.contracts import SafetyGateOutcome
from testing_support.decision_e2e.model_matrix.analysis import FifteenKBEffectR6
from testing_support.decision_e2e.model_matrix.qualification_plan import ModelAvailability


class QualificationAnalysisStrategy(Protocol):
    """Case classification, metrics derivation, and per-model verdict generation."""

    def classify_fifteen_kb_effect(
        self,
        *,
        availability: ModelAvailability,
        model_overcommit_count: int,
        revision_attempted: int,
        typed_context_delivered: int,
        repair_count: int,
        total_runs: int,
        third_pass: int,
        reconciliation_leak: SafetyGateOutcome,
        alignment_events: int,
    ) -> FifteenKBEffectR6:
        ...


class DefaultQualificationAnalysisStrategy:
    """Default R6 Case A/B/C classification (15K-B effect)."""

    def classify_fifteen_kb_effect(
        self,
        *,
        availability: ModelAvailability,
        model_overcommit_count: int,
        revision_attempted: int,
        typed_context_delivered: int,
        repair_count: int,
        total_runs: int,
        third_pass: int,
        reconciliation_leak: SafetyGateOutcome,
        alignment_events: int,
    ) -> FifteenKBEffectR6:
        from testing_support.decision_e2e.model_matrix.analysis import (
            classify_fifteen_kb_effect,
        )

        return classify_fifteen_kb_effect(
            availability=availability,
            model_overcommit_count=model_overcommit_count,
            revision_attempted=revision_attempted,
            typed_context_delivered=typed_context_delivered,
            repair_count=repair_count,
            total_runs=total_runs,
            third_pass=third_pass,
            reconciliation_leak=reconciliation_leak,
            alignment_events=alignment_events,
        )


__all__ = [
    "DefaultQualificationAnalysisStrategy",
    "QualificationAnalysisStrategy",
]
