# © Artur Czarnecki. All rights reserved.

"""Test-only external ERL grouping strategy — groups by correlation_id (ERL-DIAG-001C-H)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.enterprise_reliability.diagnostics import (
    RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID,
    RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION,
    ExternalEffectReliabilityObservation,
    ReliabilityCaseSubjectRef,
    ReliabilityProblemGroupingStrategyId,
    ReliabilityProblemGroupingStrategyVersion,
    reliability_correlation_subject_index_token,
)


@dataclass(frozen=True, slots=True)
class _CorrelationGroupingSubjectRef:
    tenant_id: str
    correlation_id: str

    @property
    def index_token(self) -> str:
        return reliability_correlation_subject_index_token(self.correlation_id)


class CorrelationGroupingStrategy:
    """Plugin implementation: one Problem per tenant + correlation_id."""

    @property
    def strategy_id(self) -> ReliabilityProblemGroupingStrategyId:
        return RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID

    @property
    def strategy_version(self) -> ReliabilityProblemGroupingStrategyVersion:
        return RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION

    def group(
        self,
        observation: ExternalEffectReliabilityObservation,
    ) -> ReliabilityCaseSubjectRef:
        return _CorrelationGroupingSubjectRef(
            tenant_id=observation.tenant_id,
            correlation_id=observation.correlation.correlation_id,
        )


__all__ = ["CorrelationGroupingStrategy"]
