# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory adapter for ``AutonomyDecisionRepository`` (SELF-HEALING R6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.self_healing.autonomy.evaluation_result import AutonomyEvaluationResult
from intergrax.contracts.self_healing.autonomy.repository import AutonomyDecisionCorrelationQuery


@dataclass
class InMemoryAutonomyDecisionRepository:
    _by_correlation: dict[tuple[str, str], AutonomyEvaluationResult] = field(default_factory=dict)

    def append_evaluation(self, result: AutonomyEvaluationResult) -> AutonomyEvaluationResult:
        key = (
            result.audit_bundle.tenant_id,
            result.recommendation_correlation_id,
        )
        self._by_correlation[key] = result
        return result

    def get_latest_evaluation(
        self,
        query: AutonomyDecisionCorrelationQuery,
    ) -> AutonomyEvaluationResult | None:
        key = (query.tenant_id, query.recommendation_correlation_id)
        return self._by_correlation.get(key)


__all__ = ["InMemoryAutonomyDecisionRepository"]
