# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory autonomy decision store for tests and local bootstrap (R6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.self_healing.autonomy.decision import AutonomyControlDecision
from intergrax.contracts.self_healing.autonomy.repository import AutonomyDecisionCorrelationQuery


@dataclass
class InMemoryAutonomyRepository:
    _by_correlation: dict[tuple[str, str], AutonomyControlDecision] = field(default_factory=dict)

    def append_decision(self, decision: AutonomyControlDecision) -> AutonomyControlDecision:
        key = (decision.audit_bundle.tenant_id, decision.recommendation_correlation_id)
        self._by_correlation[key] = decision
        return decision

    def get_latest_by_correlation(
        self,
        query: AutonomyDecisionCorrelationQuery,
    ) -> AutonomyControlDecision | None:
        return self._by_correlation.get((query.tenant_id, query.recommendation_correlation_id))


__all__ = ["InMemoryAutonomyRepository"]
