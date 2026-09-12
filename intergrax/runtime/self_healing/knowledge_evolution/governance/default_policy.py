# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Default knowledge governance — record-only controls (SELF-HEALING R5.6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.knowledge_evolution.governance.change_record import (
    StrategyKnowledgeChangeRecord,
)
from intergrax.contracts.self_healing.knowledge_evolution.governance.policy import (
    StrategyKnowledgeGovernanceAssessment,
    StrategyKnowledgeGovernanceControlLevel,
)
from intergrax.contracts.self_healing.knowledge_evolution.profile import StrategyKnowledgeRevision


@dataclass(frozen=True, slots=True)
class DefaultKnowledgeGovernancePolicy:
    _policy_id: str = "platform.default_knowledge_governance"

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def assess_change(
        self,
        change: StrategyKnowledgeChangeRecord,
        revision: StrategyKnowledgeRevision,
    ) -> StrategyKnowledgeGovernanceAssessment:
        _ = revision
        return StrategyKnowledgeGovernanceAssessment(
            policy_id=self.policy_id,
            control_level=StrategyKnowledgeGovernanceControlLevel.RECORD_ONLY,
            audit_tags=("knowledge_evolution", change.change_type.value),
            rationale=(
                f"Recorded knowledge transition to version {change.new_knowledge_version} "
                f"via {change.evolution_mechanism_id}."
            ),
        )


__all__ = ["DefaultKnowledgeGovernancePolicy"]
