# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Knowledge governance policy port — controls audit metadata, not execution (R5.6)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.knowledge_evolution.governance.change_record import (
    StrategyKnowledgeChangeRecord,
)
from intergrax.contracts.self_healing.knowledge_evolution.profile import StrategyKnowledgeRevision


class StrategyKnowledgeGovernanceControlLevel(StrEnum):
    """Descriptive control tier for audit — not an approval gate in R5.6."""

    RECORD_ONLY = "RECORD_ONLY"
    ENHANCED_AUDIT = "ENHANCED_AUDIT"


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeGovernanceAssessment:
    """Outcome of policy evaluation — informational for audit trails only."""

    policy_id: str
    control_level: StrategyKnowledgeGovernanceControlLevel
    audit_tags: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id required")
        if not self.rationale.strip():
            raise ValueError("rationale required")


@runtime_checkable
class StrategyKnowledgeGovernancePolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def assess_change(
        self,
        change: StrategyKnowledgeChangeRecord,
        revision: StrategyKnowledgeRevision,
    ) -> StrategyKnowledgeGovernanceAssessment:
        """Describe how this change is governed — must not trigger execution or auto-rollback."""
        ...


__all__ = [
    "StrategyKnowledgeGovernanceAssessment",
    "StrategyKnowledgeGovernanceControlLevel",
    "StrategyKnowledgeGovernancePolicy",
]
