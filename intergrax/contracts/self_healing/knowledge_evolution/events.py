# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Event-driven knowledge evolution hooks — not wired to workflow (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.knowledge_evolution.evolution import StrategyKnowledgeEvolutionContext
from intergrax.contracts.self_healing.knowledge_evolution.profile import StrategyKnowledgeRevision


@dataclass(frozen=True, slots=True)
class SelfHealingWorkflowCompleted:
    """
    Domain event shape for asynchronous knowledge evolution.

    Emission and delivery are out of scope for R5.4 foundation; consumers must be idempotent.
    """

    tenant_id: str
    workflow_id: str
    strategy_id: str
    plan_id: str
    diagnostic_investigation_id: str
    context_fingerprint: str
    context_refs: tuple[str, ...]
    experience_ids: tuple[str, ...]
    completed_at: datetime

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.workflow_id.startswith("sh_wf_"):
            raise ValueError("workflow_id must be sh_wf_*")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.plan_id.startswith("sh_plan_"):
            raise ValueError("plan_id must be sh_plan_*")
        if not self.diagnostic_investigation_id.strip():
            raise ValueError("diagnostic_investigation_id required")
        if not self.context_fingerprint.strip():
            raise ValueError("context_fingerprint required")
        if not self.context_refs:
            raise ValueError("context_refs must be non-empty")
        if not self.experience_ids:
            raise ValueError("experience_ids must be non-empty")


@runtime_checkable
class KnowledgeEvolutionProcessor(Protocol):
    def process_workflow_completed(
        self,
        event: SelfHealingWorkflowCompleted,
    ) -> StrategyKnowledgeRevision | None:
        """
        Idempotent consumer — duplicate events must not create conflicting revisions.

        Returns None when evolution is skipped (no change).
        """
        ...


@runtime_checkable
class KnowledgeEvolutionContextBuilder(Protocol):
    def build_from_workflow_completed(
        self,
        event: SelfHealingWorkflowCompleted,
    ) -> StrategyKnowledgeEvolutionContext: ...


__all__ = [
    "KnowledgeEvolutionContextBuilder",
    "KnowledgeEvolutionProcessor",
    "SelfHealingWorkflowCompleted",
]
