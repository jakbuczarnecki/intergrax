# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Maps workflow completion events to evolution context (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.knowledge_evolution.events import SelfHealingWorkflowCompleted
from intergrax.contracts.self_healing.knowledge_evolution.evolution import StrategyKnowledgeEvolutionContext
from intergrax.contracts.self_healing.knowledge_evolution.profile import (
    StrategyKnowledgeContext,
    StrategyKnowledgeEvolutionTrigger,
)


@dataclass(frozen=True, slots=True)
class WorkflowCompletedKnowledgeEvolutionContextBuilder:
    def build_from_workflow_completed(
        self,
        event: SelfHealingWorkflowCompleted,
    ) -> StrategyKnowledgeEvolutionContext:
        knowledge_context = StrategyKnowledgeContext(
            tenant_id=event.tenant_id,
            strategy_id=event.strategy_id,
            context_fingerprint=event.context_fingerprint,
            context_refs=event.context_refs,
            diagnostic_investigation_id=event.diagnostic_investigation_id,
        )
        trigger_refs = (
            event.workflow_id,
            event.plan_id,
            *event.experience_ids,
        )
        return StrategyKnowledgeEvolutionContext(
            knowledge_context=knowledge_context,
            trigger=StrategyKnowledgeEvolutionTrigger.WORKFLOW_COMPLETED,
            trigger_refs=trigger_refs,
        )


__all__ = ["WorkflowCompletedKnowledgeEvolutionContextBuilder"]
