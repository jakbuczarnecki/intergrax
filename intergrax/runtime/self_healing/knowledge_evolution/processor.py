# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Asynchronous knowledge evolution processor (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.knowledge_evolution.events import SelfHealingWorkflowCompleted
from intergrax.contracts.self_healing.knowledge_evolution.profile import StrategyKnowledgeRevision
from intergrax.runtime.self_healing.knowledge_evolution.service import StrategyKnowledgeEvolutionService
from intergrax.runtime.self_healing.knowledge_evolution.workflow_context_builder import (
    WorkflowCompletedKnowledgeEvolutionContextBuilder,
)


@dataclass(frozen=True, slots=True)
class WorkflowCompletedKnowledgeEvolutionProcessor:
    evolution_service: StrategyKnowledgeEvolutionService
    context_builder: WorkflowCompletedKnowledgeEvolutionContextBuilder

    def process_workflow_completed(
        self,
        event: SelfHealingWorkflowCompleted,
    ) -> StrategyKnowledgeRevision | None:
        context = self.context_builder.build_from_workflow_completed(event)
        result = self.evolution_service.evolve(context)
        if result.no_change:
            return None
        return result.proposed_revision


__all__ = ["WorkflowCompletedKnowledgeEvolutionProcessor"]
