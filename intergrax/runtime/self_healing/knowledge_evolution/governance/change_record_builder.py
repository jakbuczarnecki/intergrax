# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Build governance change records from evolution revisions (R5.6)."""

from __future__ import annotations

from intergrax.contracts.self_healing.knowledge_evolution.governance.change_record import (
    StrategyKnowledgeChangeRecord,
    StrategyKnowledgeChangeSource,
    StrategyKnowledgeChangeType,
    mint_strategy_knowledge_change_id,
)
from intergrax.contracts.self_healing.knowledge_evolution.profile import (
    StrategyKnowledgeEvolutionTrigger,
    StrategyKnowledgeRevision,
)

_TRIGGER_TO_SOURCE: dict[StrategyKnowledgeEvolutionTrigger, StrategyKnowledgeChangeSource] = {
    StrategyKnowledgeEvolutionTrigger.WORKFLOW_COMPLETED: StrategyKnowledgeChangeSource.WORKFLOW_COMPLETED,
    StrategyKnowledgeEvolutionTrigger.SCHEDULED_REBUILD: StrategyKnowledgeChangeSource.SCHEDULED_REBUILD,
    StrategyKnowledgeEvolutionTrigger.OPERATOR_REQUEST: StrategyKnowledgeChangeSource.OPERATOR_REQUEST,
    StrategyKnowledgeEvolutionTrigger.BACKFILL: StrategyKnowledgeChangeSource.BACKFILL,
}


def build_change_record_from_revision(revision: StrategyKnowledgeRevision) -> StrategyKnowledgeChangeRecord:
    profile = revision.profile
    change_type = (
        StrategyKnowledgeChangeType.INITIAL_PROFILE
        if revision.previous_knowledge_version is None
        else StrategyKnowledgeChangeType.VERSION_INCREMENT
    )
    change_source = _TRIGGER_TO_SOURCE.get(revision.trigger, StrategyKnowledgeChangeSource.LEARNING_ENGINE)
    return StrategyKnowledgeChangeRecord(
        change_id=mint_strategy_knowledge_change_id(),
        tenant_id=profile.tenant_id,
        strategy_id=profile.strategy_id,
        context_fingerprint=profile.context_fingerprint,
        revision_id=revision.revision_id,
        previous_knowledge_version=revision.previous_knowledge_version,
        new_knowledge_version=profile.knowledge_version,
        change_source=change_source,
        change_type=change_type,
        evolution_mechanism_id=revision.evolution_mechanism_id,
        source_experience_refs=revision.input_experience_ids,
        rationale=revision.change_summary,
        recorded_at=revision.recorded_at,
    )


__all__ = ["build_change_record_from_revision"]
