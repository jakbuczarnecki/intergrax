# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Knowledge evolution inputs and outputs (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.knowledge_evolution.contextual.operating_context import (
    StrategyKnowledgeOperatingContext,
)
from intergrax.contracts.self_healing.knowledge_evolution.profile import (
    StrategyKnowledgeContext,
    StrategyKnowledgeEvolutionTrigger,
    StrategyKnowledgeProfile,
    StrategyKnowledgeRevision,
)
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeEvolutionContext:
    knowledge_context: StrategyKnowledgeContext
    trigger: StrategyKnowledgeEvolutionTrigger
    trigger_refs: tuple[str, ...]
    optional_quality_assessment: StrategyQualityAssessment | None = None
    comparison_strategy_ids: tuple[str, ...] = ()
    resolved_operating_context: StrategyKnowledgeOperatingContext | None = None
    freshness_policy_id: str | None = None

    def __post_init__(self) -> None:
        if not self.trigger_refs:
            raise ValueError("trigger_refs must be non-empty")


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeEvolutionRevisionMetadata:
    change_summary: str
    input_experience_ids: tuple[str, ...]
    input_assessment_refs: tuple[str, ...]
    metric_snapshot_refs: tuple[str, ...]
    comparison_policy_id: str | None
    previous_knowledge_version: int | None

    def __post_init__(self) -> None:
        if not self.change_summary.strip():
            raise ValueError("change_summary required")


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeEvolutionResult:
    proposed_profile: StrategyKnowledgeProfile | None
    revision_metadata: StrategyKnowledgeEvolutionRevisionMetadata | None
    no_change: bool
    proposed_revision: StrategyKnowledgeRevision | None = None

    def __post_init__(self) -> None:
        if self.no_change:
            if self.proposed_profile is not None or self.proposed_revision is not None:
                raise ValueError("no_change result must not include profile or revision")
            return
        if self.proposed_profile is None or self.revision_metadata is None:
            raise ValueError("evolution with changes requires profile and revision_metadata")


__all__ = [
    "StrategyKnowledgeEvolutionContext",
    "StrategyKnowledgeEvolutionRevisionMetadata",
    "StrategyKnowledgeEvolutionResult",
]
