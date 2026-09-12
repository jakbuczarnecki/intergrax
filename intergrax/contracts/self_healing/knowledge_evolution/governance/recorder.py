# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Knowledge evolution governance hook — audit after durable revision (SELF-HEALING R5.6)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.knowledge_evolution.profile import StrategyKnowledgeRevision
from intergrax.contracts.self_healing.knowledge_evolution.repository import StrategyKnowledgeRepository


@runtime_checkable
class StrategyKnowledgeEvolutionGovernanceRecorder(Protocol):
    def record_knowledge_evolution(
        self,
        revision: StrategyKnowledgeRevision,
        *,
        knowledge_repository: StrategyKnowledgeRepository,
    ) -> object:
        ...


__all__ = ["StrategyKnowledgeEvolutionGovernanceRecorder"]
