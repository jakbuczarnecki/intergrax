# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy knowledge persistence port (SELF-HEALING R5.4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.knowledge_evolution.profile import (
    StrategyKnowledgeProfile,
    StrategyKnowledgeRevision,
)
from intergrax.contracts.self_healing.knowledge_evolution.query import (
    StrategyKnowledgeProfileQuery,
    StrategyKnowledgeRevisionQuery,
    StrategyKnowledgeVersionQuery,
)


@runtime_checkable
class StrategyKnowledgeRepository(Protocol):
    def append_revision(self, revision: StrategyKnowledgeRevision) -> StrategyKnowledgeRevision: ...

    def get_latest_profile(
        self,
        criteria: StrategyKnowledgeProfileQuery,
    ) -> StrategyKnowledgeProfile | None: ...

    def get_profile_version(
        self,
        criteria: StrategyKnowledgeVersionQuery,
    ) -> StrategyKnowledgeProfile | None: ...

    def list_revisions(
        self,
        criteria: StrategyKnowledgeRevisionQuery,
    ) -> tuple[StrategyKnowledgeRevision, ...]: ...


__all__ = ["StrategyKnowledgeRepository"]
