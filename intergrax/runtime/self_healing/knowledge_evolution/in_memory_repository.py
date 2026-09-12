# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory strategy knowledge store (SELF-HEALING R5.4 test double)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.self_healing.knowledge_evolution.profile import (
    StrategyKnowledgeProfile,
    StrategyKnowledgeRevision,
)
from intergrax.contracts.self_healing.knowledge_evolution.query import (
    StrategyKnowledgeProfileQuery,
    StrategyKnowledgeRevisionQuery,
    StrategyKnowledgeVersionQuery,
)


def _scope_key(tenant_id: str, strategy_id: str, context_fingerprint: str) -> tuple[str, str, str]:
    return (tenant_id, strategy_id, context_fingerprint)


@dataclass
class InMemoryStrategyKnowledgeRepository:
    _revisions: list[StrategyKnowledgeRevision] = field(default_factory=list)

    def append_revision(self, revision: StrategyKnowledgeRevision) -> StrategyKnowledgeRevision:
        for existing in self._revisions:
            if existing.revision_id == revision.revision_id:
                return existing
        self._revisions.append(revision)
        return revision

    def get_latest_profile(
        self,
        criteria: StrategyKnowledgeProfileQuery,
    ) -> StrategyKnowledgeProfile | None:
        key = _scope_key(criteria.tenant_id, criteria.strategy_id, criteria.context_fingerprint)
        latest: StrategyKnowledgeProfile | None = None
        for revision in self._revisions:
            revision_key = _scope_key(
                revision.profile.tenant_id,
                revision.profile.strategy_id,
                revision.profile.context_fingerprint,
            )
            if revision_key != key:
                continue
            if latest is None or revision.profile.knowledge_version > latest.knowledge_version:
                latest = revision.profile
        return latest

    def get_profile_version(
        self,
        criteria: StrategyKnowledgeVersionQuery,
    ) -> StrategyKnowledgeProfile | None:
        key = _scope_key(criteria.tenant_id, criteria.strategy_id, criteria.context_fingerprint)
        for revision in self._revisions:
            revision_key = _scope_key(
                revision.profile.tenant_id,
                revision.profile.strategy_id,
                revision.profile.context_fingerprint,
            )
            if revision_key != key:
                continue
            if revision.profile.knowledge_version == criteria.knowledge_version:
                return revision.profile
        return None

    def list_revisions(
        self,
        criteria: StrategyKnowledgeRevisionQuery,
    ) -> tuple[StrategyKnowledgeRevision, ...]:
        key = _scope_key(criteria.tenant_id, criteria.strategy_id, criteria.context_fingerprint)
        rows = [
            revision
            for revision in self._revisions
            if _scope_key(
                revision.profile.tenant_id,
                revision.profile.strategy_id,
                revision.profile.context_fingerprint,
            )
            == key
        ]
        rows.sort(key=lambda row: row.profile.knowledge_version)
        if len(rows) > criteria.limit:
            rows = rows[-criteria.limit :]
        return tuple(rows)


__all__ = ["InMemoryStrategyKnowledgeRepository"]
