# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Knowledge audit persistence port — separate from StrategyKnowledgeRepository (R5.6)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.knowledge_evolution.events import StrategyKnowledgeUpdated
from intergrax.contracts.self_healing.knowledge_evolution.governance.audit_query import (
    StrategyKnowledgeChangeRecordQuery,
    StrategyKnowledgeUpdatedEventQuery,
)
from intergrax.contracts.self_healing.knowledge_evolution.governance.change_record import (
    StrategyKnowledgeChangeRecord,
)


@runtime_checkable
class StrategyKnowledgeAuditRepository(Protocol):
    def append_change_record(
        self,
        record: StrategyKnowledgeChangeRecord,
    ) -> StrategyKnowledgeChangeRecord: ...

    def list_change_records(
        self,
        criteria: StrategyKnowledgeChangeRecordQuery,
    ) -> tuple[StrategyKnowledgeChangeRecord, ...]: ...

    def append_knowledge_updated_event(
        self,
        event: StrategyKnowledgeUpdated,
    ) -> StrategyKnowledgeUpdated: ...

    def list_knowledge_updated_events(
        self,
        criteria: StrategyKnowledgeUpdatedEventQuery,
    ) -> tuple[StrategyKnowledgeUpdated, ...]: ...


__all__ = ["StrategyKnowledgeAuditRepository"]
