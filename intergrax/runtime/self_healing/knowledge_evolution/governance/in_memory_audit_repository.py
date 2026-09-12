# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory knowledge audit store (SELF-HEALING R5.6 test double)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.self_healing.knowledge_evolution.events import StrategyKnowledgeUpdated
from intergrax.contracts.self_healing.knowledge_evolution.governance.audit_query import (
    StrategyKnowledgeChangeRecordQuery,
    StrategyKnowledgeUpdatedEventQuery,
)
from intergrax.contracts.self_healing.knowledge_evolution.governance.change_record import (
    StrategyKnowledgeChangeRecord,
)


def _scope_key(tenant_id: str, strategy_id: str, context_fingerprint: str) -> tuple[str, str, str]:
    return (tenant_id, strategy_id, context_fingerprint)


@dataclass
class InMemoryStrategyKnowledgeAuditRepository:
    _change_records: list[StrategyKnowledgeChangeRecord] = field(default_factory=list)
    _updated_events: list[StrategyKnowledgeUpdated] = field(default_factory=list)

    def append_change_record(
        self,
        record: StrategyKnowledgeChangeRecord,
    ) -> StrategyKnowledgeChangeRecord:
        for existing in self._change_records:
            if existing.change_id == record.change_id:
                return existing
        self._change_records.append(record)
        return record

    def list_change_records(
        self,
        criteria: StrategyKnowledgeChangeRecordQuery,
    ) -> tuple[StrategyKnowledgeChangeRecord, ...]:
        key = _scope_key(criteria.tenant_id, criteria.strategy_id, criteria.context_fingerprint)
        rows = [
            row
            for row in self._change_records
            if _scope_key(row.tenant_id, row.strategy_id, row.context_fingerprint) == key
        ]
        rows.sort(key=lambda row: row.new_knowledge_version)
        if len(rows) > criteria.limit:
            rows = rows[-criteria.limit :]
        return tuple(rows)

    def append_knowledge_updated_event(
        self,
        event: StrategyKnowledgeUpdated,
    ) -> StrategyKnowledgeUpdated:
        for existing in self._updated_events:
            if existing.change_id == event.change_id and existing.revision_id == event.revision_id:
                return existing
        self._updated_events.append(event)
        return event

    def list_knowledge_updated_events(
        self,
        criteria: StrategyKnowledgeUpdatedEventQuery,
    ) -> tuple[StrategyKnowledgeUpdated, ...]:
        key = _scope_key(criteria.tenant_id, criteria.strategy_id, criteria.context_fingerprint)
        rows = [
            row
            for row in self._updated_events
            if _scope_key(row.tenant_id, row.strategy_id, row.context_fingerprint) == key
        ]
        rows.sort(key=lambda row: row.new_knowledge_version)
        if len(rows) > criteria.limit:
            rows = rows[-criteria.limit :]
        return tuple(rows)


__all__ = ["InMemoryStrategyKnowledgeAuditRepository"]
