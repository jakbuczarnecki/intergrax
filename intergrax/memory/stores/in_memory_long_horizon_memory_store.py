# © Artur Czarnecki. All rights reserved.

"""Default in-process long-horizon memory store (MEM-ENT-9)."""

from __future__ import annotations

from dataclasses import replace

from intergrax.memory.contracts.long_horizon_memory import (
    LongHorizonMemoryScope,
    LongHorizonMemoryViolation,
    LongHorizonRecallQuery,
    LongHorizonSummaryRecord,
    SummaryStatus,
    order_long_horizon_summaries_deterministic,
)
from intergrax.memory.contracts.enterprise_memory_record import parse_memory_record_timestamp as _parse_ts
from intergrax.memory.contracts.temporal_chronology import memory_timestamps_same_awareness


def _scope_user_key(scope: LongHorizonMemoryScope) -> str:
    return (scope.user_id or "").strip()


def _scope_workspace_key(scope: LongHorizonMemoryScope) -> str:
    if scope.workspace_id is None:
        return ""
    stripped = scope.workspace_id.strip()
    if not stripped:
        raise LongHorizonMemoryViolation(
            "workspace_id when set must be non-empty for long-horizon scope"
        )
    return stripped


def _storage_key(scope: LongHorizonMemoryScope, summary_id: str) -> tuple[str, str, str, str]:
    tenant = (scope.tenant_id or "").strip()
    if not tenant:
        raise LongHorizonMemoryViolation("tenant_id must be non-empty")
    return (tenant, _scope_user_key(scope), _scope_workspace_key(scope), summary_id.strip())


def _record_equal(existing: LongHorizonSummaryRecord, incoming: LongHorizonSummaryRecord) -> bool:
    return existing == incoming


def _resolve_upsert(
    existing: LongHorizonSummaryRecord | None,
    record: LongHorizonSummaryRecord,
) -> LongHorizonSummaryRecord:
    if existing is None:
        return record
    if record.revision < existing.revision:
        raise LongHorizonMemoryViolation(
            "incoming summary revision is older than stored revision"
        )
    if record.revision == existing.revision:
        if _record_equal(existing, record):
            return existing
        raise LongHorizonMemoryViolation(
            "conflicting long-horizon summary payload for the same revision"
        )
    return record


def _matches_query(record: LongHorizonSummaryRecord, query: LongHorizonRecallQuery) -> bool:
    if query.levels and record.summary_level not in query.levels:
        return False
    if record.status not in query.statuses:
        return False
    if query.covered_from and record.covered_until:
        query_from = _parse_ts("covered_from", query.covered_from)
        record_until = _parse_ts("covered_until", record.covered_until)
        if not memory_timestamps_same_awareness(query_from, record_until):
            return False
        if record_until < query_from:
            return False
    if query.covered_until and record.covered_from:
        query_until = _parse_ts("covered_until", query.covered_until)
        record_from = _parse_ts("covered_from", record.covered_from)
        if not memory_timestamps_same_awareness(query_until, record_from):
            return False
        if record_from > query_until:
            return False
    return True


class InMemoryLongHorizonMemoryStore:
    """Vendor-neutral in-memory ``LongHorizonMemoryStore``."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str, str], LongHorizonSummaryRecord] = {}

    def upsert_summary(
        self,
        scope: LongHorizonMemoryScope,
        record: LongHorizonSummaryRecord,
    ) -> LongHorizonSummaryRecord:
        key = _storage_key(scope, record.summary_id)
        existing = self._records.get(key)
        merged = _resolve_upsert(existing, record)
        if existing is not None and merged is existing:
            return existing
        self._records[key] = merged
        return merged

    def get_summary(
        self,
        scope: LongHorizonMemoryScope,
        summary_id: str,
    ) -> LongHorizonSummaryRecord | None:
        return self._records.get(_storage_key(scope, summary_id))

    def query_summaries(
        self,
        scope: LongHorizonMemoryScope,
        query: LongHorizonRecallQuery,
    ) -> tuple[LongHorizonSummaryRecord, ...]:
        tenant = (scope.tenant_id or "").strip()
        user_key = _scope_user_key(scope)
        workspace_key = _scope_workspace_key(scope)
        matched: list[LongHorizonSummaryRecord] = []
        for key, record in self._records.items():
            key_tenant, key_user, key_workspace, _sid = key
            if key_tenant != tenant:
                continue
            if key_user != user_key:
                continue
            if key_workspace != workspace_key:
                continue
            if _matches_query(record, query):
                matched.append(record)
        ordered = order_long_horizon_summaries_deterministic(tuple(matched))
        return ordered[: query.limit]

    def mark_summary_stale(
        self,
        scope: LongHorizonMemoryScope,
        summary_id: str,
    ) -> LongHorizonSummaryRecord | None:
        key = _storage_key(scope, summary_id)
        existing = self._records.get(key)
        if existing is None:
            return None
        updated = replace(existing, status=SummaryStatus.STALE)
        self._records[key] = updated
        return updated

    def invalidate_summary(
        self,
        scope: LongHorizonMemoryScope,
        summary_id: str,
    ) -> LongHorizonSummaryRecord | None:
        key = _storage_key(scope, summary_id)
        existing = self._records.get(key)
        if existing is None:
            return None
        updated = replace(existing, status=SummaryStatus.INVALIDATED)
        self._records[key] = updated
        return updated
