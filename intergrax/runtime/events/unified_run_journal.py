# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Unified run journal — derived chronological ``RuntimeEvent`` read model (§42.24).

Canonical execution truth is ``RuntimeEventPersistence.list_for_run``.
This module does not own identity, does not mint identity, and does not
reconstruct identity from Plane B trace tags, payload, or active ContextVar.
"""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.execution_identity import validate_run_id
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun

JOURNAL_SCHEMA_VERSION = "unified_run_journal.v1"


def build_unified_run_journal(
    persisted: PersistedRun,
    *,
    runtime_store: RuntimeEventPersistence,
    limit: int = 2000,
) -> list[RuntimeEvent]:
    """
    Return the canonical journal for one persisted run.

    Identity is read from already-canonical ``RuntimeEvent`` records.
    Plane B ``TraceEvent`` rows on ``PersistedRun`` are not converted here.
    """
    _validate_journal_limit(limit)
    tenant_id = _require_tenant_id(persisted.metadata.tenant_id)
    run_id = validate_run_id(persisted.metadata.run_id)
    stored = runtime_store.list_for_run(run_id, tenant_id=tenant_id, limit=limit)
    return _sort_journal(list(stored))[:limit]


def _validate_journal_limit(limit: int) -> None:
    if type(limit) is not int or isinstance(limit, bool) or limit <= 0:
        raise ValueError("journal limit must be > 0")


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise TypeError(f"tenant_id must be str, got {type(tenant_id).__name__}")
    if not tenant_id.strip():
        raise ValueError("tenant_id is required")
    return tenant_id


def _sort_journal(events: list[RuntimeEvent]) -> list[RuntimeEvent]:
    return sorted(events, key=_journal_sort_key)


def _journal_sort_key(event: RuntimeEvent) -> tuple[datetime, int, str]:
    return (event.timestamp, _trace_seq(event), event.event_id)


def _trace_seq(event: RuntimeEvent) -> int:
    raw_seq = event.payload.get("trace_seq")
    if type(raw_seq) is int:
        return raw_seq
    return 0
