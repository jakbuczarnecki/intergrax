# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Unified run journal — derived execution-position ``RuntimeEvent`` read model (§42.24).

Canonical execution truth is ``RuntimeEventPersistence.list_positioned_for_run``.
This module does not own identity, does not mint identity, and does not
reconstruct identity from Plane B trace tags, payload, or active ContextVar.
"""

from __future__ import annotations

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
    positioned = runtime_store.list_positioned_for_run(
        run_id,
        tenant_id=tenant_id,
        limit=limit,
    )
    return [row.event for row in positioned]


def _validate_journal_limit(limit: int) -> None:
    if type(limit) is not int or isinstance(limit, bool) or limit <= 0:
        raise ValueError("journal limit must be > 0")


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise TypeError(f"tenant_id must be str, got {type(tenant_id).__name__}")
    if not tenant_id.strip():
        raise ValueError("tenant_id is required")
    return tenant_id


