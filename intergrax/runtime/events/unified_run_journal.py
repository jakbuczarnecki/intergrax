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
from intergrax.runtime.events.execution_position import AsOfBoundary, PositionedRuntimeEvent
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun

JOURNAL_SCHEMA_VERSION = "unified_run_journal.v1"


class PositionedJournalPrefixTruncatedError(Exception):
    """Raised when a positioned journal prefix read hits the configured limit."""


def load_positioned_run_journal_through(
    runtime_store: RuntimeEventPersistence,
    *,
    tenant_id: str,
    boundary: AsOfBoundary,
    initial_limit: int = 1000,
    max_limit: int = 1_000_000,
) -> tuple[PositionedRuntimeEvent, ...]:
    """
    Return the complete positioned prefix for ``boundary`` via the canonical store read path.

    Reads paginate by increasing ``limit`` until the inclusive prefix is complete or
    ``max_limit`` is exceeded. A truncated read fails closed.
    """
    _require_tenant_id(tenant_id)
    _validate_journal_limit(initial_limit)
    _validate_journal_limit(max_limit)
    if initial_limit > max_limit:
        raise ValueError("initial_limit must be <= max_limit")

    limit = initial_limit
    while True:
        batch = tuple(
            runtime_store.list_positioned_through(
                boundary,
                tenant_id=tenant_id,
                limit=limit,
            )
        )
        if not batch:
            return batch
        last_position = batch[-1].position
        if len(batch) < limit:
            return batch
        if last_position >= boundary.position:
            return batch
        if limit >= max_limit:
            raise PositionedJournalPrefixTruncatedError(
                f"execution history prefix for run {boundary.run_id!r} through position "
                f"{boundary.position.value} exceeds max_limit={max_limit}"
            )
        limit = min(limit * 2, max_limit)


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


