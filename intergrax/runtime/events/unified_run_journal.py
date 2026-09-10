# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Unified run journal — derived execution-position ``RuntimeEvent`` read model (§42.24).

Canonical execution truth is ``RuntimeEventPersistence.list_positioned_for_run``.
This module does not own identity, does not mint identity, and does not
reconstruct identity from Plane B trace tags, payload, or active ContextVar.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.runtime.events.execution_position import (
    AsOfBoundary,
    ExecutionEventPosition,
    PositionedRuntimeEvent,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun

JOURNAL_SCHEMA_VERSION = "unified_run_journal.v1"
JOURNAL_READ_DEFAULT_MAX_EVENTS = 2_000_000
JOURNAL_READ_DEFAULT_PAGE_SIZE = 2000
JOURNAL_SNAPSHOT_PROBE_PAGE_SIZE = 1000


class PositionedJournalPrefixTruncatedError(Exception):
    """Raised when a positioned journal prefix read hits the configured limit."""


class PositionedJournalBoundaryNotFoundError(Exception):
    """Raised when canonical history has no accepted event at the requested boundary position."""


class JournalReadLimitExceededError(Exception):
    """Raised when a complete journal read exceeds ``max_events``."""


class JournalReadIncompleteError(Exception):
    """Raised when a complete journal read cannot prove exhaustiveness."""


class JournalCursorScopeMismatchError(Exception):
    """Raised when a continuation cursor does not match the requested tenant/run scope."""


@dataclass(frozen=True, slots=True)
class RunJournalSnapshotBoundary:
    """Inclusive terminal execution position frozen at the start of a paginated read."""

    run_id: RunId
    terminal_position: ExecutionEventPosition | None


@dataclass(frozen=True, slots=True)
class RunJournalContinuationCursor:
    """Position-based continuation scoped to one tenant + run stream."""

    tenant_id: str
    run_id: RunId
    exclusive_after: ExecutionEventPosition
    snapshot_through: ExecutionEventPosition | None


@dataclass(frozen=True, slots=True)
class RunJournalReadPage:
    """Bounded journal page with explicit completeness within the frozen snapshot."""

    events: tuple[RuntimeEvent, ...]
    is_complete: bool
    next_cursor: RunJournalContinuationCursor | None


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

    Ownership: this helper is the single authority for prefix completeness and exact
    boundary existence. It returns a prefix whose last event position equals
    ``boundary.position``, or raises when history is absent, the exact boundary event
    is missing, or completeness cannot be proven within ``max_limit``.
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
            if last_position < boundary.position:
                raise PositionedJournalBoundaryNotFoundError(
                    f"no accepted execution event at position {boundary.position.value} "
                    f"for run {boundary.run_id!r}"
                )
            return batch
        if last_position >= boundary.position:
            if last_position != boundary.position:
                raise PositionedJournalBoundaryNotFoundError(
                    f"positioned prefix ends at {last_position.value}, "
                    f"not at requested boundary {boundary.position.value} "
                    f"for run {boundary.run_id!r}"
                )
            return batch
        if limit >= max_limit:
            raise PositionedJournalPrefixTruncatedError(
                f"execution history prefix for run {boundary.run_id!r} through position "
                f"{boundary.position.value} exceeds max_limit={max_limit}"
            )
        limit = min(limit * 2, max_limit)


def read_run_journal_page(
    runtime_store: RuntimeEventPersistence,
    *,
    tenant_id: str,
    run_id: str,
    page_size: int,
    cursor: RunJournalContinuationCursor | None = None,
) -> RunJournalReadPage:
    """
    Return one bounded journal page with explicit completeness within a snapshot boundary.

    Snapshot semantics: the inclusive terminal position is frozen on the first page of a
    sequence (``cursor is None``). Events appended after that snapshot are excluded from
    later pages in the same sequence.
    """
    scope_tenant = _require_tenant_id(tenant_id)
    scope_run = validate_run_id(run_id)
    _validate_journal_limit(page_size)

    if cursor is None:
        snapshot = _resolve_run_journal_snapshot_boundary(
            runtime_store,
            tenant_id=scope_tenant,
            run_id=scope_run,
        )
        exclusive_after = None
        snapshot_through = snapshot.terminal_position
    else:
        _validate_cursor_scope(
            cursor,
            tenant_id=scope_tenant,
            run_id=scope_run,
        )
        exclusive_after = cursor.exclusive_after
        snapshot_through = cursor.snapshot_through

    positioned = runtime_store.list_positioned_for_run(
        scope_run,
        tenant_id=scope_tenant,
        limit=page_size + 1,
        through=snapshot_through,
        after=exclusive_after,
    )
    page_rows = tuple(positioned[:page_size])
    events = tuple(row.event for row in page_rows)

    if not page_rows:
        return RunJournalReadPage(events=(), is_complete=True, next_cursor=None)

    has_more_in_snapshot = len(positioned) > page_size
    if not has_more_in_snapshot:
        return RunJournalReadPage(events=events, is_complete=True, next_cursor=None)

    last_row = page_rows[-1]
    next_cursor = RunJournalContinuationCursor(
        tenant_id=scope_tenant,
        run_id=scope_run,
        exclusive_after=last_row.position,
        snapshot_through=snapshot_through,
    )
    return RunJournalReadPage(
        events=events,
        is_complete=False,
        next_cursor=next_cursor,
    )


def load_complete_run_journal(
    runtime_store: RuntimeEventPersistence,
    *,
    tenant_id: str,
    run_id: str,
    max_events: int = JOURNAL_READ_DEFAULT_MAX_EVENTS,
    page_size: int = JOURNAL_READ_DEFAULT_PAGE_SIZE,
) -> tuple[RuntimeEvent, ...]:
    """Load the full run journal or fail closed when bounds are exceeded."""
    _validate_journal_limit(max_events)
    _validate_journal_limit(page_size)

    collected: list[RuntimeEvent] = []
    cursor: RunJournalContinuationCursor | None = None
    while True:
        page = read_run_journal_page(
            runtime_store,
            tenant_id=tenant_id,
            run_id=run_id,
            page_size=page_size,
            cursor=cursor,
        )
        collected.extend(page.events)
        if len(collected) > max_events:
            raise JournalReadLimitExceededError(
                f"run journal for {run_id!r} exceeds max_events={max_events}"
            )
        if page.is_complete:
            return tuple(collected)
        if page.next_cursor is None:
            raise JournalReadIncompleteError(
                f"run journal for {run_id!r} ended without completeness proof"
            )
        cursor = page.next_cursor


def build_unified_run_journal(
    persisted: PersistedRun,
    *,
    runtime_store: RuntimeEventPersistence,
    max_events: int = JOURNAL_READ_DEFAULT_MAX_EVENTS,
    page_size: int = JOURNAL_READ_DEFAULT_PAGE_SIZE,
) -> list[RuntimeEvent]:
    """
    Return the complete canonical journal for one persisted run.

    Identity is read from already-canonical ``RuntimeEvent`` records.
    Plane B ``TraceEvent`` rows on ``PersistedRun`` are not converted here.
    """
    tenant_id = _require_tenant_id(persisted.metadata.tenant_id)
    run_id = validate_run_id(persisted.metadata.run_id)
    return list(
        load_complete_run_journal(
            runtime_store,
            tenant_id=tenant_id,
            run_id=run_id,
            max_events=max_events,
            page_size=page_size,
        )
    )


def _resolve_run_journal_snapshot_boundary(
    runtime_store: RuntimeEventPersistence,
    *,
    tenant_id: str,
    run_id: RunId,
) -> RunJournalSnapshotBoundary:
    """Prove the inclusive terminal execution position at read start (snapshot boundary)."""
    exclusive_after: ExecutionEventPosition | None = None
    terminal: ExecutionEventPosition | None = None
    while True:
        batch = runtime_store.list_positioned_for_run(
            run_id,
            tenant_id=tenant_id,
            limit=JOURNAL_SNAPSHOT_PROBE_PAGE_SIZE,
            after=exclusive_after,
        )
        if not batch:
            return RunJournalSnapshotBoundary(run_id=run_id, terminal_position=terminal)
        terminal = batch[-1].position
        if len(batch) < JOURNAL_SNAPSHOT_PROBE_PAGE_SIZE:
            return RunJournalSnapshotBoundary(run_id=run_id, terminal_position=terminal)
        exclusive_after = terminal


def _validate_cursor_scope(
    cursor: RunJournalContinuationCursor,
    *,
    tenant_id: str,
    run_id: RunId,
) -> None:
    if cursor.tenant_id != tenant_id:
        raise JournalCursorScopeMismatchError(
            "journal continuation cursor tenant does not match read scope",
        )
    if cursor.run_id != run_id:
        raise JournalCursorScopeMismatchError(
            "journal continuation cursor run_id does not match read scope",
        )


def _validate_journal_limit(limit: int) -> None:
    if type(limit) is not int or isinstance(limit, bool) or limit <= 0:
        raise ValueError("journal limit must be > 0")


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise TypeError(f"tenant_id must be str, got {type(tenant_id).__name__}")
    if not tenant_id.strip():
        raise ValueError("tenant_id is required")
    return tenant_id
