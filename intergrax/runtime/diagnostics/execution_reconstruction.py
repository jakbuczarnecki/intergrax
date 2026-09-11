# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution reconstruction projection (DIAG-2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptDiscoveryRecord,
    ExecutionLineageDiscoveryCoverageOrigin,
    ExecutionLineageDiscoveryRunState,
    ExecutionLineageIntegrityError,
    ExecutionLineageReader,
    ExecutionLineageRunScope,
    ExecutionLineageUnavailableError,
    build_execution_lineage_run_scope,
)
from intergrax.runtime.diagnostics.execution_lineage_reconstruction import (
    ExecutionLineageCompleteness,
    ExecutionLineageReadStatus,
    ExecutionLineageReconstructionIntegrityError,
    ReconstructedAttemptLineage,
    reconstruct_attempt_lineage,
)
from intergrax.runtime.events.execution_position import AsOfBoundary, PositionedRuntimeEvent
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.unified_run_journal import (
    PositionedJournalBoundaryNotFoundError,
    PositionedJournalPrefixTruncatedError,
    load_positioned_run_journal_through,
)
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
    causal_evidence_query_order_key,
)


class ExecutionReconstructionIntegrityError(Exception):
    """Raised when canonical persistence returns facts outside the requested scope."""


class RuntimeHistoryCompleteness(StrEnum):
    """Whether positioned runtime history for the run is complete or truncated."""

    COMPLETE = "complete"
    TRUNCATED = "truncated"


class ExecutionAttemptDiscoveryReadStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ExecutionAttemptDiscoveryCompleteness(StrEnum):
    COMPLETE = "complete"
    LEGACY_UNKNOWN = "legacy_unknown"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class _RunDiscoverySnapshot:
    records: tuple[ExecutionLineageAttemptDiscoveryRecord, ...]
    run_state_before: ExecutionLineageDiscoveryRunState | None
    run_state_after: ExecutionLineageDiscoveryRunState | None
    truncated: bool
    read_status: ExecutionAttemptDiscoveryReadStatus
    completeness: ExecutionAttemptDiscoveryCompleteness | None


@dataclass(frozen=True, slots=True)
class _AttemptBuildResult:
    attempts: tuple[ReconstructedAttempt, ...]
    discovery_read_status: ExecutionAttemptDiscoveryReadStatus | None
    discovery_completeness: ExecutionAttemptDiscoveryCompleteness | None


@dataclass(frozen=True, slots=True)
class ReconstructedAttempt:
    """One attempt within an execution reconstruction — derived, not canonical."""

    attempt_id: AttemptId
    causal_evidence: tuple[PlatformCausalEvidence, ...]
    positioned_events: tuple[PositionedRuntimeEvent, ...]
    lineage: ReconstructedAttemptLineage | None = None

    @property
    def has_transport_evidence(self) -> bool:
        return bool(self.causal_evidence)

    @property
    def has_runtime_events(self) -> bool:
        return bool(self.positioned_events)


@dataclass(frozen=True, slots=True)
class ExecutionReconstruction:
    """
    Derived read model joining runtime execution history and causal evidence.

    NOT persisted and NOT a source of truth.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    causal_evidence: tuple[PlatformCausalEvidence, ...]
    positioned_events: tuple[PositionedRuntimeEvent, ...]
    attempts: tuple[ReconstructedAttempt, ...]
    runtime_history_completeness: RuntimeHistoryCompleteness
    attempt_discovery_read_status: ExecutionAttemptDiscoveryReadStatus | None = None
    attempt_discovery_completeness: ExecutionAttemptDiscoveryCompleteness | None = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def has_transport_evidence(self) -> bool:
        return bool(self.causal_evidence)

    @property
    def has_runtime_events(self) -> bool:
        return bool(self.positioned_events)

    @property
    def is_runtime_history_complete(self) -> bool:
        return self.runtime_history_completeness is RuntimeHistoryCompleteness.COMPLETE

    @property
    def has_lineage_evidence(self) -> bool:
        return any(
            attempt.lineage is not None
            and attempt.lineage.read_status is ExecutionLineageReadStatus.AVAILABLE
            for attempt in self.attempts
        )

    @property
    def has_complete_lineage(self) -> bool:
        return any(
            attempt.lineage is not None
            and attempt.lineage.completeness is ExecutionLineageCompleteness.COMPLETE
            for attempt in self.attempts
        )

    @property
    def has_partial_lineage(self) -> bool:
        return any(
            attempt.lineage is not None
            and attempt.lineage.completeness is ExecutionLineageCompleteness.PARTIAL
            for attempt in self.attempts
        )


class ExecutionReconstructor:
    """
    Platform-owned deterministic reconstruction from canonical persistence only.

    Depends on ``RuntimeEventPersistence`` (execution truth) and
    ``CausalEvidencePersistence`` (relation truth). Optional
    ``ExecutionLineageReader`` enriches forensic parent topology.
    """

    def __init__(
        self,
        runtime_events: RuntimeEventPersistence,
        causal_evidence: CausalEvidencePersistence,
        execution_lineage: ExecutionLineageReader | None = None,
        *,
        initial_lineage_page_limit: int = 100,
        max_lineage_records: int = 10_000,
        initial_attempt_discovery_page_limit: int = 100,
        max_attempt_discovery_records: int = 10_000,
        max_attempt_discovery_snapshot_retries: int = 8,
        max_lineage_snapshot_retries: int = 8,
    ) -> None:
        if max_attempt_discovery_snapshot_retries <= 0:
            raise ValueError("max_attempt_discovery_snapshot_retries must be > 0")
        self._runtime_events = runtime_events
        self._causal_evidence = causal_evidence
        self._execution_lineage = execution_lineage
        self._initial_lineage_page_limit = initial_lineage_page_limit
        self._max_lineage_records = max_lineage_records
        self._initial_attempt_discovery_page_limit = (
            initial_attempt_discovery_page_limit
        )
        self._max_attempt_discovery_records = max_attempt_discovery_records
        self._max_attempt_discovery_snapshot_retries = (
            max_attempt_discovery_snapshot_retries
        )
        self._max_lineage_snapshot_retries = max_lineage_snapshot_retries

    def reconstruct_execution(
        self,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        *,
        execution_as_of: AsOfBoundary | None = None,
        initial_limit: int = 1000,
        max_limit: int = 1_000_000,
    ) -> ExecutionReconstruction:
        tenant_id = _require_tenant_id(tenant_id)
        task_id = validate_task_id(task_id)
        run_id = validate_run_id(run_id)
        if execution_as_of is not None:
            if type(execution_as_of) is not AsOfBoundary:
                raise TypeError("execution_as_of must be AsOfBoundary or None")
            if execution_as_of.run_id != run_id:
                raise ExecutionReconstructionIntegrityError(
                    "execution_as_of.run_id must match reconstruct_execution run_id"
                )
        _validate_history_limit(initial_limit)
        _validate_history_limit(max_limit)
        if initial_limit > max_limit:
            raise ValueError("initial_limit must be <= max_limit")

        causal = self._causal_evidence.list_for_execution(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        )
        for evidence in causal:
            _validate_causal_evidence_scope(
                evidence,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
            )

        if execution_as_of is None:
            positioned, completeness = _load_positioned_events_for_run(
                self._runtime_events,
                tenant_id=tenant_id,
                run_id=run_id,
                initial_limit=initial_limit,
                max_limit=max_limit,
            )
        else:
            positioned, completeness = _load_positioned_events_through_boundary(
                self._runtime_events,
                tenant_id=tenant_id,
                boundary=execution_as_of,
                initial_limit=initial_limit,
                max_limit=max_limit,
            )
        for row in positioned:
            _validate_runtime_event_scope(
                row,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
            )

        discovery_snapshot = _load_run_discovery_snapshot(
            self._execution_lineage,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            page_limit=self._initial_attempt_discovery_page_limit,
            max_records=self._max_attempt_discovery_records,
            max_retries=self._max_attempt_discovery_snapshot_retries,
        )

        attempt_build = _build_attempts(
            causal,
            positioned,
            execution_lineage=self._execution_lineage,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            initial_lineage_page_limit=self._initial_lineage_page_limit,
            max_lineage_records=self._max_lineage_records,
            max_lineage_snapshot_retries=self._max_lineage_snapshot_retries,
            discovery_snapshot=discovery_snapshot,
        )
        discovery_read_status = attempt_build.discovery_read_status
        discovery_completeness = attempt_build.discovery_completeness
        return ExecutionReconstruction(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            causal_evidence=causal,
            positioned_events=positioned,
            attempts=attempt_build.attempts,
            runtime_history_completeness=completeness,
            attempt_discovery_read_status=discovery_read_status,
            attempt_discovery_completeness=discovery_completeness,
        )


def _load_run_discovery_snapshot(
    reader: ExecutionLineageReader | None,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    page_limit: int,
    max_records: int,
    max_retries: int,
) -> _RunDiscoverySnapshot:
    if reader is None:
        return _RunDiscoverySnapshot(
            records=(),
            run_state_before=None,
            run_state_after=None,
            truncated=False,
            read_status=ExecutionAttemptDiscoveryReadStatus.AVAILABLE,
            completeness=None,
        )
    run_scope = build_execution_lineage_run_scope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
    )
    run_state_before: ExecutionLineageDiscoveryRunState | None = None
    run_state_after: ExecutionLineageDiscoveryRunState | None = None
    for _ in range(max_retries):
        try:
            run_state_before = reader.read_discovery_run_state(run_scope)
            records, truncated, next_cursor = _load_discovery_pages(
                reader,
                run_scope,
                page_limit=page_limit,
                max_records=max_records,
            )
            run_state_after = reader.read_discovery_run_state(run_scope)
        except ExecutionLineageUnavailableError:
            return _RunDiscoverySnapshot(
                records=(),
                run_state_before=None,
                run_state_after=None,
                truncated=False,
                read_status=ExecutionAttemptDiscoveryReadStatus.UNAVAILABLE,
                completeness=None,
            )
        except ExecutionLineageIntegrityError as exc:
            raise ExecutionReconstructionIntegrityError(str(exc)) from exc

        if run_state_before is None and run_state_after is None:
            if records:
                raise ExecutionReconstructionIntegrityError(
                    "discovery rows exist without run meta",
                )
            return _RunDiscoverySnapshot(
                records=records,
                run_state_before=None,
                run_state_after=None,
                truncated=truncated,
                read_status=ExecutionAttemptDiscoveryReadStatus.AVAILABLE,
                completeness=ExecutionAttemptDiscoveryCompleteness.LEGACY_UNKNOWN,
            )

        if (
            run_state_before is not None
            and run_state_after is not None
            and run_state_before.generation == run_state_after.generation
        ):
            if not records:
                if _is_legal_empty_from_run_start_run_meta(run_state_after):
                    pass
                elif run_state_after.next_discovery_position > 1:
                    raise ExecutionReconstructionIntegrityError(
                        "empty discovery snapshot with impossible counter",
                    )
                else:
                    raise ExecutionReconstructionIntegrityError(
                        "invalid empty discovery run meta",
                    )
            if truncated or next_cursor is not None:
                completeness = ExecutionAttemptDiscoveryCompleteness.TRUNCATED
            elif (
                run_state_after.coverage_origin
                is ExecutionLineageDiscoveryCoverageOrigin.FROM_RUN_START
                and run_state_after.coverage_contract_version == 1
            ):
                _validate_full_discovery_snapshot(
                    records,
                    run_state=run_state_after,
                )
                completeness = ExecutionAttemptDiscoveryCompleteness.COMPLETE
            else:
                if records:
                    _validate_full_discovery_snapshot(
                        records,
                        run_state=run_state_after,
                    )
                completeness = ExecutionAttemptDiscoveryCompleteness.LEGACY_UNKNOWN
            return _RunDiscoverySnapshot(
                records=records,
                run_state_before=run_state_before,
                run_state_after=run_state_after,
                truncated=truncated or next_cursor is not None,
                read_status=ExecutionAttemptDiscoveryReadStatus.AVAILABLE,
                completeness=completeness,
            )

    return _RunDiscoverySnapshot(
        records=(),
        run_state_before=run_state_before,
        run_state_after=run_state_after,
        truncated=True,
        read_status=ExecutionAttemptDiscoveryReadStatus.AVAILABLE,
        completeness=ExecutionAttemptDiscoveryCompleteness.TRUNCATED,
    )


def _load_discovery_pages(
    reader: ExecutionLineageReader,
    run_scope: ExecutionLineageRunScope,
    *,
    page_limit: int,
    max_records: int,
) -> tuple[tuple[ExecutionLineageAttemptDiscoveryRecord, ...], bool, str | None]:
    collected: list[ExecutionLineageAttemptDiscoveryRecord] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    truncated = False
    next_cursor: str | None = None
    while True:
        if cursor is not None:
            if cursor in seen_cursors:
                raise ExecutionReconstructionIntegrityError(
                    "attempt discovery cursor cycle",
                )
            seen_cursors.add(cursor)
        page = reader.list_attempts_for_run(run_scope, page_limit, cursor=cursor)
        collected.extend(page.attempts)
        if len(collected) > max_records:
            truncated = True
            collected = collected[:max_records]
            next_cursor = page.next_cursor
            break
        next_cursor = page.next_cursor
        if next_cursor is None:
            break
        if next_cursor == cursor:
            raise ExecutionReconstructionIntegrityError(
                "attempt discovery cursor cycle",
            )
        cursor = next_cursor
    return tuple(collected), truncated, next_cursor


def _is_legal_empty_from_run_start_run_meta(
    run_state: ExecutionLineageDiscoveryRunState,
) -> bool:
    return (
        run_state.coverage_origin
        is ExecutionLineageDiscoveryCoverageOrigin.FROM_RUN_START
        and run_state.coverage_contract_version == 1
        and run_state.next_discovery_position == 1
    )


def _effective_attempt_discovery_metadata(
    snapshot: _RunDiscoverySnapshot,
    *,
    discovery_only_candidate_unavailable: bool,
    lineage_configured: bool,
) -> tuple[
    ExecutionAttemptDiscoveryReadStatus | None,
    ExecutionAttemptDiscoveryCompleteness | None,
]:
    if not lineage_configured:
        return None, None
    if discovery_only_candidate_unavailable:
        return ExecutionAttemptDiscoveryReadStatus.UNAVAILABLE, None
    return snapshot.read_status, snapshot.completeness


def _validate_full_discovery_snapshot(
    records: tuple[ExecutionLineageAttemptDiscoveryRecord, ...],
    *,
    run_state: ExecutionLineageDiscoveryRunState,
) -> None:
    attempt_ids: set[AttemptId] = set()
    positions: set[int] = set()
    for record in records:
        if record.attempt_id in attempt_ids:
            raise ExecutionReconstructionIntegrityError(
                "duplicate attempt discovery record",
            )
        if record.discovery_position in positions:
            raise ExecutionReconstructionIntegrityError(
                "duplicate discovery position",
            )
        attempt_ids.add(record.attempt_id)
        positions.add(record.discovery_position)
        if record.discovery_position < 1:
            raise ExecutionReconstructionIntegrityError("invalid discovery position")
    if records:
        expected_positions = set(range(1, len(records) + 1))
        if positions != expected_positions:
            raise ExecutionReconstructionIntegrityError(
                "discovery positions must be contiguous from 1",
            )
        if run_state.next_discovery_position != len(records) + 1:
            raise ExecutionReconstructionIntegrityError(
                "discovery run state counter mismatch",
            )


def _load_positioned_events_for_run(
    runtime_store: RuntimeEventPersistence,
    *,
    tenant_id: str,
    run_id: RunId,
    initial_limit: int,
    max_limit: int,
) -> tuple[tuple[PositionedRuntimeEvent, ...], RuntimeHistoryCompleteness]:
    limit = initial_limit
    while True:
        batch = tuple(
            runtime_store.list_positioned_for_run(
                run_id,
                tenant_id=tenant_id,
                limit=limit,
            )
        )
        if len(batch) < limit:
            return batch, RuntimeHistoryCompleteness.COMPLETE
        if limit >= max_limit:
            return batch, RuntimeHistoryCompleteness.TRUNCATED
        limit = min(limit * 2, max_limit)


def _load_positioned_events_through_boundary(
    runtime_store: RuntimeEventPersistence,
    *,
    tenant_id: str,
    boundary: AsOfBoundary,
    initial_limit: int,
    max_limit: int,
) -> tuple[tuple[PositionedRuntimeEvent, ...], RuntimeHistoryCompleteness]:
    try:
        positioned = load_positioned_run_journal_through(
            runtime_store,
            tenant_id=tenant_id,
            boundary=boundary,
            initial_limit=initial_limit,
            max_limit=max_limit,
        )
    except PositionedJournalPrefixTruncatedError as exc:
        raise ExecutionReconstructionIntegrityError(str(exc)) from exc
    except PositionedJournalBoundaryNotFoundError as exc:
        raise ExecutionReconstructionIntegrityError(str(exc)) from exc
    return positioned, RuntimeHistoryCompleteness.COMPLETE


def _build_attempts(
    causal: tuple[PlatformCausalEvidence, ...],
    positioned: tuple[PositionedRuntimeEvent, ...],
    *,
    execution_lineage: ExecutionLineageReader | None,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    initial_lineage_page_limit: int,
    max_lineage_records: int,
    max_lineage_snapshot_retries: int,
    discovery_snapshot: _RunDiscoverySnapshot,
) -> _AttemptBuildResult:
    causal_by_attempt: dict[AttemptId, list[PlatformCausalEvidence]] = {}
    for evidence in causal:
        attempt_id = evidence.target.attempt_id
        causal_by_attempt.setdefault(attempt_id, []).append(evidence)

    events_by_attempt: dict[AttemptId, list[PositionedRuntimeEvent]] = {}
    for row in positioned:
        attempt_id = row.event.attempt_id
        events_by_attempt.setdefault(attempt_id, []).append(row)

    discovery_by_attempt = {
        record.attempt_id: record for record in discovery_snapshot.records
    }
    runtime_or_causal_ids = set(causal_by_attempt) | set(events_by_attempt)
    candidate_ids = runtime_or_causal_ids | set(discovery_by_attempt)

    attempt_ids = sorted(
        candidate_ids,
        key=lambda attempt_id: _attempt_projection_order_key(
            attempt_id,
            causal_by_attempt=causal_by_attempt,
            events_by_attempt=events_by_attempt,
            discovery_by_attempt=discovery_by_attempt,
            lineage_reader=execution_lineage,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        ),
    )

    attempts: list[ReconstructedAttempt] = []
    discovery_only_candidate_unavailable = False
    for attempt_id in attempt_ids:
        discovery_record = discovery_by_attempt.get(attempt_id)
        is_discovery_only = (
            attempt_id in discovery_by_attempt
            and attempt_id not in runtime_or_causal_ids
        )
        reconstructed = _build_reconstructed_attempt(
            attempt_id=attempt_id,
            causal_by_attempt=causal_by_attempt,
            events_by_attempt=events_by_attempt,
            execution_lineage=execution_lineage,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            initial_lineage_page_limit=initial_lineage_page_limit,
            max_lineage_records=max_lineage_records,
            max_lineage_snapshot_retries=max_lineage_snapshot_retries,
            discovery_record=discovery_record,
            discovery_snapshot=discovery_snapshot,
        )
        if is_discovery_only:
            lineage = reconstructed.lineage
            if lineage is None:
                continue
            if lineage.read_status is ExecutionLineageReadStatus.ABSENT:
                continue
            if lineage.read_status is ExecutionLineageReadStatus.UNAVAILABLE:
                discovery_only_candidate_unavailable = True
                continue
        attempts.append(reconstructed)
    discovery_read_status, discovery_completeness = (
        _effective_attempt_discovery_metadata(
            discovery_snapshot,
            discovery_only_candidate_unavailable=discovery_only_candidate_unavailable,
            lineage_configured=execution_lineage is not None,
        )
    )
    return _AttemptBuildResult(
        attempts=tuple(attempts),
        discovery_read_status=discovery_read_status,
        discovery_completeness=discovery_completeness,
    )


def _build_reconstructed_attempt(
    *,
    attempt_id: AttemptId,
    causal_by_attempt: dict[AttemptId, list[PlatformCausalEvidence]],
    events_by_attempt: dict[AttemptId, list[PositionedRuntimeEvent]],
    execution_lineage: ExecutionLineageReader | None,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    initial_lineage_page_limit: int,
    max_lineage_records: int,
    max_lineage_snapshot_retries: int,
    discovery_record: ExecutionLineageAttemptDiscoveryRecord | None,
    discovery_snapshot: _RunDiscoverySnapshot,
) -> ReconstructedAttempt:
    lineage: ReconstructedAttemptLineage | None = None
    if execution_lineage is not None:
        try:
            lineage = reconstruct_attempt_lineage(
                execution_lineage,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                initial_lineage_page_limit=initial_lineage_page_limit,
                max_lineage_records=max_lineage_records,
                max_lineage_snapshot_retries=max_lineage_snapshot_retries,
            )
        except ExecutionLineageReconstructionIntegrityError as exc:
            raise ExecutionReconstructionIntegrityError(str(exc)) from exc
        if lineage.read_status is ExecutionLineageReadStatus.AVAILABLE:
            _validate_post_v1_discovery_requirements(
                execution_lineage,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                lineage=lineage,
                discovery_record=discovery_record,
                discovery_snapshot=discovery_snapshot,
            )
            if discovery_record is not None:
                lineage = ReconstructedAttemptLineage(
                    attempt_id=lineage.attempt_id,
                    read_status=lineage.read_status,
                    completeness=lineage.completeness,
                    degraded=lineage.degraded,
                    closure_kind=lineage.closure_kind,
                    segments=lineage.segments,
                    discovery_contract_version=lineage.discovery_contract_version,
                    discovery_position=discovery_record.discovery_position,
                )
    return ReconstructedAttempt(
        attempt_id=attempt_id,
        causal_evidence=tuple(causal_by_attempt.get(attempt_id, ())),
        positioned_events=tuple(events_by_attempt.get(attempt_id, ())),
        lineage=lineage,
    )


def _validate_post_v1_discovery_requirements(
    reader: ExecutionLineageReader,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    lineage: ReconstructedAttemptLineage,
    discovery_record: ExecutionLineageAttemptDiscoveryRecord | None,
    discovery_snapshot: _RunDiscoverySnapshot,
) -> None:
    if lineage.discovery_contract_version != 1:
        return
    if (
        discovery_snapshot.read_status
        is ExecutionAttemptDiscoveryReadStatus.UNAVAILABLE
    ):
        return
    run_scope = build_execution_lineage_run_scope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
    )
    if discovery_record is None:
        if (
            discovery_snapshot.completeness
            is ExecutionAttemptDiscoveryCompleteness.TRUNCATED
        ):
            try:
                point = reader.read_attempt_discovery_record(run_scope, attempt_id)
            except ExecutionLineageUnavailableError:
                return
            if point is None:
                raise ExecutionReconstructionIntegrityError(
                    "post-v1 attempt missing discovery record",
                )
        else:
            raise ExecutionReconstructionIntegrityError(
                "post-v1 attempt missing discovery record",
            )
    if (
        discovery_snapshot.read_status is ExecutionAttemptDiscoveryReadStatus.AVAILABLE
        and discovery_snapshot.completeness
        is not ExecutionAttemptDiscoveryCompleteness.TRUNCATED
        and discovery_snapshot.run_state_after is None
    ):
        raise ExecutionReconstructionIntegrityError(
            "post-v1 attempt missing discovery run state",
        )


def _attempt_projection_order_key(
    attempt_id: AttemptId,
    *,
    causal_by_attempt: dict[AttemptId, list[PlatformCausalEvidence]],
    events_by_attempt: dict[AttemptId, list[PositionedRuntimeEvent]],
    discovery_by_attempt: dict[AttemptId, ExecutionLineageAttemptDiscoveryRecord],
    lineage_reader: ExecutionLineageReader | None,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
) -> tuple[int, int | datetime, str]:
    del lineage_reader, tenant_id, task_id, run_id
    discovery = discovery_by_attempt.get(attempt_id)
    if discovery is not None:
        return (0, discovery.discovery_position, str(attempt_id))
    events = events_by_attempt.get(attempt_id, ())
    if events:
        first_position = min(row.position.value for row in events)
        return (1, first_position, str(attempt_id))
    evidence_rows = causal_by_attempt.get(attempt_id, ())
    if evidence_rows:
        first = min(evidence_rows, key=causal_evidence_query_order_key)
        recorded_at, evidence_id = causal_evidence_query_order_key(first)
        return (2, recorded_at, evidence_id)
    return (3, 0, str(attempt_id))


def _validate_causal_evidence_scope(
    evidence: PlatformCausalEvidence,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
) -> None:
    if evidence.tenant_id != tenant_id:
        raise ExecutionReconstructionIntegrityError(
            "causal evidence tenant_id does not match reconstruction scope"
        )
    if evidence.target.tenant_id != tenant_id:
        raise ExecutionReconstructionIntegrityError(
            "causal evidence target.tenant_id does not match reconstruction scope"
        )
    if evidence.target.task_id != task_id:
        raise ExecutionReconstructionIntegrityError(
            "causal evidence target.task_id does not match reconstruction scope"
        )
    if evidence.target.run_id != run_id:
        raise ExecutionReconstructionIntegrityError(
            "causal evidence target.run_id does not match reconstruction scope"
        )


def _validate_runtime_event_scope(
    positioned: PositionedRuntimeEvent,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
) -> None:
    event = positioned.event
    if event.task_id != task_id:
        raise ExecutionReconstructionIntegrityError(
            "runtime event task_id does not match reconstruction scope"
        )
    if event.run_id != run_id:
        raise ExecutionReconstructionIntegrityError(
            "runtime event run_id does not match reconstruction scope"
        )
    if event.tenant_id is not None and event.tenant_id != tenant_id:
        raise ExecutionReconstructionIntegrityError(
            "runtime event tenant_id does not match reconstruction scope"
        )


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise TypeError(f"tenant_id must be str, got {type(tenant_id).__name__}")
    if not tenant_id.strip():
        raise ValueError("tenant_id is required")
    return tenant_id


def _validate_history_limit(limit: int) -> None:
    if type(limit) is not int or isinstance(limit, bool) or limit <= 0:
        raise ValueError("history limit must be > 0")


__all__ = [
    "ExecutionAttemptDiscoveryCompleteness",
    "ExecutionAttemptDiscoveryReadStatus",
    "ExecutionReconstruction",
    "ExecutionReconstructionIntegrityError",
    "ExecutionReconstructor",
    "ReconstructedAttempt",
    "RuntimeHistoryCompleteness",
]
