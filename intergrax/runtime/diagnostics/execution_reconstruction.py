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
from intergrax.contracts.execution_lineage import ExecutionLineageReader
from intergrax.runtime.diagnostics.execution_lineage_reconstruction import (
    ExecutionLineageCompleteness,
    ExecutionLineageReadStatus,
    ExecutionLineageReconstructionIntegrityError,
    ReconstructedAttemptLineage,
    reconstruct_attempt_lineage,
)
from intergrax.runtime.events.execution_position import PositionedRuntimeEvent
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
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
    ) -> None:
        self._runtime_events = runtime_events
        self._causal_evidence = causal_evidence
        self._execution_lineage = execution_lineage
        self._initial_lineage_page_limit = initial_lineage_page_limit
        self._max_lineage_records = max_lineage_records

    def reconstruct_execution(
        self,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        *,
        initial_limit: int = 1000,
        max_limit: int = 1_000_000,
    ) -> ExecutionReconstruction:
        tenant_id = _require_tenant_id(tenant_id)
        task_id = validate_task_id(task_id)
        run_id = validate_run_id(run_id)
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

        positioned, completeness = _load_positioned_events_for_run(
            self._runtime_events,
            tenant_id=tenant_id,
            run_id=run_id,
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

        attempts = _build_attempts(
            causal,
            positioned,
            execution_lineage=self._execution_lineage,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            initial_lineage_page_limit=self._initial_lineage_page_limit,
            max_lineage_records=self._max_lineage_records,
        )
        return ExecutionReconstruction(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            causal_evidence=causal,
            positioned_events=positioned,
            attempts=attempts,
            runtime_history_completeness=completeness,
        )


def _load_positioned_events_for_run(
    runtime_store: RuntimeEventPersistence,
    *,
    tenant_id: str,
    run_id: RunId,
    initial_limit: int,
    max_limit: int,
) -> tuple[tuple[PositionedRuntimeEvent, ...], RuntimeHistoryCompleteness]:
    """
    Load positioned runtime history via the canonical store read path.

    Paginates by increasing ``limit`` until the run history is complete or
    ``max_limit`` is reached with a full batch (truncated).
    """
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
) -> tuple[ReconstructedAttempt, ...]:
    causal_by_attempt: dict[AttemptId, list[PlatformCausalEvidence]] = {}
    for evidence in causal:
        attempt_id = evidence.target.attempt_id
        causal_by_attempt.setdefault(attempt_id, []).append(evidence)

    events_by_attempt: dict[AttemptId, list[PositionedRuntimeEvent]] = {}
    for row in positioned:
        attempt_id = row.event.attempt_id
        events_by_attempt.setdefault(attempt_id, []).append(row)

    attempt_ids = sorted(
        set(causal_by_attempt) | set(events_by_attempt),
        key=lambda attempt_id: _attempt_projection_order_key(
            attempt_id,
            causal_by_attempt=causal_by_attempt,
            events_by_attempt=events_by_attempt,
        ),
    )

    return tuple(
        _build_reconstructed_attempt(
            attempt_id=attempt_id,
            causal_by_attempt=causal_by_attempt,
            events_by_attempt=events_by_attempt,
            execution_lineage=execution_lineage,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            initial_lineage_page_limit=initial_lineage_page_limit,
            max_lineage_records=max_lineage_records,
        )
        for attempt_id in attempt_ids
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
            )
        except ExecutionLineageReconstructionIntegrityError as exc:
            raise ExecutionReconstructionIntegrityError(str(exc)) from exc
    return ReconstructedAttempt(
        attempt_id=attempt_id,
        causal_evidence=tuple(causal_by_attempt.get(attempt_id, ())),
        positioned_events=tuple(events_by_attempt.get(attempt_id, ())),
        lineage=lineage,
    )


def _attempt_projection_order_key(
    attempt_id: AttemptId,
    *,
    causal_by_attempt: dict[AttemptId, list[PlatformCausalEvidence]],
    events_by_attempt: dict[AttemptId, list[PositionedRuntimeEvent]],
) -> tuple[int, int | datetime, str]:
    """
    Projection-only attempt ordering — not execution identity.

    Prefer first canonical ``ExecutionEventPosition`` when runtime events exist;
    otherwise earliest causal evidence ``(recorded_at, evidence_id)``.
    """
    events = events_by_attempt.get(attempt_id, ())
    if events:
        return (0, events[0].position.value, str(attempt_id))
    evidence_rows = causal_by_attempt.get(attempt_id, ())
    if evidence_rows:
        first = min(evidence_rows, key=causal_evidence_query_order_key)
        recorded_at, evidence_id = causal_evidence_query_order_key(first)
        return (1, recorded_at, evidence_id)
    return (2, 0, str(attempt_id))


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
