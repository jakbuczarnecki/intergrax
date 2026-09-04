# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker execution dispatch contracts (AW-5A).

Typed boundary between Autonomous Work and canonical Unified Execution Runtime.
Autonomous Work translates worker/work correlation into runtime-neutral intake;
it does not own Run/Attempt/Execution identity or trusted execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Generic, TypeVar

from intergrax.contracts.autonomous_work._validation import (
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.collaborative_work_bridge import (
    CollaborativeWorkRequestIdentity,
)
from intergrax.contracts.autonomous_work.execution_authority import (
    validate_authority_scopes,
)
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WakeUpId,
    WorkerGoalId,
    WorkerInstanceId,
    validate_responsibility_id,
    validate_wake_up_id,
    validate_worker_goal_id,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.references import (
    WorkReference,
    validate_work_reference,
)
from intergrax.contracts.autonomous_work.revision import Revision, validate_revision
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
)
from intergrax.runtime.execution.request import ExecutionRequest

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ResultT = TypeVar("ResultT")


class WorkerExecutionSourceKind(StrEnum):
    """Implemented dispatch source categories only."""

    GOAL_DECISION = "goal_decision"
    OPERATOR = "operator"
    RECOVERY = "recovery"
    COLLABORATIVE_WORK = "collaborative_work"


class WorkerExecutionDispatchDisposition(StrEnum):
    """Typed dispatch outcome — no fake execution identities on failure."""

    DISPATCHED = "DISPATCHED"
    REJECTED = "REJECTED"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICT = "CONFLICT"


class WorkerExecutionDispatchRejectionReason(StrEnum):
    """Fail-closed rejection semantics for worker execution dispatch."""

    WORKER_NOT_ELIGIBLE = "WORKER_NOT_ELIGIBLE"
    OWNERSHIP_MISMATCH = "OWNERSHIP_MISMATCH"
    STALE_SOURCE = "STALE_SOURCE"
    COLLABORATIVE_AUTHORITY_DENIED = "COLLABORATIVE_AUTHORITY_DENIED"
    RUNTIME_AUTHORITY_DENIED = "RUNTIME_AUTHORITY_DENIED"
    RUNTIME_UNAVAILABLE = "RUNTIME_UNAVAILABLE"
    DISPATCH_FAILED = "DISPATCH_FAILED"


@dataclass(frozen=True, slots=True)
class WorkerExecutionSource:
    """Typed source correlation — not an arbitrary metadata bag."""

    source_kind: WorkerExecutionSourceKind
    source_ref: str

    def __post_init__(self) -> None:
        if type(self.source_kind) is not WorkerExecutionSourceKind:
            raise TypeError("source_kind must be WorkerExecutionSourceKind")
        object.__setattr__(
            self,
            "source_ref",
            require_non_empty_text(self.source_ref, label="source_ref"),
        )


@dataclass(frozen=True, slots=True)
class WorkerExecutionCorrelation:
    """Immutable worker→execution correlation — not a second event store."""

    worker_instance_id: WorkerInstanceId
    source: WorkerExecutionSource
    run_id: RunId | None
    attempt_id: AttemptId | None
    execution_id: ExecutionId | None
    goal_id: WorkerGoalId | None = None
    responsibility_id: ResponsibilityId | None = None
    wake_up_id: WakeUpId | None = None
    collaborative_work_ref: WorkReference | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.source) is not WorkerExecutionSource:
            raise TypeError("source must be WorkerExecutionSource")
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        if self.goal_id is not None:
            validate_worker_goal_id(self.goal_id)
        if self.responsibility_id is not None:
            validate_responsibility_id(self.responsibility_id)
        if self.wake_up_id is not None:
            validate_wake_up_id(self.wake_up_id)
        if self.collaborative_work_ref is not None:
            validate_work_reference(self.collaborative_work_ref)
        if self.created_at is not None:
            object.__setattr__(
                self,
                "created_at",
                require_aware_utc(self.created_at, label="created_at"),
            )


@dataclass(frozen=True, slots=True)
class WorkerExecutionDispatchRequest(Generic[InputT, OutputT]):
    """Immutable worker execution dispatch request."""

    worker_instance_id: WorkerInstanceId
    worker_revision: Revision
    requested_scopes: tuple[str, ...]
    runtime_request: ExecutionRequest[InputT, OutputT]
    source: WorkerExecutionSource
    requested_at: datetime
    goal_id: WorkerGoalId | None = None
    goal_revision: Revision | None = None
    responsibility_id: ResponsibilityId | None = None
    wake_up_id: WakeUpId | None = None
    collaborative_work_ref: WorkReference | None = None
    work_request_identity: CollaborativeWorkRequestIdentity | None = None
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.worker_revision) is not Revision:
            raise TypeError("worker_revision must be Revision")
        validate_revision(self.worker_revision)
        object.__setattr__(
            self,
            "requested_scopes",
            validate_authority_scopes(self.requested_scopes),
        )
        if type(self.runtime_request) is not ExecutionRequest:
            raise TypeError("runtime_request must be ExecutionRequest")
        if type(self.source) is not WorkerExecutionSource:
            raise TypeError("source must be WorkerExecutionSource")
        object.__setattr__(
            self,
            "requested_at",
            require_aware_utc(self.requested_at, label="requested_at"),
        )
        if self.goal_id is not None:
            validate_worker_goal_id(self.goal_id)
        if self.goal_revision is not None:
            if type(self.goal_revision) is not Revision:
                raise TypeError("goal_revision must be Revision")
            validate_revision(self.goal_revision)
        if self.responsibility_id is not None:
            validate_responsibility_id(self.responsibility_id)
        if self.wake_up_id is not None:
            validate_wake_up_id(self.wake_up_id)
        if self.collaborative_work_ref is not None:
            validate_work_reference(self.collaborative_work_ref)
        if self.work_request_identity is not None:
            if type(self.work_request_identity) is not CollaborativeWorkRequestIdentity:
                raise TypeError(
                    "work_request_identity must be CollaborativeWorkRequestIdentity"
                )
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)


@dataclass(frozen=True, slots=True)
class WorkerExecutionDispatchResult(Generic[ResultT]):
    """Typed dispatch result — runtime IDs only after successful dispatch."""

    disposition: WorkerExecutionDispatchDisposition
    correlation: WorkerExecutionCorrelation
    rejection_reason: WorkerExecutionDispatchRejectionReason | None = None
    runtime_result: ResultT | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not WorkerExecutionDispatchDisposition:
            raise TypeError("disposition must be WorkerExecutionDispatchDisposition")
        if type(self.correlation) is not WorkerExecutionCorrelation:
            raise TypeError("correlation must be WorkerExecutionCorrelation")
        if self.rejection_reason is not None:
            if type(self.rejection_reason) is not WorkerExecutionDispatchRejectionReason:
                raise TypeError(
                    "rejection_reason must be WorkerExecutionDispatchRejectionReason"
                )
        if self.disposition is WorkerExecutionDispatchDisposition.DISPATCHED:
            if self.correlation.run_id is None:
                raise ValueError("DISPATCHED requires run_id")
            if self.correlation.attempt_id is None:
                raise ValueError("DISPATCHED requires attempt_id")
            if self.correlation.execution_id is None:
                raise ValueError("DISPATCHED requires execution_id")
        else:
            if (
                self.correlation.run_id is not None
                or self.correlation.attempt_id is not None
                or self.correlation.execution_id is not None
            ):
                raise ValueError(
                    "non-DISPATCHED results must not expose runtime execution identities"
                )
