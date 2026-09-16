# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GR-5-R1 — canonical Execution Engine HITL continuation contract.

Execution Engine owns pause / wait / resume lifecycle for one exact four-ID Execution.
Governance owns REQUIRE_HUMAN scope and grant evidence; this port owns lifecycle truth.

Nexus is **not** referenced here — it remains an internal Execution Engine subsystem (ADR-GR-5-001).

Resume of the same Execution is **not** root admission (GR-2). Changing AttemptId or
ExecutionId is a new execution, not continuation.

Two-phase resolution model (R1):
  ``apply_resolution(APPROVE)`` → ``RESUME_AUTHORIZED`` (human verdict recorded separately)
  ``resume()`` → ``RESUMED`` with CAS on revision.

| Current state      | Command / event              | Next state         | Allowed |
| ------------------ | ---------------------------- | ------------------ | ------: |
| (none)             | request_pause                | PAUSE_REQUESTED    | yes     |
| PAUSE_REQUESTED    | advance_to_paused            | PAUSED             | yes     |
| PAUSED             | advance_to_human_wait        | WAITING_FOR_HUMAN  | yes     |
| WAITING_FOR_HUMAN  | apply_resolution(APPROVE)    | RESUME_AUTHORIZED  | yes     |
| WAITING_FOR_HUMAN  | apply_resolution(REJECT)     | REJECTED           | yes     |
| WAITING_FOR_HUMAN  | apply_resolution(ESCALATE)   | ESCALATED          | yes     |
| WAITING_FOR_HUMAN  | cancel_continuation          | CANCELLED          | yes     |
| RESUME_AUTHORIZED  | resume                       | RESUMED            | yes     |
| RESUMED            | any lifecycle command        | —                  | no      |
| REJECTED           | any lifecycle command        | —                  | no      |
| ESCALATED          | any lifecycle command        | —                  | no      |
| CANCELLED          | any lifecycle command        | —                  | no      |

Revision / CAS (successful transition):
  ``expected_revision == pending.revision`` → persist with ``revision + 1``.
  Mismatch → ``STALE_REVISION``. Terminal states do not accept further transitions.

``continuation_id`` is the stable continuation identity; it equals governed
``continuation_request_id`` when the pause is HITL-governed.

When ``governed_correlation`` is present, ``continuation_id``, four-ID ``identity``,
and ``reason`` must match that correlation at construction (fail closed).

``human_request_id`` and ``pause_id`` on a pending snapshot are **authoritative**
correlation identifiers when set; resolution commands must match exactly and must
not replace canonical values.

Governed resolution scope: ``operation_id`` and ``side_effect_scope_id`` (when set)
must match exactly. When ``side_effect_scope_digest`` is set on correlation, the
command digest must match; when correlation digest is ``None``, digest is not required
and a command-only digest does not establish scope authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewDecision,
    DecisionHumanReviewOutcome,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.contracts.human_approver import HumanApproverEvidence

SCHEMA_PENDING_EXECUTION_CONTINUATION_V1: Final = "pending_execution_continuation.v1"
SCHEMA_EXECUTION_CONTINUATION_RESOLUTION_V1: Final = "execution_continuation_resolution.v1"

_NON_EMPTY = Field(min_length=1)


class ExecutionContinuationLifecycleState(StrEnum):
    """Execution-owned continuation lifecycle (UER-aligned + RESUME_AUTHORIZED for two-phase)."""

    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    WAITING_FOR_HUMAN = "waiting_for_human"
    RESUME_AUTHORIZED = "resume_authorized"
    RESUMED = "resumed"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


_TERMINAL_LIFECYCLE_STATES: frozenset[ExecutionContinuationLifecycleState] = frozenset(
    {
        ExecutionContinuationLifecycleState.RESUMED,
        ExecutionContinuationLifecycleState.REJECTED,
        ExecutionContinuationLifecycleState.ESCALATED,
        ExecutionContinuationLifecycleState.CANCELLED,
    }
)


class ExecutionHumanVerdict(StrEnum):
    """Human resolution verdict — evidence only until lifecycle transition applies it."""

    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


class ExecutionContinuationTransition(StrEnum):
    """Explicit lifecycle commands validated by ``advance_continuation_lifecycle``."""

    REQUEST_PAUSE = "request_pause"
    ADVANCE_TO_PAUSED = "advance_to_paused"
    ADVANCE_TO_HUMAN_WAIT = "advance_to_human_wait"
    APPLY_RESOLUTION = "apply_resolution"
    CANCEL_CONTINUATION = "cancel_continuation"
    RESUME = "resume"


class ExecutionContinuationErrorCode(StrEnum):
    """Fail-closed continuation contract outcomes."""

    NOT_FOUND = "not_found"
    IDENTITY_MISMATCH = "identity_mismatch"
    STALE_REVISION = "stale_revision"
    INVALID_TRANSITION = "invalid_transition"
    ALREADY_RESOLVED = "already_resolved"
    ALREADY_RESUMED = "already_resumed"
    INVALID_RESOLUTION = "invalid_resolution"
    SCOPE_MISMATCH = "scope_mismatch"
    DUPLICATE_CONTINUATION = "duplicate_continuation"
    EXECUTION_PROGRESS_BLOCKED = "execution_progress_blocked"
    AMBIGUOUS_IDENTITY = "ambiguous_identity"
    STORE_QUERY_FAILED = "store_query_failed"
    INCOMPLETE_EXECUTION_IDENTITY = "incomplete_execution_identity"
    NON_DURABLE_CONTINUATION_STORE = "non_durable_continuation_store"
    CORRUPT_CONTINUATION_STATE = "corrupt_continuation_state"


class ExecutionContinuationError(ValueError):
    """Typed continuation boundary failure."""

    __slots__ = ("code",)

    def __init__(self, message: str, *, code: ExecutionContinuationErrorCode) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ExecutionContinuationRecoveryHandle:
    """Opaque durable recovery reference to one canonical continuation episode snapshot.

    Identifies persisted continuation state for process-boundary restore. The host
    obtains this when persisting the current episode; recovery resolves exact four-ID
    from the canonical snapshot after load (not from caller-supplied identity).
    """

    continuation_id: str

    def __post_init__(self) -> None:
        normalized = self.continuation_id.strip()
        if not normalized:
            raise ValueError("execution continuation recovery handle requires continuation_id")
        object.__setattr__(self, "continuation_id", normalized)


def execution_continuation_recovery_handle_for_continuation_id(
    continuation_id: str,
) -> ExecutionContinuationRecoveryHandle:
    """Build a recovery handle for the given canonical ``continuation_id``."""
    return ExecutionContinuationRecoveryHandle(continuation_id=continuation_id)


@dataclass(frozen=True, slots=True)
class ExecutionContinuationIdentity:
    """Mandatory four-ID binding for every canonical continuation operation."""

    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", validate_task_id(self.task_id))
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "attempt_id", validate_attempt_id(self.attempt_id))
        object.__setattr__(self, "execution_id", validate_execution_id(self.execution_id))


def assert_governed_correlation_matches_continuation(
    *,
    continuation_id: str,
    identity: ExecutionContinuationIdentity,
    reason: ContinuationReason,
    governed_correlation: GovernedContinuationCorrelation | None,
    label: str = "continuation",
) -> None:
    """Fail closed when governed correlation disagrees with continuation authority fields."""
    if governed_correlation is None:
        return
    if continuation_id != governed_correlation.continuation_request_id:
        raise ExecutionContinuationError(
            f"{label} continuation_id mismatch with governed correlation",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if identity.task_id != governed_correlation.task_id:
        raise ExecutionContinuationError(
            f"{label} task_id mismatch with governed correlation",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if identity.run_id != governed_correlation.run_id:
        raise ExecutionContinuationError(
            f"{label} run_id mismatch with governed correlation",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if identity.attempt_id != governed_correlation.attempt_id:
        raise ExecutionContinuationError(
            f"{label} attempt_id mismatch with governed correlation",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if identity.execution_id != governed_correlation.execution_id:
        raise ExecutionContinuationError(
            f"{label} execution_id mismatch with governed correlation",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if reason != governed_correlation.reason:
        raise ExecutionContinuationError(
            f"{label} reason mismatch with governed correlation",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )


def assert_execution_continuation_identity_match(
    expected: ExecutionContinuationIdentity,
    actual: ExecutionContinuationIdentity,
    *,
    label: str = "continuation",
) -> None:
    """Fail closed on any single four-ID mismatch (independent checks)."""
    if expected.task_id != actual.task_id:
        raise ExecutionContinuationError(
            f"{label} task_id mismatch",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if expected.run_id != actual.run_id:
        raise ExecutionContinuationError(
            f"{label} run_id mismatch",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if expected.attempt_id != actual.attempt_id:
        raise ExecutionContinuationError(
            f"{label} attempt_id mismatch",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if expected.execution_id != actual.execution_id:
        raise ExecutionContinuationError(
            f"{label} execution_id mismatch",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )


def validate_continuation_revision(value: object) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError("continuation revision must be int")
    if value < 1:
        raise ValueError("continuation revision must be >= 1")
    return value


_PROGRESS_BLOCKING_LIFECYCLE_STATES: frozenset[ExecutionContinuationLifecycleState] = frozenset(
    {
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        ExecutionContinuationLifecycleState.REJECTED,
        ExecutionContinuationLifecycleState.ESCALATED,
        ExecutionContinuationLifecycleState.CANCELLED,
    },
)


def execution_continuation_lifecycle_is_terminal(
    state: ExecutionContinuationLifecycleState,
) -> bool:
    """Whether the lifecycle state closes the continuation episode."""
    return state in _TERMINAL_LIFECYCLE_STATES


def execution_continuation_lifecycle_permits_successor_episode(
    state: ExecutionContinuationLifecycleState,
) -> bool:
    """Whether a new continuation episode may replace ``state`` as the current episode."""
    return execution_continuation_lifecycle_is_terminal(state)


def execution_continuation_lifecycle_blocks_execution_progress(
    state: ExecutionContinuationLifecycleState,
) -> bool:
    """Platform invariant: when true, canonical Execution must not progress blocked work.

    ``PAUSE_REQUESTED`` is intentionally excluded — pause is requested but the Engine may
    still reach a safe quiescence point. ``RESUMED`` clears the gate for the same Execution.
    """
    return state in _PROGRESS_BLOCKING_LIFECYCLE_STATES


def advance_continuation_lifecycle(
    current: ExecutionContinuationLifecycleState | None,
    transition: ExecutionContinuationTransition,
    *,
    verdict: ExecutionHumanVerdict | None = None,
) -> ExecutionContinuationLifecycleState:
    """Pure transition table — raises ``INVALID_TRANSITION`` when not allowed."""
    if current in _TERMINAL_LIFECYCLE_STATES:
        raise ExecutionContinuationError(
            f"terminal state {current} rejects {transition}",
            code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
        )
    if transition is ExecutionContinuationTransition.REQUEST_PAUSE:
        if current is not None:
            raise ExecutionContinuationError(
                "continuation already exists",
                code=ExecutionContinuationErrorCode.DUPLICATE_CONTINUATION,
            )
        return ExecutionContinuationLifecycleState.PAUSE_REQUESTED
    if transition is ExecutionContinuationTransition.ADVANCE_TO_PAUSED:
        if current is not ExecutionContinuationLifecycleState.PAUSE_REQUESTED:
            raise ExecutionContinuationError(
                "advance_to_paused requires pause_requested",
                code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
            )
        return ExecutionContinuationLifecycleState.PAUSED
    if transition is ExecutionContinuationTransition.ADVANCE_TO_HUMAN_WAIT:
        if current is not ExecutionContinuationLifecycleState.PAUSED:
            raise ExecutionContinuationError(
                "advance_to_human_wait requires paused",
                code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
            )
        return ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
    if transition is ExecutionContinuationTransition.CANCEL_CONTINUATION:
        if current is not ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN:
            raise ExecutionContinuationError(
                "cancel requires waiting_for_human",
                code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
            )
        return ExecutionContinuationLifecycleState.CANCELLED
    if transition is ExecutionContinuationTransition.APPLY_RESOLUTION:
        if current is not ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN:
            raise ExecutionContinuationError(
                "resolution requires waiting_for_human",
                code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
            )
        if verdict is None:
            raise ExecutionContinuationError(
                "resolution verdict required",
                code=ExecutionContinuationErrorCode.INVALID_RESOLUTION,
            )
        if verdict is ExecutionHumanVerdict.APPROVE:
            return ExecutionContinuationLifecycleState.RESUME_AUTHORIZED
        if verdict is ExecutionHumanVerdict.REJECT:
            return ExecutionContinuationLifecycleState.REJECTED
        if verdict is ExecutionHumanVerdict.ESCALATE:
            return ExecutionContinuationLifecycleState.ESCALATED
        raise ExecutionContinuationError(
            f"unsupported verdict {verdict}",
            code=ExecutionContinuationErrorCode.INVALID_RESOLUTION,
        )
    if transition is ExecutionContinuationTransition.RESUME:
        if current is not ExecutionContinuationLifecycleState.RESUME_AUTHORIZED:
            raise ExecutionContinuationError(
                "resume requires resume_authorized",
                code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
            )
        return ExecutionContinuationLifecycleState.RESUMED
    raise ExecutionContinuationError(
        f"unknown transition {transition}",
        code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
    )


class PendingExecutionContinuation(BaseModel):
    """Immutable snapshot of one pending continuation (restorable after restart)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["pending_execution_continuation.v1"] = (
        SCHEMA_PENDING_EXECUTION_CONTINUATION_V1
    )
    continuation_id: str = _NON_EMPTY
    identity: ExecutionContinuationIdentity
    lifecycle_state: ExecutionContinuationLifecycleState
    revision: int = Field(ge=1)
    reason: ContinuationReason
    human_verdict: ExecutionHumanVerdict | None = None
    governed_correlation: GovernedContinuationCorrelation | None = None
    pause_id: str | None = None
    human_request_id: str | None = None
    requested_at: str | None = None

    @field_validator("continuation_id", "pause_id", "human_request_id", "requested_at")
    @classmethod
    def _strip_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("revision")
    @classmethod
    def _validate_revision(cls, value: int) -> int:
        return validate_continuation_revision(value)

    @model_validator(mode="after")
    def _identity_dataclass(self) -> Self:
        if not isinstance(self.identity, ExecutionContinuationIdentity):
            raise TypeError("identity must be ExecutionContinuationIdentity")
        try:
            assert_governed_correlation_matches_continuation(
                continuation_id=self.continuation_id,
                identity=self.identity,
                reason=self.reason,
                governed_correlation=self.governed_correlation,
            )
        except ExecutionContinuationError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="before")
    @classmethod
    def _coerce_identity(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        identity = data.get("identity")
        if identity is not None and not isinstance(identity, ExecutionContinuationIdentity):
            data = dict(data)
            data["identity"] = ExecutionContinuationIdentity(**identity)
        return data


class ExecutionPauseRequest(BaseModel):
    """Begin canonical pause for one exact Execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: ExecutionContinuationIdentity
    continuation_id: str = _NON_EMPTY
    reason: ContinuationReason
    governed_correlation: GovernedContinuationCorrelation | None = None
    pause_id: str | None = None
    human_request_id: str | None = None
    requested_at: str | None = None

    @field_validator("continuation_id")
    @classmethod
    def _strip_continuation_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("continuation_id must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _governed_correlation_consistency(self) -> Self:
        try:
            assert_governed_correlation_matches_continuation(
                continuation_id=self.continuation_id,
                identity=self.identity,
                reason=self.reason,
                governed_correlation=self.governed_correlation,
            )
        except ExecutionContinuationError as exc:
            raise ValueError(str(exc)) from exc
        return self


class ExecutionContinuationLookup(BaseModel):
    """Exact lookup — continuation id and/or full four-ID identity (never partial)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    continuation_id: str | None = None
    identity: ExecutionContinuationIdentity | None = None

    @model_validator(mode="after")
    def _require_exact_key(self) -> Self:
        if self.continuation_id is None and self.identity is None:
            raise ValueError("continuation_id or full identity required")
        return self


class ExecutionContinuationResolutionCommand(BaseModel):
    """Atomic apply: match identity, continuation, revision, optional governed scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["execution_continuation_resolution.v1"] = (
        SCHEMA_EXECUTION_CONTINUATION_RESOLUTION_V1
    )
    continuation_id: str = _NON_EMPTY
    identity: ExecutionContinuationIdentity
    expected_revision: int = Field(ge=1)
    verdict: ExecutionHumanVerdict
    approver: HumanApproverEvidence
    human_request_id: str = _NON_EMPTY
    pause_id: str | None = None
    operation_id: str | None = None
    side_effect_scope_id: str | None = None
    side_effect_scope_digest: str | None = None
    resolved_at: str = _NON_EMPTY

    @field_validator("expected_revision")
    @classmethod
    def _validate_expected_revision(cls, value: int) -> int:
        return validate_continuation_revision(value)


class ExecutionContinuationResumeCommand(BaseModel):
    """CAS resume after approved resolution — does not execute business graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    continuation_id: str = _NON_EMPTY
    identity: ExecutionContinuationIdentity
    expected_revision: int = Field(ge=1)

    @field_validator("expected_revision")
    @classmethod
    def _validate_expected_revision(cls, value: int) -> int:
        return validate_continuation_revision(value)


def assert_pending_matches_resolution_command(
    pending: PendingExecutionContinuation,
    command: ExecutionContinuationResolutionCommand,
) -> None:
    """Precondition checks shared by port implementations (fail closed)."""
    if pending.continuation_id != command.continuation_id:
        raise ExecutionContinuationError(
            "continuation_id mismatch",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    assert_execution_continuation_identity_match(pending.identity, command.identity)
    if pending.revision != command.expected_revision:
        raise ExecutionContinuationError(
            "stale continuation revision for resolution",
            code=ExecutionContinuationErrorCode.STALE_REVISION,
        )
    if pending.lifecycle_state is not ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN:
        if pending.lifecycle_state in {
            ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
            ExecutionContinuationLifecycleState.REJECTED,
            ExecutionContinuationLifecycleState.ESCALATED,
        }:
            raise ExecutionContinuationError(
                "continuation already resolved",
                code=ExecutionContinuationErrorCode.ALREADY_RESOLVED,
            )
        raise ExecutionContinuationError(
            f"invalid state {pending.lifecycle_state} for resolution",
            code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
        )
    correlation = pending.governed_correlation
    if correlation is not None:
        if command.operation_id is None or command.operation_id != correlation.operation_id:
            raise ExecutionContinuationError(
                "operation_id mismatch for governed continuation",
                code=ExecutionContinuationErrorCode.SCOPE_MISMATCH,
            )
        if (
            correlation.side_effect_scope_id is not None
            and command.side_effect_scope_id != correlation.side_effect_scope_id
        ):
            raise ExecutionContinuationError(
                "side_effect_scope_id mismatch",
                code=ExecutionContinuationErrorCode.SCOPE_MISMATCH,
            )
        if (
            correlation.side_effect_scope_digest is not None
            and command.side_effect_scope_digest != correlation.side_effect_scope_digest
        ):
            raise ExecutionContinuationError(
                "side_effect_scope_digest mismatch",
                code=ExecutionContinuationErrorCode.SCOPE_MISMATCH,
            )
    if pending.human_request_id is not None:
        if command.human_request_id != pending.human_request_id:
            raise ExecutionContinuationError(
                "human_request_id mismatch",
                code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
            )
    if pending.pause_id is not None:
        if command.pause_id != pending.pause_id:
            raise ExecutionContinuationError(
                "pause_id mismatch",
                code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
            )


def decision_human_review_outcome_to_execution_verdict(
    outcome: DecisionHumanReviewOutcome,
) -> ExecutionHumanVerdict:
    """Map canonical Decision human review outcome to execution continuation verdict."""
    if type(outcome) is not DecisionHumanReviewOutcome:
        raise TypeError("outcome must be DecisionHumanReviewOutcome")
    if outcome is DecisionHumanReviewOutcome.APPROVED:
        return ExecutionHumanVerdict.APPROVE
    if outcome is DecisionHumanReviewOutcome.REJECTED:
        return ExecutionHumanVerdict.REJECT
    if outcome is DecisionHumanReviewOutcome.ESCALATED:
        return ExecutionHumanVerdict.ESCALATE
    raise ValueError(f"unsupported DecisionHumanReviewOutcome: {outcome!s}")


def execution_continuation_resolution_command_for_pending_human_verdict(
    pending: PendingExecutionContinuation,
    *,
    verdict: ExecutionHumanVerdict,
    approver: HumanApproverEvidence,
    human_request_id: str,
    resolved_at: str,
) -> ExecutionContinuationResolutionCommand:
    """Build one resolution command from canonical pending snapshot and human evidence."""
    if type(pending) is not PendingExecutionContinuation:
        raise TypeError("pending must be PendingExecutionContinuation")
    if type(verdict) is not ExecutionHumanVerdict:
        raise TypeError("verdict must be ExecutionHumanVerdict")
    if type(approver) is not HumanApproverEvidence:
        raise TypeError("approver must be HumanApproverEvidence")
    normalized_request_id = human_request_id.strip()
    if not normalized_request_id:
        raise ValueError("human_request_id must be non-empty")
    if pending.pause_id is None:
        raise ExecutionContinuationError(
            "canonical pending pause_id required",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if pending.human_request_id is not None and pending.human_request_id != normalized_request_id:
        raise ExecutionContinuationError(
            "human_request_id mismatch against pending continuation",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    correlation = pending.governed_correlation
    operation_id = correlation.operation_id if correlation is not None else None
    side_effect_scope_id = (
        correlation.side_effect_scope_id if correlation is not None else None
    )
    side_effect_scope_digest = (
        correlation.side_effect_scope_digest if correlation is not None else None
    )
    return ExecutionContinuationResolutionCommand(
        continuation_id=pending.continuation_id,
        identity=pending.identity,
        expected_revision=pending.revision,
        verdict=verdict,
        approver=approver,
        human_request_id=normalized_request_id,
        pause_id=pending.pause_id,
        operation_id=operation_id,
        side_effect_scope_id=side_effect_scope_id,
        side_effect_scope_digest=side_effect_scope_digest,
        resolved_at=resolved_at,
    )


def execution_continuation_resolution_command_from_decision_human_review_decision(
    pending: PendingExecutionContinuation,
    decision: DecisionHumanReviewDecision,
    *,
    resolved_at: str,
) -> ExecutionContinuationResolutionCommand:
    """Project one consumed Decision human review decision onto continuation resolution."""
    if type(decision) is not DecisionHumanReviewDecision:
        raise TypeError("decision must be DecisionHumanReviewDecision")
    provenance_request_id = decision.provenance.human_request_id.strip()
    if provenance_request_id != str(decision.request_id):
        raise ExecutionContinuationError(
            "decision human_request_id must match request_id",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    if decision.approver.tenant_id != decision.proposal_ref.identity.tenant_id:
        raise ExecutionContinuationError(
            "decision approver tenant_id mismatch",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    verdict = decision_human_review_outcome_to_execution_verdict(decision.outcome)
    return execution_continuation_resolution_command_for_pending_human_verdict(
        pending,
        verdict=verdict,
        approver=decision.approver,
        human_request_id=provenance_request_id,
        resolved_at=resolved_at,
    )


def assert_pending_matches_resume_command(
    pending: PendingExecutionContinuation,
    command: ExecutionContinuationResumeCommand,
) -> None:
    if pending.continuation_id != command.continuation_id:
        raise ExecutionContinuationError(
            "continuation_id mismatch",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    assert_execution_continuation_identity_match(pending.identity, command.identity)
    if pending.revision != command.expected_revision:
        raise ExecutionContinuationError(
            "stale continuation revision for resume",
            code=ExecutionContinuationErrorCode.STALE_REVISION,
        )
    if pending.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED:
        raise ExecutionContinuationError(
            "continuation already resumed",
            code=ExecutionContinuationErrorCode.ALREADY_RESUMED,
        )
    if pending.lifecycle_state is not ExecutionContinuationLifecycleState.RESUME_AUTHORIZED:
        raise ExecutionContinuationError(
            f"invalid state {pending.lifecycle_state} for resume",
            code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
        )


def apply_resolution_to_pending(
    pending: PendingExecutionContinuation,
    command: ExecutionContinuationResolutionCommand,
) -> PendingExecutionContinuation:
    """Pure CAS transition helper after preconditions pass."""
    assert_pending_matches_resolution_command(pending, command)
    next_state = advance_continuation_lifecycle(
        pending.lifecycle_state,
        ExecutionContinuationTransition.APPLY_RESOLUTION,
        verdict=command.verdict,
    )
    update: dict[str, object] = {
        "lifecycle_state": next_state,
        "revision": pending.revision + 1,
        "human_verdict": command.verdict,
    }
    if pending.human_request_id is None:
        update["human_request_id"] = command.human_request_id
    if pending.pause_id is None and command.pause_id is not None:
        update["pause_id"] = command.pause_id
    return pending.model_copy(update=update)


def apply_resume_to_pending(
    pending: PendingExecutionContinuation,
    command: ExecutionContinuationResumeCommand,
) -> PendingExecutionContinuation:
    assert_pending_matches_resume_command(pending, command)
    next_state = advance_continuation_lifecycle(
        pending.lifecycle_state,
        ExecutionContinuationTransition.RESUME,
    )
    return pending.model_copy(
        update={
            "lifecycle_state": next_state,
            "revision": pending.revision + 1,
        },
    )


@runtime_checkable
class ExecutionContinuationPort(Protocol):
    """Semantic Execution Engine continuation boundary (persistence is internal)."""

    def request_pause(self, request: ExecutionPauseRequest) -> PendingExecutionContinuation:
        """Record PAUSE_REQUESTED for one continuation_id and four-ID identity."""

    def get_pending(self, lookup: ExecutionContinuationLookup) -> PendingExecutionContinuation:
        """Return exact pending snapshot; NOT_FOUND when missing."""

    def apply_resolution(
        self,
        command: ExecutionContinuationResolutionCommand,
    ) -> PendingExecutionContinuation:
        """CAS human resolution while WAITING_FOR_HUMAN."""

    def resume(self, command: ExecutionContinuationResumeCommand) -> PendingExecutionContinuation:
        """CAS transition RESUME_AUTHORIZED → RESUMED for same four IDs."""


__all__ = [
    "ExecutionContinuationError",
    "ExecutionContinuationErrorCode",
    "ExecutionContinuationIdentity",
    "ExecutionContinuationRecoveryHandle",
    "execution_continuation_recovery_handle_for_continuation_id",
    "ExecutionContinuationLifecycleState",
    "ExecutionContinuationLookup",
    "ExecutionContinuationPort",
    "ExecutionContinuationResolutionCommand",
    "ExecutionContinuationResumeCommand",
    "ExecutionContinuationTransition",
    "ExecutionHumanVerdict",
    "ExecutionPauseRequest",
    "PendingExecutionContinuation",
    "SCHEMA_EXECUTION_CONTINUATION_RESOLUTION_V1",
    "SCHEMA_PENDING_EXECUTION_CONTINUATION_V1",
    "advance_continuation_lifecycle",
    "execution_continuation_lifecycle_blocks_execution_progress",
    "execution_continuation_lifecycle_is_terminal",
    "execution_continuation_lifecycle_permits_successor_episode",
    "apply_resolution_to_pending",
    "apply_resume_to_pending",
    "decision_human_review_outcome_to_execution_verdict",
    "execution_continuation_resolution_command_for_pending_human_verdict",
    "execution_continuation_resolution_command_from_decision_human_review_decision",
    "assert_execution_continuation_identity_match",
    "assert_governed_correlation_matches_continuation",
    "assert_pending_matches_resolution_command",
    "assert_pending_matches_resume_command",
    "validate_continuation_revision",
]
