# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical dynamic orchestration topology submission contracts (Execution/Nexus)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Mapping, NewType, Protocol, TypeVar

from typing_extensions import TypeAlias

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT", covariant=True)

OrchestrationSlotId = NewType("OrchestrationSlotId", str)
OrchestrationTopologyExecutionId = NewType("OrchestrationTopologyExecutionId", str)


class OrchestrationTopologyValidationError(ValueError):
    """Raised when an orchestration topology fails structural validation."""


class OrchestrationSchedulingPolicyValidationError(ValueError):
    """Raised when orchestration scheduling policy is invalid."""


class OrchestrationOutcomeValidationError(ValueError):
    """Raised when orchestration slot outcome or failure invariants are violated."""


class OrchestrationSlotExecutionError(Exception):
    """Public typed orchestration slot execution failure projected as slot FAILURE."""

    __slots__ = ("code", "message")

    def __init__(self, *, code: str, message: str) -> None:
        if not code or not code.strip():
            raise ValueError("OrchestrationSlotExecutionError code must be non-empty")
        if not message or not message.strip():
            raise ValueError("OrchestrationSlotExecutionError message must be non-empty")
        if code != code.strip():
            raise ValueError(
                "OrchestrationSlotExecutionError code must not contain leading or trailing whitespace"
            )
        if message != message.strip():
            raise ValueError(
                "OrchestrationSlotExecutionError message must not contain leading or trailing whitespace"
            )
        self.code = code.strip()
        self.message = message.strip()
        super().__init__(self.message)


class OrchestrationSlotStatus(str, Enum):
    """Per-slot orchestration outcome status."""

    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class OrchestrationSlotFailure:
    """Typed orchestration slot failure."""

    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.code or not self.code.strip():
            raise OrchestrationOutcomeValidationError(
                "orchestration failure code must be non-empty"
            )
        if self.code != self.code.strip():
            raise OrchestrationOutcomeValidationError(
                "orchestration failure code must not contain leading or trailing whitespace"
            )
        if not self.message or not self.message.strip():
            raise OrchestrationOutcomeValidationError(
                "orchestration failure message must be non-empty"
            )
        if self.message != self.message.strip():
            raise OrchestrationOutcomeValidationError(
                "orchestration failure message must not contain leading or trailing whitespace"
            )


@dataclass(frozen=True, slots=True)
class OrchestrationSlot(Generic[PayloadT]):
    """One typed orchestration slot with explicit identity and payload."""

    slot_id: OrchestrationSlotId
    payload: PayloadT
    depends_on: tuple[OrchestrationSlotId, ...] = ()


@dataclass(frozen=True, slots=True)
class OrchestrationTopology(Generic[PayloadT]):
    """Consumer-defined orchestration topology with typed slot payloads."""

    slots: tuple[OrchestrationSlot[PayloadT], ...]


@dataclass(frozen=True, slots=True)
class OrchestrationSchedulingPolicy:
    """Bounded scheduling policy enforced by canonical Nexus scheduling."""

    max_concurrency: int | None = None


@dataclass(frozen=True, slots=True)
class OrchestrationSlotOutcome(Generic[ResultT]):
    """Typed per-slot orchestration outcome.

    SUCCESS: ``failure`` must be absent; ``result`` may be ``None``.
    FAILURE: ``failure`` required; ``result`` must be absent.
    SKIPPED: ``result`` must be absent; ``failure`` may carry an optional skip reason.
    """

    slot_id: OrchestrationSlotId
    status: OrchestrationSlotStatus
    result: ResultT | None = None
    failure: OrchestrationSlotFailure | None = None

    def __post_init__(self) -> None:
        if self.status is OrchestrationSlotStatus.SUCCESS:
            if self.failure is not None:
                raise OrchestrationOutcomeValidationError(
                    "SUCCESS orchestration outcome must not include failure"
                )
            return
        if self.status is OrchestrationSlotStatus.FAILURE:
            if self.failure is None:
                raise OrchestrationOutcomeValidationError(
                    "FAILURE orchestration outcome must include failure"
                )
            if self.result is not None:
                raise OrchestrationOutcomeValidationError(
                    "FAILURE orchestration outcome must not include result"
                )
            return
        if self.status is OrchestrationSlotStatus.SKIPPED and self.result is not None:
            raise OrchestrationOutcomeValidationError(
                "SKIPPED orchestration outcome must not include result"
            )


@dataclass(frozen=True, slots=True)
class OrchestrationResult(Generic[ResultT]):
    """Deterministic aggregate orchestration result in submission order."""

    outcomes: tuple[OrchestrationSlotOutcome[ResultT], ...]


OrchestrationSlotPayloadMap: TypeAlias = Mapping[OrchestrationSlotId, PayloadT]


def validate_orchestration_slot_id(value: object) -> OrchestrationSlotId:
    if type(value) is not str:
        raise TypeError(f"OrchestrationSlotId must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError("OrchestrationSlotId must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError("OrchestrationSlotId must not contain leading or trailing whitespace")
    return OrchestrationSlotId(value.strip())


def validate_orchestration_scheduling_policy(
    policy: OrchestrationSchedulingPolicy,
) -> OrchestrationSchedulingPolicy:
    if policy.max_concurrency is not None and policy.max_concurrency < 1:
        raise OrchestrationSchedulingPolicyValidationError(
            "max_concurrency must be >= 1 when provided"
        )
    return policy


def validate_orchestration_topology(
    topology: OrchestrationTopology[PayloadT],
) -> OrchestrationTopology[PayloadT]:
    if not topology.slots:
        raise OrchestrationTopologyValidationError("orchestration topology must contain slots")

    seen: set[OrchestrationSlotId] = set()
    for slot in topology.slots:
        slot_id = validate_orchestration_slot_id(slot.slot_id)
        if slot_id in seen:
            raise OrchestrationTopologyValidationError(
                f"duplicate orchestration slot id: {slot_id!r}"
            )
        seen.add(slot_id)

    for slot in topology.slots:
        for dependency in slot.depends_on:
            dependency_id = validate_orchestration_slot_id(dependency)
            if dependency_id not in seen:
                raise OrchestrationTopologyValidationError(
                    f"missing orchestration slot dependency: {dependency_id!r}"
                )

    _validate_orchestration_topology_acyclic(topology)
    return topology


def orchestration_slot_order(
    topology: OrchestrationTopology[PayloadT],
) -> tuple[OrchestrationSlotId, ...]:
    validate_orchestration_topology(topology)
    return tuple(slot.slot_id for slot in topology.slots)


def resolve_effective_orchestration_concurrency(
    platform_limit: int | None,
    submission_limit: int | None,
) -> int | None:
    """Resolve per-submission concurrency without mutating shared scheduler state."""
    if platform_limit is None and submission_limit is None:
        return None
    if platform_limit is None:
        return submission_limit
    if submission_limit is None:
        return platform_limit
    return min(platform_limit, submission_limit)


def build_orchestration_result(
    topology: OrchestrationTopology[PayloadT],
    *,
    outcomes_by_slot: Mapping[OrchestrationSlotId, OrchestrationSlotOutcome[ResultT]],
) -> OrchestrationResult[ResultT]:
    validate_orchestration_topology(topology)
    expected_slot_ids = {slot.slot_id for slot in topology.slots}
    actual_slot_ids = set(outcomes_by_slot.keys())

    missing = sorted(expected_slot_ids - actual_slot_ids, key=str)
    if missing:
        raise OrchestrationTopologyValidationError(
            f"missing orchestration outcomes: {[str(slot_id) for slot_id in missing]!r}"
        )
    extra = sorted(actual_slot_ids - expected_slot_ids, key=str)
    if extra:
        raise OrchestrationTopologyValidationError(
            f"extra orchestration outcomes: {[str(slot_id) for slot_id in extra]!r}"
        )

    ordered: list[OrchestrationSlotOutcome[ResultT]] = []
    for slot in topology.slots:
        outcome = outcomes_by_slot[slot.slot_id]
        if outcome.slot_id != slot.slot_id:
            raise OrchestrationTopologyValidationError(
                "orchestration outcome slot_id does not match topology slot"
            )
        ordered.append(outcome)
    return OrchestrationResult(outcomes=tuple(ordered))


def _validate_orchestration_topology_acyclic(
    topology: OrchestrationTopology[PayloadT],
) -> None:
    slot_ids = {slot.slot_id for slot in topology.slots}
    in_degree = {slot_id: 0 for slot_id in slot_ids}
    dependents: dict[OrchestrationSlotId, list[OrchestrationSlotId]] = {
        slot_id: [] for slot_id in slot_ids
    }
    for slot in topology.slots:
        for dependency in slot.depends_on:
            in_degree[slot.slot_id] += 1
            dependents[dependency].append(slot.slot_id)

    ready = [slot_id for slot_id, degree in in_degree.items() if degree == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                ready.append(dependent)

    if visited != len(slot_ids):
        raise OrchestrationTopologyValidationError(
            "orchestration topology contains a dependency cycle"
        )


class OrchestrationSlotExecutor(Protocol[PayloadT, ResultT]):
    """Injected typed child-work executor for one orchestration slot."""

    async def execute_slot(
        self,
        *,
        slot_id: OrchestrationSlotId,
        payload: PayloadT,
    ) -> ResultT:
        ...


class OrchestrationSlotContinuationExecutor(Protocol[PayloadT, ResultT]):
    """Optional capability for exact resumed slot execution after governed pause."""

    async def continue_slot(
        self,
        *,
        slot_id: OrchestrationSlotId,
        payload: PayloadT,
    ) -> ResultT:
        ...


@dataclass(frozen=True, slots=True)
class OrchestrationSlotContinuationRequest:
    """Identity-bound request to continue one previously blocked orchestration slot."""

    execution_id: OrchestrationTopologyExecutionId
    slot_id: OrchestrationSlotId
    correlation_id: str

    def __post_init__(self) -> None:
        if not str(self.execution_id).strip():
            raise ValueError("execution_id must be non-empty")
        validate_orchestration_slot_id(self.slot_id)
        if not self.correlation_id or not self.correlation_id.strip():
            raise ValueError("correlation_id must be non-empty")
        if self.correlation_id != self.correlation_id.strip():
            raise ValueError("correlation_id must not contain leading or trailing whitespace")


class OrchestrationSlotContinuationError(ValueError):
    """Fail-closed orchestration slot continuation validation error."""

    __slots__ = ("code",)

    def __init__(self, message: str, *, code: str) -> None:
        if not code or not code.strip():
            raise ValueError("OrchestrationSlotContinuationError code must be non-empty")
        self.code = code.strip()
        super().__init__(message)


class OrchestrationTopologySubmissionPort(Protocol[PayloadT, ResultT]):
    """Canonical dynamic topology submission entry point."""

    async def submit(
        self,
        topology: OrchestrationTopology[PayloadT],
        scheduling_policy: OrchestrationSchedulingPolicy,
        slot_executor: OrchestrationSlotExecutor[PayloadT, ResultT],
    ) -> OrchestrationResult[ResultT]:
        ...


class OrchestrationTopologyContinuationPort(Protocol[PayloadT, ResultT]):
    """Canonical exact-slot continuation within a prior topology execution context."""

    def register_governed_continuation_slots(
        self,
        execution_id: OrchestrationTopologyExecutionId,
        slot_ids: tuple[OrchestrationSlotId, ...],
    ) -> None:
        ...

    async def continue_slot(
        self,
        request: OrchestrationSlotContinuationRequest,
        *,
        slot_continuation_executor: OrchestrationSlotContinuationExecutor[PayloadT, ResultT],
    ) -> OrchestrationResult[ResultT]:
        ...


def mint_orchestration_topology_execution_id(
    *,
    host_task_id: str,
    topology: OrchestrationTopology[PayloadT],
) -> OrchestrationTopologyExecutionId:
    """Derive stable in-process topology execution identity from active execution facts."""
    from intergrax.contracts.execution_identity import require_active_execution_identity

    if not host_task_id or not host_task_id.strip():
        raise ValueError("host_task_id must be non-empty")
    validate_orchestration_topology(topology)
    run_id, attempt_id = require_active_execution_identity()
    slot_fingerprint = "|".join(str(slot.slot_id) for slot in topology.slots)
    return OrchestrationTopologyExecutionId(
        f"{run_id}:{attempt_id}:{host_task_id.strip()}:{slot_fingerprint}"
    )
