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


class OrchestrationTopologyValidationError(ValueError):
    """Raised when an orchestration topology fails structural validation."""


class OrchestrationSchedulingPolicyValidationError(ValueError):
    """Raised when orchestration scheduling policy is invalid."""


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
    """Typed per-slot orchestration outcome."""

    slot_id: OrchestrationSlotId
    status: OrchestrationSlotStatus
    result: ResultT | None = None
    failure: OrchestrationSlotFailure | None = None


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


def build_orchestration_result(
    topology: OrchestrationTopology[PayloadT],
    *,
    outcomes_by_slot: Mapping[OrchestrationSlotId, OrchestrationSlotOutcome[ResultT]],
) -> OrchestrationResult[ResultT]:
    validate_orchestration_topology(topology)
    ordered: list[OrchestrationSlotOutcome[ResultT]] = []
    for slot in topology.slots:
        outcome = outcomes_by_slot.get(slot.slot_id)
        if outcome is None:
            raise OrchestrationTopologyValidationError(
                f"missing orchestration outcome for slot: {slot.slot_id!r}"
            )
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


class OrchestrationTopologySubmissionPort(Protocol[PayloadT, ResultT]):
    """Canonical dynamic topology submission entry point."""

    async def submit(
        self,
        topology: OrchestrationTopology[PayloadT],
        scheduling_policy: OrchestrationSchedulingPolicy,
        slot_executor: OrchestrationSlotExecutor[PayloadT, ResultT],
    ) -> OrchestrationResult[ResultT]:
        ...
