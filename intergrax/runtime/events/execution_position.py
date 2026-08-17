# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution-history position and as-of boundary (TRACE-ASOF-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import EventId, RunId, validate_run_id


@dataclass(frozen=True, slots=True, order=True)
class ExecutionEventPosition:
    """
    Authoritative run-scoped acceptance position.

    Assigned by ``RuntimeEventPersistence.append`` — producers MUST NOT mint this.
    """

    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int or isinstance(self.value, bool) or self.value < 1:
            raise ValueError("ExecutionEventPosition must be a positive int >= 1")


def validate_execution_event_position(value: object) -> ExecutionEventPosition:
    if isinstance(value, ExecutionEventPosition):
        return value
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(
            f"ExecutionEventPosition must be int >= 1, got {type(value).__name__}"
        )
    return ExecutionEventPosition(value)


@dataclass(frozen=True, slots=True)
class PositionedRuntimeEvent:
    """Accepted ``RuntimeEvent`` with persistence-owned execution position."""

    event: "RuntimeEvent"
    position: ExecutionEventPosition

    @property
    def run_id(self) -> RunId:
        return self.event.run_id

    @property
    def event_id(self) -> EventId:
        return self.event.event_id


@dataclass(frozen=True, slots=True)
class AsOfBoundary:
    """
    Inclusive execution-history boundary for one run.

    Semantics: events with ``position <= boundary.position`` belong to the prefix.
    """

    run_id: RunId
    position: ExecutionEventPosition

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "position", validate_execution_event_position(self.position))

    def includes(self, positioned: PositionedRuntimeEvent) -> bool:
        if positioned.run_id != self.run_id:
            raise ValueError(
                "AsOfBoundary run_id does not match positioned event run_id"
            )
        return positioned.position <= self.position


def as_of_boundary_for_positioned(positioned: PositionedRuntimeEvent) -> AsOfBoundary:
    return AsOfBoundary(run_id=positioned.run_id, position=positioned.position)


from intergrax.runtime.events.runtime_event import RuntimeEvent  # noqa: E402  — cycle break
