# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Run-scoped execution acceptance position and as-of boundary (TRACE-ASOF-1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import RunId, validate_run_id


@dataclass(frozen=True, slots=True, order=True)
class ExecutionEventPosition:
    """
    Authoritative run-scoped acceptance position (one tenant + run stream).

    Assigned by persistence append — producers MUST NOT mint this.
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


@runtime_checkable
class PositionedExecutionEvent(Protocol):
    """Minimal positioned-event surface for as-of boundary checks."""

    @property
    def run_id(self) -> RunId: ...

    @property
    def position(self) -> ExecutionEventPosition: ...


@dataclass(frozen=True, slots=True)
class AsOfBoundary:
    """Inclusive execution-history boundary for one run."""

    run_id: RunId
    position: ExecutionEventPosition

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "position", validate_execution_event_position(self.position))

    def includes(self, positioned: PositionedExecutionEvent) -> bool:
        if positioned.run_id != self.run_id:
            raise ValueError(
                "AsOfBoundary run_id does not match positioned event run_id"
            )
        return positioned.position <= self.position
