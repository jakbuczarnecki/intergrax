# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Positioned runtime evidence row (TRACE-ASOF-1 read model)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_event_position import (
    AsOfBoundary,
    ExecutionEventPosition,
)
from intergrax.contracts.execution_identity import EventId, RunId
from intergrax.contracts.runtime_event import RuntimeEvent


@dataclass(frozen=True, slots=True)
class PositionedRuntimeEvent:
    """Accepted ``RuntimeEvent`` with persistence-owned execution position."""

    event: RuntimeEvent
    position: ExecutionEventPosition

    @property
    def run_id(self) -> RunId:
        return self.event.run_id

    @property
    def event_id(self) -> EventId:
        return self.event.event_id


def as_of_boundary_for_positioned(positioned: PositionedRuntimeEvent) -> AsOfBoundary:
    return AsOfBoundary(run_id=positioned.run_id, position=positioned.position)


__all__ = [
    "PositionedRuntimeEvent",
    "as_of_boundary_for_positioned",
]
