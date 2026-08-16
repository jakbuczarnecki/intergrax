# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Emit context for HOS public APIs (OBS-EVOL-9.3 · SAR-01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_event_id,
    validate_run_id,
    validate_task_id,
)

if TYPE_CHECKING:
    from intergrax.runtime.events.event_bus import RuntimeEventBus


@dataclass(frozen=True, slots=True)
class EmitContext:
    """Correlation bundle passed to ``emit_domain_signal`` / ``emit_platform_event``."""

    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    tenant_id: str | None = None
    correlation_id: str = ""
    parent_event_id: EventId | None = None
    traceparent: str | None = None
    tracestate: str | None = None
    bus: RuntimeEventBus | None = None
    production_mode: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", validate_task_id(self.task_id))
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "attempt_id", validate_attempt_id(self.attempt_id))
        if self.parent_event_id is not None:
            object.__setattr__(
                self,
                "parent_event_id",
                validate_event_id(self.parent_event_id),
            )

    @property
    def effective_correlation_id(self) -> str:
        return self.correlation_id or self.task_id
