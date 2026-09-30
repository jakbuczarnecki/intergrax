# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral application host orchestration session (no Nexus escape)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.observability.qualification_runtime_trace import (
    DeferredPersistedTraceFinalize,
)


@dataclass(frozen=True, slots=True)
class HostOrchestrationTraceLifecycleControl:
    """Neutral persisted-trace finalize control for scenario/host composition roots."""

    _target: HostOrchestrationApplicationWiringTarget

    def set_hold_persisted_trace_finalize(self, hold: bool) -> None:
        self._target.set_hold_persisted_trace_finalize(hold)

    def take_deferred_persisted_trace_finalize(
        self,
    ) -> DeferredPersistedTraceFinalize | None:
        return self._target.take_deferred_persisted_trace_finalize()


@dataclass(frozen=True, slots=True)
class ApplicationHostOrchestrationSession:
    """Canonical EE-owned host orchestration artifacts exposed outside owner zone."""

    host_execution: HostTaskExecutionPort
    runtime_event_bus: RuntimeEventBus
    trace_lifecycle: HostOrchestrationTraceLifecycleControl


__all__ = [
    "ApplicationHostOrchestrationSession",
    "HostOrchestrationTraceLifecycleControl",
]
