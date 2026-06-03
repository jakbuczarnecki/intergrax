# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Runtime event publishing for NexusLoop (Phase Q-N.1)."""

from __future__ import annotations

from typing import Callable, Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader
from intergrax.runtime.observability.modality_metrics import build_task_completed_modality_payload
from intergrax.runtime.task.task import Task


class NexusRuntimeEventPublisher:
    """Publishes canonical ``RuntimeEvent`` instances on the task event bus."""

    def __init__(
        self,
        event_bus: RuntimeEventBus,
        *,
        current_task: Callable[[], Optional[Task]],
        trace_reader: RunTraceReader | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._current_task = current_task
        self._trace_reader = trace_reader

    async def publish(self, event: RuntimeEvent, *, task: Optional[Task] = None) -> None:
        scoped_task = task or self._current_task()
        if scoped_task is not None and not event.tenant_id:
            event = event.model_copy(update={"tenant_id": scoped_task.tenant_id})
        await self._event_bus.publish(event)

    async def publish_terminal(self, task: Task) -> None:
        base = runtime_event_from_task_state(task, run_id=task.task_id, message="task terminal")
        modality_payload = self._modality_payload_for_task(task)
        if modality_payload is not None:
            merged = {**base.payload, **modality_payload}
            base = base.model_copy(update={"payload": merged})
        await self.publish(base, task=task)

    def _modality_payload_for_task(self, task: Task) -> dict[str, object] | None:
        if self._trace_reader is None:
            return None
        try:
            persisted = self._trace_reader.read_run(task.task_id, task.tenant_id)
        except (KeyError, ValueError):
            return None
        return build_task_completed_modality_payload(persisted.events)

    async def publish_from_task_state(
        self,
        task: Task,
        *,
        message: str,
        event_type: RuntimeEventType,
        phase: ExecutionPhase,
        payload: Optional[dict] = None,
    ) -> None:
        base = runtime_event_from_task_state(task, run_id=task.task_id, message=message)
        update: dict = {
            "event_type": event_type,
            "phase": phase,
        }
        if payload is not None:
            update["payload"] = payload
        await self.publish(base.model_copy(update=update), task=task)
