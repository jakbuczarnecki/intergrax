# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Runtime event publishing for NexusLoop (Phase Q-N.1)."""

from __future__ import annotations

from typing import Callable, Optional

from intergrax.contracts.execution_identity import ActiveExecutionIdentity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.trace_bridge import runtime_event_from_task_state
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunTraceReader
from intergrax.runtime.observability.journal_export import build_journal_ref_payload
from intergrax.runtime.observability.modality_metrics import build_task_completed_modality_payload
from intergrax.runtime.task.task import Task


class NexusRuntimeEventPublisher:
    """Publishes canonical ``RuntimeEvent`` instances on the task event bus."""

    def __init__(
        self,
        event_bus: RuntimeEventBus,
        *,
        current_task: Callable[[], Optional[Task]],
        execution_identity: ActiveExecutionIdentity,
        trace_reader: RunTraceReader | None = None,
        runtime_event_store: RuntimeEventPersistence | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._current_task = current_task
        self._execution_identity = execution_identity
        self._trace_reader = trace_reader
        self._runtime_event_store = runtime_event_store

    async def publish(self, event: RuntimeEvent, *, task: Optional[Task] = None) -> None:
        scoped_task = task or self._current_task()
        if scoped_task is not None:
            from intergrax.runtime.events.w3c_trace_context import inject_w3c_trace_on_event

            event = inject_w3c_trace_on_event(event, scoped_task)
            if not event.tenant_id:
                event = event.model_copy(update={"tenant_id": scoped_task.tenant_id})
        await self._event_bus.publish(event)

    async def publish_terminal(self, task: Task) -> None:
        run_id, attempt_id = self._execution_identity.require()
        base = runtime_event_from_task_state(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            message="task terminal",
        )
        terminal_payload = self._terminal_payload_for_task(task)
        if terminal_payload:
            merged = {**base.payload, **terminal_payload}
            base = base.model_copy(update={"payload": merged})
        await self.publish(base, task=task)

    def _read_persisted_run(self, task: Task) -> PersistedRun | None:
        if self._trace_reader is None:
            return None
        try:
            return self._trace_reader.read_run(task.task_id, task.tenant_id)
        except (KeyError, ValueError):
            return None

    def _terminal_payload_for_task(self, task: Task) -> dict[str, object]:
        persisted = self._read_persisted_run(task)
        if persisted is None:
            return {}
        fragments: dict[str, object] = {}
        modality_payload = build_task_completed_modality_payload(persisted.events)
        if modality_payload is not None:
            fragments.update(modality_payload)
        journal_ref = build_journal_ref_payload(
            persisted,
            runtime_store=self._runtime_event_store,
        )
        if journal_ref is not None:
            fragments["journal_ref"] = journal_ref
        return fragments

    async def publish_from_task_state(
        self,
        task: Task,
        *,
        message: str,
        event_type: RuntimeEventType,
        phase: ExecutionPhase,
        payload: Optional[dict] = None,
    ) -> None:
        run_id, attempt_id = self._execution_identity.require()
        base = runtime_event_from_task_state(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            message=message,
        )
        update: dict = {
            "event_type": event_type,
            "phase": phase,
        }
        if payload is not None:
            update["payload"] = payload
        await self.publish(base.model_copy(update=update), task=task)
