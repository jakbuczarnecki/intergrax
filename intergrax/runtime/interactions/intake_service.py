# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Inbound interaction intake → Task (+ optional Nexus execution) (§18)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol, runtime_checkable

from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.interactions.factory import create_interaction_adapter, intake_payload_to_task
from intergrax.runtime.interactions.http_intake import parse_inbound_http_body
from intergrax.runtime.interactions.metadata_keys import INTERACTION_CHANNEL_KEY
from intergrax.runtime.interactions.task_executor import NexusLoopTaskExecutor, TaskExecutor
from intergrax.runtime.interactions.verification.contract import InboundRequestVerifier, NullInboundRequestVerifier
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskResult


@runtime_checkable
class TaskPreparationExecutor(TaskExecutor, Protocol):
    def prepare(self, task: Task) -> Task:
        ...


@dataclass(frozen=True)
class InteractionIntakeResult:
    task: Task
    executed: bool
    result: Optional[TaskResult] = None


class InteractionIntakeService:
    """Parse vendor payloads (Slack / Teams / lab JSON) and optionally run NexusLoop."""

    def __init__(
        self,
        *,
        adapter: Optional[InteractionAdapter] = None,
        verifier: Optional[InboundRequestVerifier] = None,
        nexus_loop: Optional[NexusLoop] = None,
        task_executor: Optional[TaskExecutor] = None,
        task_enricher: Optional[Callable[[Task], Task]] = None,
    ) -> None:
        self._adapter = adapter or create_interaction_adapter()
        self._verifier = verifier or NullInboundRequestVerifier()
        self._nexus_loop = nexus_loop
        self._task_executor = task_executor
        self._task_enricher = task_enricher

    def _resolve_executor(self) -> TaskExecutor | None:
        if self._task_executor is not None:
            return self._task_executor
        if self._nexus_loop is not None:
            return NexusLoopTaskExecutor(self._nexus_loop)
        return None

    def _prepare_task(self, task: Task) -> Task:
        if self._task_executor is not None and isinstance(self._task_executor, TaskPreparationExecutor):
            return self._task_executor.prepare(task)
        if self._task_enricher is not None:
            return self._task_enricher(task)
        return task

    async def intake_http(
        self,
        *,
        headers: Mapping[str, str],
        body: bytes,
        content_type: str,
        tenant_id: str,
        execute: bool = False,
    ) -> InteractionIntakeResult:
        self._verifier.verify(headers=headers, body=body)
        payload = parse_inbound_http_body(content_type=content_type, body=body)
        return await self.intake_payload(payload, tenant_id=tenant_id, execute=execute)

    async def intake_payload(
        self,
        payload: dict,
        *,
        tenant_id: str,
        execute: bool = False,
    ) -> InteractionIntakeResult:
        task = intake_payload_to_task(
            payload,
            tenant_id=tenant_id,
            adapter=self._adapter,
        )
        if not execute:
            task = self._prepare_task(task)
            return InteractionIntakeResult(task=task, executed=False)
        executor = self._resolve_executor()
        if executor is None:
            raise ValueError("Task executor is not configured for execute=true")
        if isinstance(executor, TaskPreparationExecutor) and hasattr(executor, "execute_prepared"):
            prepared = executor.prepare(task)
            result = await executor.execute_prepared(prepared)  # type: ignore[attr-defined]
            return InteractionIntakeResult(task=prepared, executed=True, result=result)
        if self._task_enricher is not None:
            task = self._task_enricher(task)
        result = await executor.execute(task)
        return InteractionIntakeResult(task=task, executed=True, result=result)

    @staticmethod
    def interaction_channel(task: Task) -> str:
        return str(task.metadata.get(INTERACTION_CHANNEL_KEY) or "")
