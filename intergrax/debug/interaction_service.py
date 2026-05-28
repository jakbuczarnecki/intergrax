# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Debug API: inbound interaction intake → Task (+ optional Nexus execution)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.interactions.factory import create_interaction_adapter, intake_payload_to_task
from intergrax.runtime.interactions.http_intake import parse_inbound_http_body
from intergrax.runtime.interactions.metadata_keys import INTERACTION_CHANNEL_KEY
from intergrax.runtime.interactions.verification.contract import InboundRequestVerifier, NullInboundRequestVerifier
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskResult


@dataclass(frozen=True)
class InteractionIntakeResult:
    task: Task
    executed: bool
    result: Optional[TaskResult] = None


class DebugInteractionIntakeService:
    """Laboratory intake — parses vendor payloads and optionally runs NexusLoop."""

    def __init__(
        self,
        *,
        adapter: Optional[InteractionAdapter] = None,
        verifier: Optional[InboundRequestVerifier] = None,
        nexus_loop: Optional[NexusLoop] = None,
        task_enricher: Optional[Callable[[Task], Task]] = None,
    ) -> None:
        self._adapter = adapter or create_interaction_adapter()
        self._verifier = verifier or NullInboundRequestVerifier()
        self._nexus_loop = nexus_loop
        self._task_enricher = task_enricher

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
        if self._task_enricher is not None:
            task = self._task_enricher(task)
        if not execute:
            return InteractionIntakeResult(task=task, executed=False)
        if self._nexus_loop is None:
            raise ValueError("NexusLoop is not configured for execute=true")
        result = await self._nexus_loop.handle_task(task)
        return InteractionIntakeResult(task=task, executed=True, result=result)

    @staticmethod
    def interaction_channel(task: Task) -> str:
        return str(task.metadata.get(INTERACTION_CHANNEL_KEY) or "")
