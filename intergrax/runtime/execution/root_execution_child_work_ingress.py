# © Artur Czarnecki. All rights reserved.

"""Cooperative root catalog child work ingress (drain on supervised root task only)."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass

from intergrax.contracts.execution_identity import (
    peek_active_execution_id,
    peek_active_execution_identity,
    peek_active_execution_task_id,
)
from intergrax.contracts.root_execution_child_work_ingress import (
    RootCatalogChildWorkTerminal,
    RootChildWorkIngressRefusal,
    RootExecutionChildWorkIngressPort,
)
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.runtime.execution.catalog_tool_child_work_port import (
    CatalogToolChildAdmissionHook,
    CatalogToolChildExecutionPort,
    CatalogToolChildInvokeWork,
)
from intergrax.runtime.governance.active_execution_authority import (
    peek_active_execution_authority,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    peek_active_execution_governance_identity,
)
from intergrax.runtime.governance.governance_identity_projection import (
    validate_governance_identity_projection,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task_state import task_state_requires_authoritative_exposure
from pydantic import BaseModel

_DEFAULT_MAX_QUEUE = 32
_DEFAULT_SUBMIT_TIMEOUT_S = 120.0


@dataclass(frozen=True, slots=True)
class RootCatalogChildWorkTicket:
    task_id: TaskId
    run_id: RunId
    caller: RequestIdentity
    work: CatalogToolChildInvokeWork
    admission_hooks: tuple[CatalogToolChildAdmissionHook, ...] = ()


@dataclass(slots=True)
class _PendingIngress:
    ticket: RootCatalogChildWorkTicket
    future: asyncio.Future[RootCatalogChildWorkTerminal]


class RootExecutionChildWorkIngress(RootExecutionChildWorkIngressPort):
    """Per-process ingress registry; one active binding per supervised root task_id."""

    __slots__ = (
        "_bound_task_id",
        "_catalog_child_execution",
        "_max_queue",
        "_pending",
        "_ready",
        "_refuse_new",
        "_serialize_tool_result",
    )

    def __init__(
        self,
        *,
        catalog_child_execution: CatalogToolChildExecutionPort,
        max_queue: int = _DEFAULT_MAX_QUEUE,
        serialize_tool_result: object | None = None,
    ) -> None:
        self._catalog_child_execution = catalog_child_execution
        self._max_queue = max_queue
        self._pending: deque[_PendingIngress] = deque()
        self._ready = asyncio.Event()
        self._bound_task_id: TaskId | None = None
        self._refuse_new = False
        self._serialize_tool_result = serialize_tool_result

    def bind_supervised_task(self, task_id: TaskId) -> None:
        self._bound_task_id = task_id

    def unbind_supervised_task(self, task_id: TaskId) -> None:
        if self._bound_task_id == task_id:
            self._bound_task_id = None
            self.refuse_new_submissions()

    def refuse_new_submissions(self) -> None:
        self._refuse_new = True
        while self._pending:
            pending = self._pending.popleft()
            if not pending.future.done():
                pending.future.set_result(
                    RootCatalogChildWorkTerminal(
                        success=False,
                        payload={
                            "error": "execution_admission_denied",
                            "detail": "ingress shutdown",
                        },
                        refusal=RootChildWorkIngressRefusal.SHUTDOWN,
                    ),
                )
        self._ready.clear()

    async def submit_catalog_child_work(
        self,
        ticket: RootCatalogChildWorkTicket,
        *,
        wait_timeout_s: float = _DEFAULT_SUBMIT_TIMEOUT_S,
    ) -> RootCatalogChildWorkTerminal:
        if self._refuse_new:
            return RootCatalogChildWorkTerminal(
                success=False,
                payload={
                    "error": "execution_admission_denied",
                    "detail": "ingress not accepting work",
                },
                refusal=RootChildWorkIngressRefusal.SHUTDOWN,
            )
        if self._bound_task_id is None or str(self._bound_task_id) != str(ticket.task_id):
            return RootCatalogChildWorkTerminal(
                success=False,
                payload={
                    "error": "execution_admission_denied",
                    "detail": "no supervised root binding for task",
                },
                refusal=RootChildWorkIngressRefusal.NOT_SUPERVISED,
            )
        if len(self._pending) >= self._max_queue:
            return RootCatalogChildWorkTerminal(
                success=False,
                payload={"error": "execution_admission_denied", "detail": "ingress queue full"},
                refusal=RootChildWorkIngressRefusal.QUEUE_FULL,
            )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RootCatalogChildWorkTerminal] = loop.create_future()
        self._pending.append(_PendingIngress(ticket=ticket, future=future))
        self.notify_ready()
        try:
            return await asyncio.wait_for(future, timeout=wait_timeout_s)
        except TimeoutError:
            self._drop_pending_future(future)
            return RootCatalogChildWorkTerminal(
                success=False,
                payload={
                    "error": "execution_admission_denied",
                    "detail": "ingress submit timed out waiting for root drain",
                },
                refusal=RootChildWorkIngressRefusal.SUBMIT_TIMEOUT,
            )

    def _drop_pending_future(
        self,
        future: asyncio.Future[RootCatalogChildWorkTerminal],
    ) -> None:
        self._pending = deque(p for p in self._pending if p.future is not future)
        if not self._pending:
            self._ready.clear()

    def notify_ready(self) -> None:
        self._ready.set()

    async def wait_ready(self) -> None:
        await self._ready.wait()

    def has_ready_work(self) -> bool:
        return bool(self._pending) and self._ready.is_set()

    async def drain_one(self) -> bool:
        if not self._pending:
            self._ready.clear()
            return False
        if not self._parent_context_matches_supervised_root():
            pending = self._pending.popleft()
            if not pending.future.done():
                pending.future.set_result(
                    RootCatalogChildWorkTerminal(
                        success=False,
                        payload={
                            "error": "execution_admission_denied",
                            "detail": "drain without active parent execution context",
                        },
                        refusal=RootChildWorkIngressRefusal.NOT_SUPERVISED,
                    ),
                )
            if not self._pending:
                self._ready.clear()
            return True
        pending = self._pending[0]
        ticket = pending.ticket
        binding = await ActiveTaskRegistry.get(ticket.task_id)
        if binding is None:
            self._pending.popleft()
            if not pending.future.done():
                pending.future.set_result(
                    RootCatalogChildWorkTerminal(
                        success=False,
                        payload={
                            "error": "missing_execution_binding",
                            "detail": "no active execution binding",
                        },
                    ),
                )
            if not self._pending:
                self._ready.clear()
            return True
        if task_state_requires_authoritative_exposure(binding.task.state):
            self._pending.popleft()
            if not pending.future.done():
                pending.future.set_result(
                    RootCatalogChildWorkTerminal(
                        success=False,
                        payload={
                            "error": "execution_admission_denied",
                            "detail": "parent execution terminal",
                        },
                    ),
                )
            if not self._pending:
                self._ready.clear()
            return True
        self._pending.popleft()
        try:
            tool_result = await self._catalog_child_execution.execute(
                work=ticket.work,
                admission_hooks=ticket.admission_hooks,
            )
            payload = self._serialize_result(tool_result, tool_id=ticket.work.tool_id)
            terminal = RootCatalogChildWorkTerminal(success=True, payload=payload)
        except Exception as exc:  # noqa: BLE001 — bounded MCP-style error surface
            terminal = RootCatalogChildWorkTerminal(
                success=False,
                payload={"error": "internal_error", "detail": str(exc)},
            )
        if not pending.future.done():
            pending.future.set_result(terminal)
        if not self._pending:
            self._ready.clear()
        else:
            self.notify_ready()
        return True

    def _parent_context_matches_supervised_root(self) -> bool:
        if self._bound_task_id is None:
            return False
        active_identity = peek_active_execution_identity()
        parent_execution_id = peek_active_execution_id()
        if active_identity is None or parent_execution_id is None:
            return False
        active_task_id = peek_active_execution_task_id()
        if active_task_id is None:
            return False
        if str(active_task_id) != str(self._bound_task_id):
            return False
        if peek_active_execution_authority() is None:
            return False
        governance_identity = peek_active_execution_governance_identity()
        if governance_identity is None:
            return False
        try:
            validate_governance_identity_projection(
                governance_identity,
                tenant_id=governance_identity.tenant_id,
            )
        except (TypeError, ValueError):
            return False
        return True

    def _serialize_result(
        self,
        result: ToolExecutionResult[BaseModel],
        *,
        tool_id: str,
    ) -> dict[str, object]:
        if self._serialize_tool_result is not None:
            serialized = self._serialize_tool_result(result)
            if isinstance(serialized, dict):
                return serialized
        payload: dict[str, object] = {
            "tool_id": tool_id,
            "success": result.success,
            "effect_certainty": result.effect_certainty.value,
        }
        if result.success and result.output is not None:
            payload["output"] = result.output.model_dump(mode="json")
        elif not result.success and result.error is not None:
            payload["error_code"] = result.error.error_code
            payload["error_message"] = result.error.error_message
        return payload


async def execute_delegate_with_root_child_work_ingress(
    delegate: object,
    request: object,
    ingress: RootExecutionChildWorkIngressPort | None,
) -> object:
    """Interleave cooperative ingress drain with root delegate execution (same asyncio.Task)."""
    execute = getattr(delegate, "execute", None)
    if execute is None:
        raise TypeError("delegate must provide execute()")
    if ingress is None:
        return await execute(request)

    main = asyncio.create_task(execute(request))
    ready_waiter = asyncio.create_task(ingress.wait_ready())
    try:
        while not main.done():
            wait_set: set[asyncio.Task[object]] = {main}
            if not ready_waiter.done():
                wait_set.add(ready_waiter)
            done, _pending = await asyncio.wait(
                wait_set,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if main in done:
                break
            if ingress.has_ready_work():
                await ingress.drain_one()
            if ready_waiter.done():
                ready_waiter = asyncio.create_task(ingress.wait_ready())
        while ingress.has_ready_work():
            await ingress.drain_one()
        return await main
    finally:
        if not main.done():
            main.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await main
        if not ready_waiter.done():
            ready_waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ready_waiter


__all__ = [
    "RootCatalogChildWorkTicket",
    "RootExecutionChildWorkIngress",
    "execute_delegate_with_root_child_work_ingress",
]
