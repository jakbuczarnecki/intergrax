# © Artur Czarnecki. All rights reserved.

"""MCP transport cancellation → governed cooperative task cancel (TLX-TL11 CANCEL-A)."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.applications._shared.mcp_harness_principal import _starlette_request_from_fastmcp
from intergrax.applications._shared.task_control import governed_cancel_active_task
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task

if TYPE_CHECKING:
    from starlette.requests import Request

MCP_TRANSPORT_CANCEL_REASON = "mcp_transport_cancel"
MCP_TRANSPORT_CANCEL_MUTATION_PREFIX = "mcp-transport-cancel"

# MCP transport-layer observation scheduling only (not root Execution deadline authority).
_ACTIVE_BINDING_WAIT_S = 5.0
_ACTIVE_BINDING_POLL_INTERVAL_S = 0.05
_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S = 30.0


@dataclass(frozen=True, slots=True)
class McpHostExecutionTransportCancelContext:
    """Sanctioned admission inputs for one MCP ``run_agent`` transport cancel request."""

    mutation_boundary: ControlPlaneMutationAuthorizationBoundary
    caller_request_identity: RequestIdentity


def resolve_mcp_task_control_mutation_boundary_from_request(
    request: Request | None = None,
) -> ControlPlaneMutationAuthorizationBoundary | None:
    """Read host task-control boundary from the MCP HTTP sub-app state (if wired)."""
    if request is None:
        request = _starlette_request_from_fastmcp()
    if request is None:
        return None
    return getattr(request.app.state, "task_control_mutation_boundary", None)


async def _client_disconnected(request: Request | None) -> bool:
    if request is None:
        return False
    try:
        return await request.is_disconnected()
    except Exception:
        return False


async def _wait_for_active_binding(task: Task) -> bool:
    deadline = asyncio.get_running_loop().time() + _ACTIVE_BINDING_WAIT_S
    while asyncio.get_running_loop().time() < deadline:
        binding = await ActiveTaskRegistry.get(task.task_id)
        if binding is not None:
            return True
        if _ACTIVE_BINDING_POLL_INTERVAL_S > 0:
            await asyncio.sleep(_ACTIVE_BINDING_POLL_INTERVAL_S)
    return False


async def request_governed_cancel_for_bound_mcp_task(
    *,
    task: Task,
    principal: RequestIdentity,
    mutation_boundary: ControlPlaneMutationAuthorizationBoundary,
    reason: str = MCP_TRANSPORT_CANCEL_REASON,
) -> bool:
    """Request cooperative cancel for the active binding of *task* (idempotent admission attempt)."""
    binding = await ActiveTaskRegistry.get(task.task_id)
    if binding is None:
        return False
    mutation_id = f"{MCP_TRANSPORT_CANCEL_MUTATION_PREFIX}:{task.task_id}"
    result = await governed_cancel_active_task(
        task_id=str(task.task_id),
        run_id=str(binding.run_id),
        mutation_id=mutation_id,
        principal=principal,
        mutation_boundary=mutation_boundary,
        reason=reason,
    )
    return result.accepted


async def _maybe_request_transport_cancel(
    *,
    task: Task,
    cancel_ctx: McpHostExecutionTransportCancelContext,
    cancel_lock: asyncio.Lock,
    cancel_admission_accepted: list[bool],
) -> None:
    async with cancel_lock:
        if cancel_admission_accepted[0]:
            return
    if not await _wait_for_active_binding(task):
        return
    async with cancel_lock:
        if cancel_admission_accepted[0]:
            return
        accepted = await request_governed_cancel_for_bound_mcp_task(
            task=task,
            principal=cancel_ctx.caller_request_identity,
            mutation_boundary=cancel_ctx.mutation_boundary,
        )
        if accepted:
            cancel_admission_accepted[0] = True


async def _await_host_task_bounded_after_transport_waiter_lost(
    host_coro: asyncio.Task[object],
    *,
    timeout_s: float,
    governed_cancel_admitted: bool,
) -> None:
    """
    End MCP transport waiter scope with a hard wall-clock bound.

    Canonical Execution remains owned by ``HostTaskExecutionPort.execute`` inside *host_coro*.
    When governed cancel was admitted (OWN-A), do not cancel the host task merely to end the HTTP waiter.
    """
    if host_coro.done():
        with contextlib.suppress(asyncio.CancelledError, Exception):
            host_coro.result()
        return
    try:
        await asyncio.wait_for(asyncio.shield(host_coro), timeout=timeout_s)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    if host_coro.done():
        with contextlib.suppress(asyncio.CancelledError, Exception):
            host_coro.result()
        return
    if governed_cancel_admitted:
        return
    host_coro.cancel()
    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(host_coro), timeout=timeout_s)


async def run_host_execution_with_mcp_transport_cancel(
    host_coro: asyncio.Task[object],
    *,
    task: Task,
    cancel_ctx: McpHostExecutionTransportCancelContext,
) -> object:
    """
    Await canonical host execution while observing MCP transport cancellation.

    Does not mutate terminal state directly; routes cancel through ``governed_cancel_active_task``.
    """
    cancel_admission_accepted: list[bool] = [False]
    cancel_lock = asyncio.Lock()
    transport_cancel_observed = False
    request = _starlette_request_from_fastmcp()

    async def _monitor_transport() -> None:
        nonlocal transport_cancel_observed
        try:
            while not host_coro.done():
                if not transport_cancel_observed and await _client_disconnected(request):
                    transport_cancel_observed = True
                if transport_cancel_observed:
                    await _maybe_request_transport_cancel(
                        task=task,
                        cancel_ctx=cancel_ctx,
                        cancel_lock=cancel_lock,
                        cancel_admission_accepted=cancel_admission_accepted,
                    )
                if _ACTIVE_BINDING_POLL_INTERVAL_S > 0:
                    await asyncio.sleep(_ACTIVE_BINDING_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            if not host_coro.done():
                transport_cancel_observed = True
                await _maybe_request_transport_cancel(
                    task=task,
                    cancel_ctx=cancel_ctx,
                    cancel_lock=cancel_lock,
                    cancel_admission_accepted=cancel_admission_accepted,
                )
            raise

    monitor = asyncio.create_task(_monitor_transport())
    transport_waiter_lost = False
    host_result_after_waiter_lost: object | None = None
    try:
        return await asyncio.shield(host_coro)
    except asyncio.CancelledError:
        transport_waiter_lost = True
        transport_cancel_observed = True
        await _maybe_request_transport_cancel(
            task=task,
            cancel_ctx=cancel_ctx,
            cancel_lock=cancel_lock,
            cancel_admission_accepted=cancel_admission_accepted,
        )
        if host_coro.done():
            host_result_after_waiter_lost = host_coro.result()
    finally:
        monitor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor
        if transport_waiter_lost or not host_coro.done():
            await _await_host_task_bounded_after_transport_waiter_lost(
                host_coro,
                timeout_s=_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S,
                governed_cancel_admitted=cancel_admission_accepted[0],
            )
        if host_coro.done() and host_result_after_waiter_lost is None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                host_result_after_waiter_lost = host_coro.result()
    if transport_waiter_lost:
        if host_result_after_waiter_lost is not None:
            return host_result_after_waiter_lost
        raise asyncio.CancelledError()
    raise AssertionError("unreachable_host_transport_cancel_waiter_state")


__all__ = [
    "MCP_TRANSPORT_CANCEL_MUTATION_PREFIX",
    "MCP_TRANSPORT_CANCEL_REASON",
    "McpHostExecutionTransportCancelContext",
    "request_governed_cancel_for_bound_mcp_task",
    "resolve_mcp_task_control_mutation_boundary_from_request",
    "run_host_execution_with_mcp_transport_cancel",
]
