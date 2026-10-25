# © Artur Czarnecki. All rights reserved.

"""MCP transport cancellation → governed cooperative task cancel (TLX-TL11 CANCEL-A)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.applications._shared.mcp_harness_principal import _starlette_request_from_fastmcp
from intergrax.applications._shared.mcp_transport_detachment_supervision_coupling import (
    resolve_transport_detachment_supervision_from_request,
)
from intergrax.applications._shared.task_control import governed_cancel_active_task
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.root_execution_transport_detachment_supervision import (
    RootExecutionTransportAdmissionRequest,
    RootExecutionTransportDetachmentHandoffDisposition,
    RootExecutionTransportDetachmentHandoffRequest,
    RootExecutionTransportDetachmentSupervisionPort,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task

if TYPE_CHECKING:
    from starlette.requests import Request

MCP_TRANSPORT_CANCEL_REASON = "mcp_transport_cancel"
MCP_TRANSPORT_CANCEL_MUTATION_PREFIX = "mcp-transport-cancel"


class McpHostExecutionTransportOwnershipInvariantError(RuntimeError):
    """Live canonical Execution cannot release MCP transport waiter without terminal or ACCEPTED handoff."""


class McpProtectedHostExecutionAdmissionError(RuntimeError):
    """Protected MCP HTTP execution rejected before host Execution starts."""

_REFUSED_ADMISSION_HOST_TERMINATE_S = 1.0


@dataclass(frozen=True, slots=True)
class McpProtectedHostExecutionLaunch:
    """Host asyncio work gated until canonical root supervision admission accepts."""

    host_task: asyncio.Task[object]
    execution_start_gate: asyncio.Event


def create_mcp_protected_host_execution_launch(
    execute: Callable[[], Awaitable[object]],
) -> McpProtectedHostExecutionLaunch:
    """Schedule host work that cannot run business Execution until the start gate opens."""
    execution_start_gate = asyncio.Event()

    async def _gated_execute() -> object:
        await execution_start_gate.wait()
        return await execute()

    host_task = asyncio.create_task(_gated_execute())
    return McpProtectedHostExecutionLaunch(
        host_task=host_task,
        execution_start_gate=execution_start_gate,
    )


async def _finalize_protected_mcp_root_execution_admission_caller_abort(
    *,
    supervision_port: RootExecutionTransportDetachmentSupervisionPort,
    task: Task,
    launch: McpProtectedHostExecutionLaunch,
    gate_opened: bool,
    transport_waiter_will_continue: bool,
) -> None:
    """
    Linearize admission caller abort without leaving a gated task unowned or indefinitely closed.

    When supervision was accepted and the transport waiter will continue, open the start gate
    so accountable supervised Execution can proceed. Otherwise relinquish supervision and
    terminate the gated host task (bounded, fail closed).
    """
    if gate_opened:
        return
    holds = supervision_port.holds_root_execution_supervision(
        task_id=task.task_id,
        host_asyncio_task=launch.host_task,
    )
    if holds:
        if transport_waiter_will_continue:
            launch.execution_start_gate.set()
            return
        await supervision_port.relinquish_root_execution_supervision_admission(
            task_id=task.task_id,
            host_asyncio_task=launch.host_task,
        )
    await _terminate_gated_host_task_after_refused_admission(launch.host_task)


async def _terminate_gated_host_task_after_refused_admission(
    host_task: asyncio.Task[object],
    *,
    timeout_s: float = _REFUSED_ADMISSION_HOST_TERMINATE_S,
) -> None:
    host_task.cancel()
    completed = await _bounded_observe_host_task(host_task, timeout_s=timeout_s)
    await _retrieve_host_task_outcome(host_task)
    if not completed:
        raise McpProtectedHostExecutionAdmissionError(
            "protected MCP root execution host task did not terminate after refused "
            "supervision admission"
        )


async def admit_protected_mcp_root_execution_supervision(
    *,
    supervision_port: RootExecutionTransportDetachmentSupervisionPort,
    task: Task,
    caller_request_identity: RequestIdentity,
    launch: McpProtectedHostExecutionLaunch,
    transport_waiter_will_continue: bool = True,
) -> None:
    """
    Linearize supervision acceptance before protected business Execution may start.

    On refusal, cancels the gated host task and verifies bounded termination (fail closed).
    """
    gate_opened = False
    try:
        if task.tenant_id != caller_request_identity.tenant_id:
            await _terminate_gated_host_task_after_refused_admission(launch.host_task)
            raise McpProtectedHostExecutionAdmissionError(
                "protected MCP root execution task tenant does not match verified principal tenant"
            )
        admission = await supervision_port.admit_root_execution_supervision(
            RootExecutionTransportAdmissionRequest(
                task_id=task.task_id,
                verified_tenant_id=caller_request_identity.tenant_id,
                caller_request_identity=caller_request_identity,
                host_asyncio_task=launch.host_task,
            )
        )
        if (
            admission.disposition
            is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED
        ):
            launch.execution_start_gate.set()
            gate_opened = True
            return
        await _terminate_gated_host_task_after_refused_admission(launch.host_task)
        raise McpProtectedHostExecutionAdmissionError(
            "root execution supervision admission refused before MCP protected execution"
        )
    except asyncio.CancelledError:
        await _finalize_protected_mcp_root_execution_admission_caller_abort(
            supervision_port=supervision_port,
            task=task,
            launch=launch,
            gate_opened=gate_opened,
            transport_waiter_will_continue=transport_waiter_will_continue,
        )
        raise
    except McpProtectedHostExecutionAdmissionError:
        raise
    except Exception:
        await _finalize_protected_mcp_root_execution_admission_caller_abort(
            supervision_port=supervision_port,
            task=task,
            launch=launch,
            gate_opened=gate_opened,
            transport_waiter_will_continue=False,
        )
        raise


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


async def _bounded_observe_host_task(
    host_coro: asyncio.Task[object],
    *,
    timeout_s: float,
) -> bool:
    """Observe *host_coro* up to *timeout_s* without cancelling it. Returns True if completed."""
    if host_coro.done():
        return True
    if timeout_s <= 0:
        return host_coro.done()
    try:
        await asyncio.wait_for(asyncio.shield(host_coro), timeout=timeout_s)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    return host_coro.done()


async def _retrieve_host_task_outcome(host_coro: asyncio.Task[object]) -> None:
    if not host_coro.done():
        return
    with contextlib.suppress(asyncio.CancelledError, Exception):
        host_coro.result()


async def _request_canonical_transport_detachment_handoff(
    host_coro: asyncio.Task[object],
    *,
    task: Task,
    cancel_ctx: McpHostExecutionTransportCancelContext,
    governed_cancel_admitted: bool,
    supervision_port: RootExecutionTransportDetachmentSupervisionPort,
) -> RootExecutionTransportDetachmentHandoffDisposition:
    binding = await ActiveTaskRegistry.get(task.task_id)
    if binding is None:
        return RootExecutionTransportDetachmentHandoffDisposition.REFUSED
    result = await supervision_port.request_transport_detachment_handoff(
        RootExecutionTransportDetachmentHandoffRequest(
            task_id=task.task_id,
            run_id=binding.run_id,
            verified_tenant_id=task.tenant_id,
            caller_request_identity=cancel_ctx.caller_request_identity,
            host_asyncio_task=host_coro,
            governed_transport_cancel_admitted=governed_cancel_admitted,
        )
    )
    return result.disposition


def _supervisor_holds_live_execution(
    supervision_port: RootExecutionTransportDetachmentSupervisionPort | None,
    *,
    task: Task,
    host_coro: asyncio.Task[object],
) -> bool:
    if supervision_port is None or host_coro.done():
        return False
    return supervision_port.holds_root_execution_supervision(
        task_id=task.task_id,
        host_asyncio_task=host_coro,
    )


async def _finalize_transport_waiter_ownership(
    host_coro: asyncio.Task[object],
    *,
    task: Task,
    cancel_ctx: McpHostExecutionTransportCancelContext | None,
    remaining_s: float,
    governed_cancel_admitted: bool,
    supervision_port: RootExecutionTransportDetachmentSupervisionPort | None,
    handoff_accepted: bool,
) -> None:
    """OWN-A2: TERMINAL, ACCEPTED handoff, or admission supervision — never LIVE+UNOWNED+RELEASED."""
    if handoff_accepted or host_coro.done():
        await _retrieve_host_task_outcome(host_coro)
        return
    if _supervisor_holds_live_execution(supervision_port, task=task, host_coro=host_coro):
        return

    if supervision_port is not None and cancel_ctx is not None:
        disposition = await _request_canonical_transport_detachment_handoff(
            host_coro,
            task=task,
            cancel_ctx=cancel_ctx,
            governed_cancel_admitted=governed_cancel_admitted,
            supervision_port=supervision_port,
        )
        if disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED:
            return
        if not governed_cancel_admitted:
            governed_cancel_admitted = await request_governed_cancel_for_bound_mcp_task(
                task=task,
                principal=cancel_ctx.caller_request_identity,
                mutation_boundary=cancel_ctx.mutation_boundary,
            )
        if remaining_s > 0:
            await _bounded_observe_host_task(host_coro, timeout_s=remaining_s)
        await _retrieve_host_task_outcome(host_coro)
        if host_coro.done():
            return
        raise McpHostExecutionTransportOwnershipInvariantError(
            "canonical execution still live without accepted transport detachment handoff"
        )

    if governed_cancel_admitted:
        if remaining_s > 0:
            await _bounded_observe_host_task(host_coro, timeout_s=remaining_s)
        await _retrieve_host_task_outcome(host_coro)
        if host_coro.done():
            return
        raise McpHostExecutionTransportOwnershipInvariantError(
            "unsupervised host execution still live after governed cancel admission"
        )

    host_coro.cancel()
    if remaining_s > 0:
        await _bounded_observe_host_task(host_coro, timeout_s=remaining_s)
    await _retrieve_host_task_outcome(host_coro)
    if not host_coro.done():
        raise McpHostExecutionTransportOwnershipInvariantError(
            "unsupervised host execution survived transport waiter finalization"
        )


async def _await_host_task_bounded_after_transport_waiter_lost(
    host_coro: asyncio.Task[object],
    *,
    task: Task,
    cancel_ctx: McpHostExecutionTransportCancelContext | None,
    timeout_s: float,
    governed_cancel_admitted: bool,
    supervision_port: RootExecutionTransportDetachmentSupervisionPort | None,
) -> None:
    """
    End MCP transport waiter scope with a hard wall-clock bound.

    Canonical Execution remains owned by ``HostTaskExecutionPort.execute`` inside *host_coro*.
    OWN-A2: confirmed supervision handoff before releasing MCP waiter when work continues.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    observe_chunk_s = max(_ACTIVE_BINDING_POLL_INTERVAL_S * 4, 0.05)
    handoff_accepted = False

    def _remaining_s() -> float:
        return max(0.0, deadline - loop.time())

    async def _try_canonical_handoff() -> bool:
        nonlocal handoff_accepted
        if supervision_port is None or cancel_ctx is None:
            return False
        binding = await ActiveTaskRegistry.get(task.task_id)
        if binding is None:
            return False
        disposition = await _request_canonical_transport_detachment_handoff(
            host_coro,
            task=task,
            cancel_ctx=cancel_ctx,
            governed_cancel_admitted=governed_cancel_admitted,
            supervision_port=supervision_port,
        )
        if disposition is RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED:
            handoff_accepted = True
            return True
        return False

    while _remaining_s() > 0 and not host_coro.done():
        if _supervisor_holds_live_execution(supervision_port, task=task, host_coro=host_coro):
            return
        if await _try_canonical_handoff():
            return
        if await _bounded_observe_host_task(
            host_coro,
            timeout_s=min(_remaining_s(), observe_chunk_s),
        ):
            break

    await _finalize_transport_waiter_ownership(
        host_coro,
        task=task,
        cancel_ctx=cancel_ctx,
        remaining_s=_remaining_s(),
        governed_cancel_admitted=governed_cancel_admitted,
        supervision_port=supervision_port,
        handoff_accepted=handoff_accepted,
    )


async def run_host_execution_with_mcp_transport_cancel(
    host_coro: asyncio.Task[object],
    *,
    task: Task,
    cancel_ctx: McpHostExecutionTransportCancelContext,
    transport_detachment_supervision: (
        RootExecutionTransportDetachmentSupervisionPort | None
    ) = None,
    execution_start_gate: asyncio.Event | None = None,
) -> object:
    """
    Await canonical host execution while observing MCP transport cancellation.

    Does not mutate terminal state directly; routes cancel through ``governed_cancel_active_task``.
    """
    cancel_admission_accepted: list[bool] = [False]
    cancel_lock = asyncio.Lock()
    transport_cancel_observed = False
    request = _starlette_request_from_fastmcp()
    supervision_port = transport_detachment_supervision
    if supervision_port is None:
        supervision_port = resolve_transport_detachment_supervision_from_request(request)

    if supervision_port is not None and not supervision_port.holds_root_execution_supervision(
        task_id=task.task_id,
        host_asyncio_task=host_coro,
    ):
        if execution_start_gate is None:
            raise McpProtectedHostExecutionAdmissionError(
                "protected MCP transport execution requires execution_start_gate for "
                "atomic supervision admission"
            )
        await admit_protected_mcp_root_execution_supervision(
            supervision_port=supervision_port,
            task=task,
            caller_request_identity=cancel_ctx.caller_request_identity,
            launch=McpProtectedHostExecutionLaunch(
                host_task=host_coro,
                execution_start_gate=execution_start_gate,
            ),
            transport_waiter_will_continue=True,
        )

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
                task=task,
                cancel_ctx=cancel_ctx,
                timeout_s=_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S,
                governed_cancel_admitted=cancel_admission_accepted[0],
                supervision_port=supervision_port,
            )
        if host_coro.done() and host_result_after_waiter_lost is None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                host_result_after_waiter_lost = host_coro.result()
    if transport_waiter_lost:
        if host_result_after_waiter_lost is not None:
            return host_result_after_waiter_lost
        if not host_coro.done():
            await _bounded_observe_host_task(
                host_coro,
                timeout_s=_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S,
            )
        if host_coro.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                return host_coro.result()
        raise asyncio.CancelledError()
    raise AssertionError("unreachable_host_transport_cancel_waiter_state")


__all__ = [
    "MCP_TRANSPORT_CANCEL_MUTATION_PREFIX",
    "MCP_TRANSPORT_CANCEL_REASON",
    "McpHostExecutionTransportOwnershipInvariantError",
    "McpProtectedHostExecutionAdmissionError",
    "McpProtectedHostExecutionLaunch",
    "McpHostExecutionTransportCancelContext",
    "admit_protected_mcp_root_execution_supervision",
    "create_mcp_protected_host_execution_launch",
    "request_governed_cancel_for_bound_mcp_task",
    "resolve_mcp_task_control_mutation_boundary_from_request",
    "run_host_execution_with_mcp_transport_cancel",
]
