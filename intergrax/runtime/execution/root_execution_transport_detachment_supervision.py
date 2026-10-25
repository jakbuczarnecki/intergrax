# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host-root Execution supervision after MCP transport detachment (TLX-TL11 OWN-A2)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

from intergrax.contracts.root_execution_transport_detachment_supervision import (
    RootExecutionTransportAdmissionRequest,
    RootExecutionTransportAdmissionResult,
    RootExecutionTransportDetachmentHandoffDisposition,
    RootExecutionTransportDetachmentHandoffRefusalReason,
    RootExecutionTransportDetachmentHandoffRequest,
    RootExecutionTransportDetachmentHandoffResult,
    RootExecutionTransportDetachmentSupervisionPort,
    RootExecutionTransportPreStartFinalizationDisposition,
    RootExecutionTransportPreStartFinalizationRequest,
    RootExecutionTransportPreStartFinalizationResult,
)
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry

_logger = logging.getLogger(__name__)

_DEFAULT_SHUTDOWN_DRAIN_S = 30.0


async def _await_host_asyncio_task_terminal(
    host_task: asyncio.Task[object],
    *,
    timeout_s: float,
) -> bool:
    """Wait for *host_task* to complete without cancelling it; shielded against caller cancel."""
    if host_task.done():
        return True
    try:
        await asyncio.wait_for(asyncio.shield(host_task), timeout=timeout_s)
    except asyncio.TimeoutError:
        return host_task.done()
    except asyncio.CancelledError:
        if host_task.done():
            return True
        try:
            await asyncio.wait_for(asyncio.shield(host_task), timeout=timeout_s)
        except asyncio.TimeoutError:
            return host_task.done()
        except asyncio.CancelledError:
            return host_task.done()
    return host_task.done()


async def _retrieve_host_asyncio_task_outcome(host_task: asyncio.Task[object]) -> None:
    if not host_task.done():
        return
    with contextlib.suppress(asyncio.CancelledError, Exception):
        host_task.result()


def _duplicate_handoff_result(
    request: RootExecutionTransportDetachmentHandoffRequest,
    existing_host_task: asyncio.Task[object],
) -> RootExecutionTransportDetachmentHandoffResult:
    if existing_host_task is not request.host_asyncio_task:
        return RootExecutionTransportDetachmentHandoffResult(
            disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
            refusal_reason=(
                RootExecutionTransportDetachmentHandoffRefusalReason.DUPLICATE_HANDOFF_CONFLICT
            ),
        )
    if request.caller_request_identity.tenant_id != request.verified_tenant_id:
        return RootExecutionTransportDetachmentHandoffResult(
            disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
            refusal_reason=(
                RootExecutionTransportDetachmentHandoffRefusalReason.PRINCIPAL_TENANT_MISMATCH
            ),
        )
    return RootExecutionTransportDetachmentHandoffResult(
        disposition=RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED,
    )


async def _validate_active_binding_for_handoff(
    request: RootExecutionTransportDetachmentHandoffRequest,
) -> RootExecutionTransportDetachmentHandoffResult | None:
    binding = await ActiveTaskRegistry.get(request.task_id)
    if binding is None:
        return RootExecutionTransportDetachmentHandoffResult(
            disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
            refusal_reason=(
                RootExecutionTransportDetachmentHandoffRefusalReason.NO_ACTIVE_BINDING
            ),
        )
    if binding.run_id != request.run_id:
        return RootExecutionTransportDetachmentHandoffResult(
            disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
            refusal_reason=RootExecutionTransportDetachmentHandoffRefusalReason.RUN_ID_MISMATCH,
        )
    if binding.task.tenant_id != request.verified_tenant_id:
        return RootExecutionTransportDetachmentHandoffResult(
            disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
            refusal_reason=RootExecutionTransportDetachmentHandoffRefusalReason.TENANT_MISMATCH,
        )
    if request.caller_request_identity.tenant_id != request.verified_tenant_id:
        return RootExecutionTransportDetachmentHandoffResult(
            disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
            refusal_reason=(
                RootExecutionTransportDetachmentHandoffRefusalReason.PRINCIPAL_TENANT_MISMATCH
            ),
        )
    return None


@dataclass
class HostRootExecutionTransportDetachmentSupervisor:
    """Private supervised-work collection owned by canonical host Execution composition."""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _accepting_handoffs: bool = True
    _supervised: dict[TaskId, asyncio.Task[object]] = field(default_factory=dict)
    _bound_run_id: dict[TaskId, RunId] = field(default_factory=dict)
    _observers: dict[TaskId, asyncio.Task[None]] = field(default_factory=dict)

    def holds_root_execution_supervision(
        self,
        *,
        task_id: TaskId,
        host_asyncio_task: asyncio.Task[object],
    ) -> bool:
        existing = self._supervised.get(task_id)
        return existing is host_asyncio_task and not host_asyncio_task.done()

    async def admit_root_execution_supervision(
        self,
        request: RootExecutionTransportAdmissionRequest,
    ) -> RootExecutionTransportAdmissionResult:
        if request.host_asyncio_task.done():
            return RootExecutionTransportAdmissionResult(
                disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                refusal_reason=(
                    RootExecutionTransportDetachmentHandoffRefusalReason.HOST_TASK_ALREADY_DONE
                ),
            )
        if request.caller_request_identity.tenant_id != request.verified_tenant_id:
            return RootExecutionTransportAdmissionResult(
                disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                refusal_reason=(
                    RootExecutionTransportDetachmentHandoffRefusalReason.PRINCIPAL_TENANT_MISMATCH
                ),
            )
        async with self._lock:
            if not self._accepting_handoffs:
                return RootExecutionTransportAdmissionResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                    refusal_reason=(
                        RootExecutionTransportDetachmentHandoffRefusalReason.SHUTDOWN_IN_PROGRESS
                    ),
                )
            existing = self._supervised.get(request.task_id)
            if existing is not None:
                if existing is not request.host_asyncio_task:
                    return RootExecutionTransportAdmissionResult(
                        disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                        refusal_reason=(
                            RootExecutionTransportDetachmentHandoffRefusalReason.DUPLICATE_HANDOFF_CONFLICT
                        ),
                    )
                return RootExecutionTransportAdmissionResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED,
                )
            self._supervised[request.task_id] = request.host_asyncio_task
            try:
                observer = asyncio.create_task(
                    self._observe_supervised_work(
                        request.task_id, request.host_asyncio_task
                    ),
                    name=f"root-exec-transport-supervision:{request.task_id}",
                )
                self._observers[request.task_id] = observer
            except Exception:
                self._supervised.pop(request.task_id, None)
                raise

        return RootExecutionTransportAdmissionResult(
            disposition=RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED,
        )

    async def finalize_root_execution_supervision_before_start(
        self,
        request: RootExecutionTransportPreStartFinalizationRequest,
    ) -> RootExecutionTransportPreStartFinalizationResult:
        host_task = request.host_asyncio_task
        if request.transport_waiter_will_continue:
            if self.holds_root_execution_supervision(
                task_id=request.task_id,
                host_asyncio_task=host_task,
            ):
                request.execution_start_gate.set()
                return RootExecutionTransportPreStartFinalizationResult(
                    disposition=(
                        RootExecutionTransportPreStartFinalizationDisposition.SUPERVISED_CONTINUATION_GATE_OPENED
                    ),
                )
            return RootExecutionTransportPreStartFinalizationResult(
                disposition=RootExecutionTransportPreStartFinalizationDisposition.NOT_SUPERVISED,
            )

        if not self.holds_root_execution_supervision(
            task_id=request.task_id,
            host_asyncio_task=host_task,
        ):
            return RootExecutionTransportPreStartFinalizationResult(
                disposition=RootExecutionTransportPreStartFinalizationDisposition.NOT_SUPERVISED,
            )

        if not host_task.done():
            host_task.cancel()
        completed = await _await_host_asyncio_task_terminal(
            host_task,
            timeout_s=request.host_terminate_timeout_s,
        )
        await _retrieve_host_asyncio_task_outcome(host_task)
        if not completed:
            return RootExecutionTransportPreStartFinalizationResult(
                disposition=(
                    RootExecutionTransportPreStartFinalizationDisposition.HOST_NOT_TERMINAL_UNDER_SUPERVISION
                ),
            )

        await self.relinquish_root_execution_supervision_admission(
            task_id=request.task_id,
            host_asyncio_task=host_task,
        )
        return RootExecutionTransportPreStartFinalizationResult(
            disposition=RootExecutionTransportPreStartFinalizationDisposition.TERMINATED_UNDER_SUPERVISION,
        )

    async def relinquish_root_execution_supervision_admission(
        self,
        *,
        task_id: TaskId,
        host_asyncio_task: asyncio.Task[object],
    ) -> bool:
        observer: asyncio.Task[None] | None = None
        async with self._lock:
            existing = self._supervised.get(task_id)
            if existing is not host_asyncio_task:
                return False
            self._supervised.pop(task_id, None)
            self._bound_run_id.pop(task_id, None)
            observer = self._observers.pop(task_id, None)
        if observer is not None:
            observer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await observer
        return True

    async def _register_supervised_handoff(
        self,
        request: RootExecutionTransportDetachmentHandoffRequest,
    ) -> RootExecutionTransportDetachmentHandoffResult:
        async with self._lock:
            if not self._accepting_handoffs:
                return RootExecutionTransportDetachmentHandoffResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                    refusal_reason=(
                        RootExecutionTransportDetachmentHandoffRefusalReason.SHUTDOWN_IN_PROGRESS
                    ),
                )
            existing = self._supervised.get(request.task_id)
            if existing is not None:
                return _duplicate_handoff_result(request, existing)
            if request.host_asyncio_task.done():
                return RootExecutionTransportDetachmentHandoffResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                    refusal_reason=(
                        RootExecutionTransportDetachmentHandoffRefusalReason.HOST_TASK_ALREADY_DONE
                    ),
                )
            self._supervised[request.task_id] = request.host_asyncio_task
            self._bound_run_id[request.task_id] = request.run_id
            observer = asyncio.create_task(
                self._observe_supervised_work(request.task_id, request.host_asyncio_task),
                name=f"root-exec-transport-supervision:{request.task_id}",
            )
            self._observers[request.task_id] = observer

        return RootExecutionTransportDetachmentHandoffResult(
            disposition=RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED,
        )

    async def request_transport_detachment_handoff(
        self,
        request: RootExecutionTransportDetachmentHandoffRequest,
    ) -> RootExecutionTransportDetachmentHandoffResult:
        if request.host_asyncio_task.done():
            return RootExecutionTransportDetachmentHandoffResult(
                disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                refusal_reason=(
                    RootExecutionTransportDetachmentHandoffRefusalReason.HOST_TASK_ALREADY_DONE
                ),
            )

        existing = self._supervised.get(request.task_id)
        if existing is not None:
            if existing is not request.host_asyncio_task:
                return RootExecutionTransportDetachmentHandoffResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                    refusal_reason=(
                        RootExecutionTransportDetachmentHandoffRefusalReason.DUPLICATE_HANDOFF_CONFLICT
                    ),
                )
            binding_failure = await _validate_active_binding_for_handoff(request)
            if binding_failure is not None:
                return binding_failure
            async with self._lock:
                self._bound_run_id[request.task_id] = request.run_id
            return RootExecutionTransportDetachmentHandoffResult(
                disposition=RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED,
            )

        binding_failure = await _validate_active_binding_for_handoff(request)
        if binding_failure is not None:
            return binding_failure

        async with self._lock:
            if not self._accepting_handoffs:
                return RootExecutionTransportDetachmentHandoffResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                    refusal_reason=(
                        RootExecutionTransportDetachmentHandoffRefusalReason.SHUTDOWN_IN_PROGRESS
                    ),
                )
            existing = self._supervised.get(request.task_id)
            if existing is not None:
                return _duplicate_handoff_result(request, existing)
            if request.host_asyncio_task.done():
                return RootExecutionTransportDetachmentHandoffResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                    refusal_reason=(
                        RootExecutionTransportDetachmentHandoffRefusalReason.HOST_TASK_ALREADY_DONE
                    ),
                )

        return await self._register_supervised_handoff(request)

    async def _observe_supervised_work(
        self,
        task_id: TaskId,
        host_task: asyncio.Task[object],
    ) -> None:
        try:
            await host_task
        except asyncio.CancelledError:
            pass
        except GeneratorExit:
            raise
        except Exception:
            _logger.exception(
                "supervised host execution failed after transport detachment task_id=%s",
                task_id,
            )
        finally:
            if host_task.done() and not host_task.cancelled():
                with contextlib.suppress(asyncio.InvalidStateError):
                    exc = host_task.exception()
                    if exc is not None:
                        _logger.error(
                            "supervised host execution terminal exception task_id=%s",
                            task_id,
                            exc_info=exc,
                        )
            async with self._lock:
                if self._supervised.get(task_id) is host_task:
                    self._supervised.pop(task_id, None)
                    self._bound_run_id.pop(task_id, None)
                observer = self._observers.get(task_id)
                if observer is asyncio.current_task():
                    self._observers.pop(task_id, None)

    async def shutdown_supervised_work(self, *, timeout_s: float) -> None:
        async with self._lock:
            self._accepting_handoffs = False
            pending = frozenset(self._supervised.values())
        if not pending:
            return
        _done, still_pending = await asyncio.wait(
            pending,
            timeout=timeout_s,
            return_when=asyncio.ALL_COMPLETED,
        )
        if still_pending:
            _logger.warning(
                "shutdown_supervised_work timed out after %ss with %s pending host tasks "
                "(execution tasks were not cancelled by supervisor shutdown)",
                timeout_s,
                len(still_pending),
            )

    def supervised_count(self) -> int:
        return len(self._supervised)


def build_host_root_execution_transport_detachment_supervisor() -> (
    HostRootExecutionTransportDetachmentSupervisor
):
    return HostRootExecutionTransportDetachmentSupervisor()


__all__ = [
    "HostRootExecutionTransportDetachmentSupervisor",
    "_DEFAULT_SHUTDOWN_DRAIN_S",
    "build_host_root_execution_transport_detachment_supervisor",
]
