# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host-root Execution supervision after MCP transport detachment (TLX-TL11 OWN-A2)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

from intergrax.contracts.root_execution_transport_detachment_supervision import (
    RootExecutionTransportDetachmentHandoffDisposition,
    RootExecutionTransportDetachmentHandoffRefusalReason,
    RootExecutionTransportDetachmentHandoffRequest,
    RootExecutionTransportDetachmentHandoffResult,
    RootExecutionTransportDetachmentSupervisionPort,
)
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry

_logger = logging.getLogger(__name__)

_DEFAULT_SHUTDOWN_DRAIN_S = 30.0


@dataclass
class HostRootExecutionTransportDetachmentSupervisor:
    """Private supervised-work collection owned by canonical host Execution composition."""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _accepting_handoffs: bool = True
    _supervised: dict[tuple[TaskId, RunId], asyncio.Task[object]] = field(
        default_factory=dict,
    )
    _observers: dict[tuple[TaskId, RunId], asyncio.Task[None]] = field(
        default_factory=dict,
    )

    async def request_transport_detachment_handoff(
        self,
        request: RootExecutionTransportDetachmentHandoffRequest,
    ) -> RootExecutionTransportDetachmentHandoffResult:
        key = (request.task_id, request.run_id)
        if request.host_asyncio_task.done():
            return RootExecutionTransportDetachmentHandoffResult(
                disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                refusal_reason=(
                    RootExecutionTransportDetachmentHandoffRefusalReason.HOST_TASK_ALREADY_DONE
                ),
            )
        async with self._lock:
            if not self._accepting_handoffs:
                return RootExecutionTransportDetachmentHandoffResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                    refusal_reason=(
                        RootExecutionTransportDetachmentHandoffRefusalReason.SHUTDOWN_IN_PROGRESS
                    ),
                )
            if key in self._supervised:
                return RootExecutionTransportDetachmentHandoffResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED,
                )

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

        async with self._lock:
            if not self._accepting_handoffs:
                return RootExecutionTransportDetachmentHandoffResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.REFUSED,
                    refusal_reason=(
                        RootExecutionTransportDetachmentHandoffRefusalReason.SHUTDOWN_IN_PROGRESS
                    ),
                )
            if key in self._supervised:
                return RootExecutionTransportDetachmentHandoffResult(
                    disposition=RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED,
                )
            self._supervised[key] = request.host_asyncio_task
            observer = asyncio.create_task(
                self._observe_supervised_work(key, request.host_asyncio_task),
                name=f"root-exec-transport-supervision:{request.task_id}",
            )
            self._observers[key] = observer

        return RootExecutionTransportDetachmentHandoffResult(
            disposition=RootExecutionTransportDetachmentHandoffDisposition.ACCEPTED,
        )

    async def _observe_supervised_work(
        self,
        key: tuple[TaskId, RunId],
        host_task: asyncio.Task[object],
    ) -> None:
        try:
            await host_task
        except asyncio.CancelledError:
            pass
        except Exception:
            _logger.exception(
                "supervised host execution failed after transport detachment task_id=%s run_id=%s",
                key[0],
                key[1],
            )
        finally:
            if host_task.done() and not host_task.cancelled():
                with contextlib.suppress(asyncio.InvalidStateError):
                    _ = host_task.exception()
            async with self._lock:
                self._supervised.pop(key, None)
                self._observers.pop(key, None)

    async def shutdown_supervised_work(self, *, timeout_s: float) -> None:
        async with self._lock:
            self._accepting_handoffs = False
            pending = list(self._supervised.values())
        if not pending:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            _logger.warning(
                "shutdown_supervised_work timed out after %ss with %s pending host tasks",
                timeout_s,
                len(pending),
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
