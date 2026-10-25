# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical MCP transport detachment → host Execution supervision handoff (TLX-TL11 OWN-A2)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_identity import RunId, TaskId, validate_run_id, validate_task_id


class RootExecutionTransportDetachmentHandoffDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    REFUSED = "REFUSED"


class RootExecutionTransportDetachmentHandoffRefusalReason(StrEnum):
    SHUTDOWN_IN_PROGRESS = "shutdown_in_progress"
    NO_ACTIVE_BINDING = "no_active_binding"
    TENANT_MISMATCH = "tenant_mismatch"
    PRINCIPAL_TENANT_MISMATCH = "principal_tenant_mismatch"
    RUN_ID_MISMATCH = "run_id_mismatch"
    HOST_TASK_ALREADY_DONE = "host_task_already_done"
    DUPLICATE_HANDOFF_CONFLICT = "duplicate_handoff_conflict"


@dataclass(frozen=True, slots=True)
class RootExecutionTransportAdmissionRequest:
    """Register accountable supervision when MCP root Execution is admitted (before transport detach)."""

    task_id: TaskId
    verified_tenant_id: str
    caller_request_identity: RequestIdentity
    host_asyncio_task: asyncio.Task[object]

    def __post_init__(self) -> None:
        validate_task_id(self.task_id)
        if not self.verified_tenant_id:
            raise ValueError("verified_tenant_id required")


@dataclass(frozen=True, slots=True)
class RootExecutionTransportAdmissionResult:
    disposition: RootExecutionTransportDetachmentHandoffDisposition
    refusal_reason: RootExecutionTransportDetachmentHandoffRefusalReason | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not RootExecutionTransportDetachmentHandoffDisposition:
            raise TypeError("disposition must be RootExecutionTransportDetachmentHandoffDisposition")
        if self.disposition is RootExecutionTransportDetachmentHandoffDisposition.REFUSED:
            if self.refusal_reason is None:
                raise ValueError("REFUSED requires refusal_reason")
        elif self.refusal_reason is not None:
            raise ValueError("ACCEPTED must not carry refusal_reason")


@dataclass(frozen=True, slots=True)
class RootExecutionTransportDetachmentHandoffRequest:
    """Hand off already-running host asyncio work to canonical Execution supervision."""

    task_id: TaskId
    run_id: RunId
    verified_tenant_id: str
    caller_request_identity: RequestIdentity
    host_asyncio_task: asyncio.Task[object]
    governed_transport_cancel_admitted: bool

    def __post_init__(self) -> None:
        validate_task_id(self.task_id)
        validate_run_id(self.run_id)
        if not self.verified_tenant_id:
            raise ValueError("verified_tenant_id required")


@dataclass(frozen=True, slots=True)
class RootExecutionTransportDetachmentHandoffResult:
    disposition: RootExecutionTransportDetachmentHandoffDisposition
    refusal_reason: RootExecutionTransportDetachmentHandoffRefusalReason | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not RootExecutionTransportDetachmentHandoffDisposition:
            raise TypeError("disposition must be RootExecutionTransportDetachmentHandoffDisposition")
        if self.disposition is RootExecutionTransportDetachmentHandoffDisposition.REFUSED:
            if self.refusal_reason is None:
                raise ValueError("REFUSED requires refusal_reason")
        elif self.refusal_reason is not None:
            raise ValueError("ACCEPTED must not carry refusal_reason")


@runtime_checkable
class RootExecutionTransportDetachmentSupervisionPort(Protocol):
    """Accept responsibility for live host Execution work after MCP transport detachment."""

    async def admit_root_execution_supervision(
        self,
        request: RootExecutionTransportAdmissionRequest,
    ) -> RootExecutionTransportAdmissionResult: ...

    def holds_root_execution_supervision(
        self,
        *,
        task_id: TaskId,
        host_asyncio_task: asyncio.Task[object],
    ) -> bool: ...

    async def request_transport_detachment_handoff(
        self,
        request: RootExecutionTransportDetachmentHandoffRequest,
    ) -> RootExecutionTransportDetachmentHandoffResult: ...

    async def shutdown_supervised_work(self, *, timeout_s: float) -> None: ...

    async def relinquish_root_execution_supervision_admission(
        self,
        *,
        task_id: TaskId,
        host_asyncio_task: asyncio.Task[object],
    ) -> bool:
        """
        Drop pre-start supervision for *host_asyncio_task* when the admission caller aborts
        before the execution start gate opens and no transport waiter will continue.
        """
        ...


__all__ = [
    "RootExecutionTransportAdmissionRequest",
    "RootExecutionTransportAdmissionResult",
    "RootExecutionTransportDetachmentHandoffDisposition",
    "RootExecutionTransportDetachmentHandoffRefusalReason",
    "RootExecutionTransportDetachmentHandoffRequest",
    "RootExecutionTransportDetachmentHandoffResult",
    "RootExecutionTransportDetachmentSupervisionPort",
]
