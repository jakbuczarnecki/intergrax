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


class RootExecutionTransportPreStartFinalizationDisposition(StrEnum):
    """Outcome of canonical pre-start supervision finalization (OWN-A2 admission cleanup)."""

    SUPERVISED_CONTINUATION_GATE_OPENED = "supervised_continuation_gate_opened"
    TERMINATED_UNDER_SUPERVISION = "terminated_under_supervision"
    NOT_SUPERVISED = "not_supervised"
    HOST_NOT_TERMINAL_UNDER_SUPERVISION = "host_not_terminal_under_supervision"


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
class RootExecutionTransportPreStartFinalizationRequest:
    """Terminate or continue gated host work while retaining accountable supervision."""

    task_id: TaskId
    host_asyncio_task: asyncio.Task[object]
    execution_start_gate: asyncio.Event
    transport_waiter_will_continue: bool
    host_terminate_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        validate_task_id(self.task_id)
        if self.host_terminate_timeout_s <= 0:
            raise ValueError("host_terminate_timeout_s must be positive")


@dataclass(frozen=True, slots=True)
class RootExecutionTransportPreStartFinalizationResult:
    disposition: RootExecutionTransportPreStartFinalizationDisposition

    def __post_init__(self) -> None:
        if type(self.disposition) is not RootExecutionTransportPreStartFinalizationDisposition:
            raise TypeError(
                "disposition must be RootExecutionTransportPreStartFinalizationDisposition"
            )


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

    async def finalize_root_execution_supervision_before_start(
        self,
        request: RootExecutionTransportPreStartFinalizationRequest,
    ) -> RootExecutionTransportPreStartFinalizationResult:
        """
        Single owner for pre-start cancellation, terminal observation, and supervision release.

        Retains strong supervision until the gated host task is terminal (or opens the start gate
        when supervised continuation is required). Does not relinquish on non-terminal timeout.
        """
        ...


__all__ = [
    "RootExecutionTransportAdmissionRequest",
    "RootExecutionTransportAdmissionResult",
    "RootExecutionTransportPreStartFinalizationDisposition",
    "RootExecutionTransportPreStartFinalizationRequest",
    "RootExecutionTransportPreStartFinalizationResult",
    "RootExecutionTransportDetachmentHandoffDisposition",
    "RootExecutionTransportDetachmentHandoffRefusalReason",
    "RootExecutionTransportDetachmentHandoffRequest",
    "RootExecutionTransportDetachmentHandoffResult",
    "RootExecutionTransportDetachmentSupervisionPort",
]
