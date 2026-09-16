# © Artur Czarnecki. All rights reserved.

"""Subprocess delegated execution provider adapter (P2.1-S2D)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic, TypeVar
from uuid import uuid4

from intergrax.contracts.delegated_execution_control import (
    DelegatedExecutionControlOperation,
    DelegatedExecutionControlOutcome,
    DelegatedExecutionControlOutcomeCategory,
    DelegatedExecutionControlRequest,
    delegated_control_outcome,
)
from intergrax.contracts.delegated_execution_continuation import (
    DelegatedExecutionContinuationRequest,
    DelegatedExecutionProviderReattachmentObservation,
    DelegatedExecutionReattachmentKind,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionCapabilities,
    DelegatedExecutionContractError,
    DelegatedExecutionOutcome,
    DelegatedExecutionOutcomeCategory,
    DelegatedExecutionProvider,
    DelegatedExecutionProviderError,
    DelegatedExecutionRequest,
    DelegatedExecutionTransportError,
    assert_provider_native_ids_distinct_from_execution,
    delegated_failure_outcome,
    delegated_success_outcome,
    digest_delegated_execution_payload,
    digest_delegated_execution_request,
    mint_delegated_provider_invocation,
    validate_provider_identity,
)
from intergrax.contracts.delegated_execution_status import (
    DelegatedExecutionProviderPhysicalStatus,
    DelegatedExecutionProviderStatusObservation,
    DelegatedExecutionStatusRequest,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.integrations.providers.delegated_execution.subprocess.config import (
    SubprocessDelegatedExecutionProviderConfig,
)
from intergrax.integrations.providers.delegated_execution.subprocess.protocol import (
    WorkerExecutePayload,
    WorkerRequest,
    WorkerResponseStatus,
    WorkerRpcMethod,
)
from intergrax.integrations.providers.delegated_execution.subprocess.transport import (
    DelegatedExecutionPostDispatchTransportError,
    SubprocessDelegatedExecutionTransport,
    TcpSubprocessDelegatedExecutionTransport,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")

SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_ID = "subprocess_delegated_execution"
SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_VERSION = "1.0.0"

_TRANSPORT_FAILURE_MESSAGE = "subprocess delegated execution transport failed"
_OUTCOME_UNKNOWN_MESSAGE = "subprocess delegated execution outcome unknown"
_PROVIDER_FAILURE_MESSAGE = "subprocess delegated execution provider failed"
_CONTRACT_MISMATCH_MESSAGE = "subprocess delegated execution contract mismatch"


@dataclass(frozen=True)
class SubprocessEchoPayload:
    value: str
    behavior: str | None = None


@dataclass(frozen=True)
class SubprocessEchoResult:
    value: str
    child_execution_id: str
    parent_execution_id: str


class SubprocessDelegatedExecutionProvider(
    Generic[RequestT, ResultT],
    DelegatedExecutionProvider[RequestT, ResultT],
):
    """External-process delegated execution via framed TCP to a worker subprocess."""

    __slots__ = (
        "_config",
        "_provider_id",
        "_provider_version",
        "_capabilities",
        "_transport",
        "_owns_transport",
    )

    def __init__(
        self,
        config: SubprocessDelegatedExecutionProviderConfig,
        *,
        transport: SubprocessDelegatedExecutionTransport | None = None,
        provider_id: str = SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_ID,
        provider_version: str = SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_VERSION,
    ) -> None:
        validate_provider_identity(provider_id=provider_id, provider_version=provider_version)
        self._config = config
        self._provider_id = provider_id.strip()
        self._provider_version = provider_version.strip()
        self._capabilities = DelegatedExecutionCapabilities(
            provider_id=self._provider_id,
            supports_cancel=True,
            supports_status_read=True,
            supports_reattachment=True,
        )
        if transport is None:
            self._transport = TcpSubprocessDelegatedExecutionTransport.spawn(config)
            self._owns_transport = True
        else:
            self._transport = transport
            self._owns_transport = False

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return self._provider_version

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return self._capabilities

    def close(self) -> None:
        if self._owns_transport:
            self._transport.close()

    async def execute(
        self,
        request: DelegatedExecutionRequest[RequestT],
    ) -> DelegatedExecutionOutcome[ResultT]:
        started_at = datetime.now(timezone.utc)
        payload = _coerce_payload(request.payload)
        payload_digest = digest_delegated_execution_payload(payload)
        request_digest = digest_delegated_execution_request(
            context=request.context,
            operation=request.operation,
            payload_digest=payload_digest,
        )
        invocation_id = f"sub-dep-inv-{uuid4().hex}"
        provider_request_id = f"sub-dep-req-{uuid4().hex}"
        provider_operation_id = f"sub-dep-op-{uuid4().hex}"
        invocation = mint_delegated_provider_invocation(
            context=request.context,
            operation=request.operation,
            provider_id=self._provider_id,
            request_digest=request_digest,
            started_at=started_at,
            invocation_id=invocation_id,
            provider_request_id=provider_request_id,
            provider_operation_id=provider_operation_id,
        )
        wire = WorkerRequest(
            method=WorkerRpcMethod.EXECUTE,
            request_id=f"wire-{uuid4().hex}",
            auth_token=self._config.connection_auth_token,
            payload=WorkerExecutePayload(value=payload.value, behavior=payload.behavior),
            provider_request_id=provider_request_id,
            provider_operation_id=provider_operation_id,
            invocation_id=invocation_id,
            execution_id=str(request.context.execution_id),
            parent_execution_id=str(request.context.parent_execution_id),
        )
        try:
            response = await asyncio.to_thread(
                self._transport.roundtrip,
                wire,
                timeout_seconds=self._config.request_timeout_seconds,
                connect_timeout_seconds=self._config.connect_timeout_seconds,
            )
        except DelegatedExecutionPostDispatchTransportError:
            return _outcome_unknown(
                invocation=invocation,
                invocation_id=invocation_id,
                started_at=started_at,
                provider_request_id=provider_request_id,
                provider_operation_id=provider_operation_id,
            )
        except DelegatedExecutionTransportError:
            return delegated_failure_outcome(
                category=DelegatedExecutionOutcomeCategory.TRANSPORT_FAILURE,
                failure_code="TRANSPORT_FAILURE",
                failure_message=_TRANSPORT_FAILURE_MESSAGE,
                provider_invocation=invocation,
                provider_outcome=_failed_outcome(
                    invocation_id=invocation_id,
                    started_at=started_at,
                    error_code="TRANSPORT_FAILURE",
                    provider_request_id=provider_request_id,
                    provider_operation_id=provider_operation_id,
                ),
            )
        except TimeoutError:
            return _outcome_unknown(
                invocation=invocation,
                invocation_id=invocation_id,
                started_at=started_at,
                provider_request_id=provider_request_id,
                provider_operation_id=provider_operation_id,
            )

        if response.status is WorkerResponseStatus.ERROR:
            code = (response.error_code or "PROVIDER_FAILURE").strip()
            return delegated_failure_outcome(
                category=DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE,
                failure_code=code,
                failure_message=_PROVIDER_FAILURE_MESSAGE,
                provider_invocation=invocation,
                provider_outcome=_failed_outcome(
                    invocation_id=invocation_id,
                    started_at=started_at,
                    error_code=code,
                    provider_request_id=provider_request_id,
                    provider_operation_id=provider_operation_id,
                ),
            )

        if response.result is None:
            return delegated_failure_outcome(
                category=DelegatedExecutionOutcomeCategory.TRANSPORT_FAILURE,
                failure_code="OUTCOME_UNKNOWN",
                failure_message=_OUTCOME_UNKNOWN_MESSAGE,
                provider_invocation=invocation,
                provider_outcome=ProviderInvocationOutcome(
                    invocation_id=invocation_id,
                    status=ProviderInvocationStatus.UNKNOWN,
                    completed_at=datetime.now(timezone.utc),
                    error_code="OUTCOME_UNKNOWN",
                    provider_request_id=provider_request_id,
                    provider_operation_id=provider_operation_id,
                ),
            )

        try:
            assert_provider_native_ids_distinct_from_execution(
                execution_id=request.context.execution_id,
                provider_request_id=response.provider_request_id,
                provider_operation_id=response.provider_operation_id,
                invocation_id=invocation_id,
            )
        except DelegatedExecutionContractError:
            return delegated_failure_outcome(
                category=DelegatedExecutionOutcomeCategory.PLATFORM_FAILURE,
                failure_code="OUTCOME_CONTRACT_MISMATCH",
                failure_message=_CONTRACT_MISMATCH_MESSAGE,
                provider_invocation=invocation,
                provider_outcome=_failed_outcome(
                    invocation_id=invocation_id,
                    started_at=started_at,
                    error_code="OUTCOME_CONTRACT_MISMATCH",
                    provider_request_id=provider_request_id,
                    provider_operation_id=provider_operation_id,
                ),
            )

        if (
            response.provider_request_id != provider_request_id
            or response.provider_operation_id != provider_operation_id
        ):
            return delegated_failure_outcome(
                category=DelegatedExecutionOutcomeCategory.PLATFORM_FAILURE,
                failure_code="OUTCOME_CONTRACT_MISMATCH",
                failure_message=_CONTRACT_MISMATCH_MESSAGE,
                provider_invocation=invocation,
                provider_outcome=_failed_outcome(
                    invocation_id=invocation_id,
                    started_at=started_at,
                    error_code="OUTCOME_CONTRACT_MISMATCH",
                    provider_request_id=provider_request_id,
                    provider_operation_id=provider_operation_id,
                ),
            )

        result = SubprocessEchoResult(
            value=response.result.value,
            child_execution_id=response.result.child_execution_id,
            parent_execution_id=response.result.parent_execution_id,
        )
        completed_at = datetime.now(timezone.utc)
        provider_outcome = ProviderInvocationOutcome(
            invocation_id=invocation_id,
            status=ProviderInvocationStatus.SUCCEEDED,
            completed_at=completed_at,
            response_digest=digest_delegated_execution_payload(result),
            provider_request_id=response.provider_request_id,
            provider_operation_id=response.provider_operation_id,
        )
        return delegated_success_outcome(
            result=result,  # type: ignore[arg-type]
            provider_invocation=invocation,
            provider_outcome=provider_outcome,
        )

    async def cancel_delegated_execution(
        self,
        request: DelegatedExecutionControlRequest,
    ) -> DelegatedExecutionControlOutcome:
        return await self._control(request, operation=DelegatedExecutionControlOperation.CANCEL)

    async def read_delegated_execution_status(
        self,
        request: DelegatedExecutionStatusRequest,
    ) -> DelegatedExecutionProviderStatusObservation:
        binding = request.invocation_binding
        inv = binding.provider_invocation
        wire = WorkerRequest(
            method=WorkerRpcMethod.STATUS,
            request_id=f"wire-{uuid4().hex}",
            auth_token=self._config.connection_auth_token,
            provider_request_id=inv.provider_request_id,
            provider_operation_id=inv.provider_operation_id,
            invocation_id=inv.invocation_id,
        )
        try:
            response = await asyncio.to_thread(
                self._transport.roundtrip,
                wire,
                timeout_seconds=self._config.request_timeout_seconds,
                connect_timeout_seconds=self._config.connect_timeout_seconds,
            )
        except DelegatedExecutionTransportError:
            raise
        if response.status is WorkerResponseStatus.ERROR:
            raise DelegatedExecutionProviderError(_PROVIDER_FAILURE_MESSAGE)
        physical = _map_physical_status(response.physical_status)
        invocation_id = response.invocation_id or ""
        return DelegatedExecutionProviderStatusObservation(
            provider_id=self._provider_id,
            invocation_id=invocation_id,
            provider_request_id=response.provider_request_id,
            provider_operation_id=response.provider_operation_id,
            physical_status=physical,
        )

    async def reattach_delegated_execution(
        self,
        request: DelegatedExecutionContinuationRequest,
    ) -> DelegatedExecutionProviderReattachmentObservation:
        inv = request.provider_invocation
        wire = WorkerRequest(
            method=WorkerRpcMethod.REATTACH,
            request_id=f"wire-{uuid4().hex}",
            auth_token=self._config.connection_auth_token,
            provider_request_id=inv.provider_request_id,
            provider_operation_id=inv.provider_operation_id,
            invocation_id=inv.invocation_id,
        )
        try:
            response = await asyncio.to_thread(
                self._transport.roundtrip,
                wire,
                timeout_seconds=self._config.request_timeout_seconds,
                connect_timeout_seconds=self._config.connect_timeout_seconds,
            )
        except DelegatedExecutionTransportError:
            raise
        if response.status is WorkerResponseStatus.ERROR:
            if response.error_code == "NOT_FOUND":
                return DelegatedExecutionProviderReattachmentObservation(
                    provider_id=self._provider_id,
                    invocation_id=inv.invocation_id,
                    provider_request_id=inv.provider_request_id,
                    provider_operation_id=inv.provider_operation_id,
                    kind=DelegatedExecutionReattachmentKind.OPERATION_NOT_FOUND,
                )
            raise DelegatedExecutionProviderError(_PROVIDER_FAILURE_MESSAGE)
        kind = DelegatedExecutionReattachmentKind.REATTACHED
        if response.reattachment_kind == "already_attached":
            kind = DelegatedExecutionReattachmentKind.ALREADY_ATTACHED
        return DelegatedExecutionProviderReattachmentObservation(
            provider_id=self._provider_id,
            invocation_id=response.invocation_id or "",
            provider_request_id=response.provider_request_id,
            provider_operation_id=response.provider_operation_id,
            kind=kind,
        )

    async def _control(
        self,
        request: DelegatedExecutionControlRequest,
        *,
        operation: DelegatedExecutionControlOperation,
    ) -> DelegatedExecutionControlOutcome:
        inv = request.provider_invocation
        wire = WorkerRequest(
            method=WorkerRpcMethod.CANCEL,
            request_id=f"wire-{uuid4().hex}",
            auth_token=self._config.connection_auth_token,
            provider_request_id=inv.provider_request_id,
            provider_operation_id=inv.provider_operation_id,
            invocation_id=inv.invocation_id,
        )
        try:
            response = await asyncio.to_thread(
                self._transport.roundtrip,
                wire,
                timeout_seconds=self._config.request_timeout_seconds,
                connect_timeout_seconds=self._config.connect_timeout_seconds,
            )
        except DelegatedExecutionTransportError:
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.TRANSPORT_FAILURE,
                request=request,
                provider_id=self._provider_id,
                failure_code="TRANSPORT_FAILURE",
                failure_message=_TRANSPORT_FAILURE_MESSAGE,
            )
        if response.status is WorkerResponseStatus.ERROR:
            if response.error_code == "NOT_FOUND":
                return delegated_control_outcome(
                    category=DelegatedExecutionControlOutcomeCategory.NOT_FOUND,
                    request=request,
                    provider_id=self._provider_id,
                )
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.PROVIDER_FAILURE,
                request=request,
                provider_id=self._provider_id,
                failure_code=response.error_code or "PROVIDER_FAILURE",
                failure_message=_PROVIDER_FAILURE_MESSAGE,
            )
        return delegated_control_outcome(
            category=DelegatedExecutionControlOutcomeCategory.COMPLETED,
            request=request,
            provider_id=self._provider_id,
        )

    async def worker_execute_count(self) -> int:
        wire = WorkerRequest(
            method=WorkerRpcMethod.METRICS,
            request_id=f"wire-{uuid4().hex}",
            auth_token=self._config.connection_auth_token,
        )
        response = await asyncio.to_thread(
            self._transport.roundtrip,
            wire,
            timeout_seconds=self._config.request_timeout_seconds,
            connect_timeout_seconds=self._config.connect_timeout_seconds,
        )
        return int(response.execute_count or 0)


def _coerce_payload(payload: object) -> SubprocessEchoPayload:
    if not isinstance(payload, SubprocessEchoPayload):
        raise DelegatedExecutionContractError("unsupported subprocess delegated payload")
    return payload


def _failed_outcome(
    *,
    invocation_id: str,
    started_at: datetime,
    error_code: str,
    provider_request_id: str | None,
    provider_operation_id: str | None,
) -> ProviderInvocationOutcome:
    return ProviderInvocationOutcome(
        invocation_id=invocation_id,
        status=ProviderInvocationStatus.FAILED,
        completed_at=datetime.now(timezone.utc),
        error_code=error_code,
        provider_request_id=provider_request_id,
        provider_operation_id=provider_operation_id,
    )


def _outcome_unknown(
    *,
    invocation: object,
    invocation_id: str,
    started_at: datetime,
    provider_request_id: str,
    provider_operation_id: str,
) -> DelegatedExecutionOutcome[ResultT]:
    return delegated_failure_outcome(
        category=DelegatedExecutionOutcomeCategory.TRANSPORT_FAILURE,
        failure_code="OUTCOME_UNKNOWN",
        failure_message=_OUTCOME_UNKNOWN_MESSAGE,
        provider_invocation=invocation,  # type: ignore[arg-type]
        provider_outcome=ProviderInvocationOutcome(
            invocation_id=invocation_id,
            status=ProviderInvocationStatus.UNKNOWN,
            completed_at=datetime.now(timezone.utc),
            error_code="OUTCOME_UNKNOWN",
            provider_request_id=provider_request_id,
            provider_operation_id=provider_operation_id,
        ),
    )


def _map_physical_status(value: str | None) -> DelegatedExecutionProviderPhysicalStatus:
    if value == "running":
        return DelegatedExecutionProviderPhysicalStatus.RUNNING
    if value == "succeeded":
        return DelegatedExecutionProviderPhysicalStatus.SUCCEEDED
    if value == "cancelled":
        return DelegatedExecutionProviderPhysicalStatus.CANCELLED
    if value == "failed":
        return DelegatedExecutionProviderPhysicalStatus.FAILED
    return DelegatedExecutionProviderPhysicalStatus.UNKNOWN
