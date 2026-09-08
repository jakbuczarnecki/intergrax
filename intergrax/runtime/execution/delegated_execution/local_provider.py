# © Artur Czarnecki. All rights reserved.

"""Local reference delegated execution provider (P2.1-S1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Generic, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionCapabilities,
    DelegatedExecutionContractError,
    DelegatedExecutionOutcome,
    DelegatedExecutionOutcomeCategory,
    DelegatedExecutionProvider,
    DelegatedExecutionRequest,
    delegated_failure_outcome,
    delegated_success_outcome,
    digest_delegated_execution_request,
    mint_delegated_provider_invocation,
    validate_provider_identity,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")

_LOCAL_PROVIDER_ID = "local_delegated_execution"
_LOCAL_PROVIDER_VERSION = "1.0.0"
_PROVIDER_EXECUTION_FAILED_MESSAGE = "delegated execution provider failed"
_TRANSPORT_TIMEOUT_MESSAGE = "delegated execution transport timed out"
_TRANSPORT_IO_MESSAGE = "delegated execution transport I/O failed"


class LocalDelegatedExecutionDelegate(Protocol[RequestT, ResultT]):
    """Executable local delegate invoked with admitted canonical context."""

    async def execute(
        self,
        request: DelegatedExecutionRequest[RequestT],
    ) -> ResultT:
        ...


class LocalDelegatedExecutionProvider(
    Generic[RequestT, ResultT],
    DelegatedExecutionProvider[RequestT, ResultT],
):
    """Reference provider wrapping an in-process delegate without vendor SDKs."""

    __slots__ = ("_delegate", "_provider_id", "_provider_version", "_capabilities")

    def __init__(
        self,
        delegate: LocalDelegatedExecutionDelegate[RequestT, ResultT],
        *,
        provider_id: str = _LOCAL_PROVIDER_ID,
        provider_version: str = _LOCAL_PROVIDER_VERSION,
    ) -> None:
        validate_provider_identity(
            provider_id=provider_id,
            provider_version=provider_version,
        )
        self._delegate = delegate
        self._provider_id = provider_id.strip()
        self._provider_version = provider_version.strip()
        self._capabilities = DelegatedExecutionCapabilities(
            provider_id=self._provider_id,
            supports_cancel=False,
            supports_pause=False,
            supports_resume=False,
            supports_streaming=False,
            supports_interrupt=False,
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return self._provider_version

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return self._capabilities

    async def execute(
        self,
        request: DelegatedExecutionRequest[RequestT],
    ) -> DelegatedExecutionOutcome[ResultT]:
        started_at = datetime.now(timezone.utc)
        payload_digest = _digest_payload(request.payload)
        request_digest = digest_delegated_execution_request(
            context=request.context,
            operation=request.operation,
            payload_digest=payload_digest,
        )
        invocation_id = f"dep-inv-{uuid4().hex}"
        invocation = mint_delegated_provider_invocation(
            context=request.context,
            operation=request.operation,
            provider_id=self._provider_id,
            request_digest=request_digest,
            started_at=started_at,
            invocation_id=invocation_id,
            provider_request_id=f"dep-req-{uuid4().hex}",
            provider_operation_id=f"dep-op-{uuid4().hex}",
        )
        try:
            result = await self._delegate.execute(request)
        except DelegatedExecutionContractError:
            raise
        except TimeoutError:
            return delegated_failure_outcome(
                category=DelegatedExecutionOutcomeCategory.TRANSPORT_FAILURE,
                failure_code="TRANSPORT_TIMEOUT",
                failure_message=_TRANSPORT_TIMEOUT_MESSAGE,
                provider_invocation=invocation,
                provider_outcome=_failed_outcome(
                    invocation_id=invocation_id,
                    started_at=started_at,
                    error_code="TRANSPORT_TIMEOUT",
                    provider_request_id=invocation.provider_request_id,
                    provider_operation_id=invocation.provider_operation_id,
                ),
            )
        except OSError:
            return delegated_failure_outcome(
                category=DelegatedExecutionOutcomeCategory.TRANSPORT_FAILURE,
                failure_code="TRANSPORT_IO",
                failure_message=_TRANSPORT_IO_MESSAGE,
                provider_invocation=invocation,
                provider_outcome=_failed_outcome(
                    invocation_id=invocation_id,
                    started_at=started_at,
                    error_code="TRANSPORT_IO",
                    provider_request_id=invocation.provider_request_id,
                    provider_operation_id=invocation.provider_operation_id,
                ),
            )
        except Exception:
            return delegated_failure_outcome(
                category=DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE,
                failure_code="PROVIDER_EXECUTION_FAILED",
                failure_message=_PROVIDER_EXECUTION_FAILED_MESSAGE,
                provider_invocation=invocation,
                provider_outcome=_failed_outcome(
                    invocation_id=invocation_id,
                    started_at=started_at,
                    error_code="PROVIDER_EXECUTION_FAILED",
                    provider_request_id=invocation.provider_request_id,
                    provider_operation_id=invocation.provider_operation_id,
                ),
            )

        completed_at = datetime.now(timezone.utc)
        provider_outcome = ProviderInvocationOutcome(
            invocation_id=invocation_id,
            status=ProviderInvocationStatus.SUCCEEDED,
            completed_at=completed_at,
            response_digest=_digest_payload(result),
            provider_request_id=invocation.provider_request_id,
            provider_operation_id=invocation.provider_operation_id,
        )
        return delegated_success_outcome(
            result=result,
            provider_invocation=invocation,
            provider_outcome=provider_outcome,
        )


def _digest_payload(payload: object) -> str:
    coerced = _coerce_payload_value(payload)
    try:
        encoded = json.dumps(coerced, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise DelegatedExecutionContractError(
            "payload is not serializable for delegated execution digest",
        ) from exc
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _coerce_payload_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        coerced: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DelegatedExecutionContractError(
                    "payload map keys must be strings for delegated execution digest",
                )
            coerced[key] = _coerce_payload_value(item)
        return coerced
    if isinstance(value, (list, tuple)):
        return [_coerce_payload_value(item) for item in value]
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _coerce_payload_value(asdict(value))
    raise DelegatedExecutionContractError(
        f"payload type {type(value).__name__} is not supported for delegated execution digest",
    )


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
