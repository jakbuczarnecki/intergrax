# © Artur Czarnecki. All rights reserved.

"""Execution-owned delegated provider status read (P2.1-S2C2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionTransportError,
)
from intergrax.contracts.delegated_execution_provider_resolver import (
    DelegatedExecutionProviderResolver,
)
from intergrax.contracts.delegated_execution_status import (
    DelegatedExecutionStatusOutcome,
    DelegatedExecutionStatusOutcomeCategory,
    DelegatedExecutionStatusProvider,
    DelegatedExecutionStatusRequest,
    DelegatedExecutionStatusView,
    DelegatedExecutionProviderStatusObservation,
    provider_status_observation_matches_binding,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DELEGATED_INVOCATION_CORRELATION_INTEGRITY_FAILURE_MESSAGE,
    DELEGATED_INVOCATION_CORRELATION_NOT_FOUND_MESSAGE,
    DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationNotFoundError,
    DelegatedInvocationCorrelationPersistenceError,
)
from intergrax.contracts.execution_identity import ExecutionId, RunId, validate_execution_id
from intergrax.runtime.execution.delegated_execution.correlation_service import (
    DelegatedInvocationCorrelationLookup,
)

_UNSUPPORTED_MESSAGE = "provider does not advertise status read capability"
_BINDING_MISMATCH_MESSAGE = "resolved provider does not match bound provider_id"
_CONTRACT_MESSAGE = (
    "provider advertises status read without implementing status contract"
)
_TRANSPORT_MESSAGE = "delegated execution status transport failed"
_OUTCOME_MISMATCH_MESSAGE = "delegated status observation failed platform correlation checks"


class DelegatedExecutionStatusReadService:
    """
    Durable ExecutionId → binding → provider status → typed read model.

    Does not mint ExecutionId or mutate canonical Execution lifecycle.
    """

    __slots__ = ("_clock", "_correlation_lookup", "_resolver")

    def __init__(
        self,
        correlation_lookup: DelegatedInvocationCorrelationLookup,
        resolver: DelegatedExecutionProviderResolver,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._correlation_lookup = correlation_lookup
        self._resolver = resolver
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def read_status_by_execution_id(
        self,
        execution_id: ExecutionId,
    ) -> DelegatedExecutionStatusOutcome:
        normalized = validate_execution_id(execution_id)
        try:
            binding = self._correlation_lookup.load_binding_by_execution_id(normalized)
        except DelegatedInvocationCorrelationNotFoundError:
            return _status_failure(
                category=DelegatedExecutionStatusOutcomeCategory.CORRELATION_NOT_FOUND,
                execution_id=normalized,
                failure_code="CORRELATION_NOT_FOUND",
                failure_message=DELEGATED_INVOCATION_CORRELATION_NOT_FOUND_MESSAGE,
            )
        except DelegatedInvocationCorrelationIntegrityError:
            return _status_failure(
                category=DelegatedExecutionStatusOutcomeCategory.CORRELATION_INTEGRITY_FAILURE,
                execution_id=normalized,
                failure_code="CORRELATION_INTEGRITY_FAILURE",
                failure_message=DELEGATED_INVOCATION_CORRELATION_INTEGRITY_FAILURE_MESSAGE,
            )
        except DelegatedInvocationCorrelationPersistenceError:
            return _status_failure(
                category=DelegatedExecutionStatusOutcomeCategory.CORRELATION_PERSISTENCE_UNAVAILABLE,
                execution_id=normalized,
                failure_code="CORRELATION_PERSISTENCE_UNAVAILABLE",
                failure_message=DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
            )

        bound_provider_id = binding.provider_invocation.provider_id
        provider = self._resolver.resolve(bound_provider_id)
        if provider is None:
            return _status_failure(
                category=DelegatedExecutionStatusOutcomeCategory.PROVIDER_UNAVAILABLE,
                execution_id=binding.execution_id,
                run_id=binding.run_id,
                provider_id=bound_provider_id,
                failure_code="PROVIDER_UNAVAILABLE",
                failure_message="delegated execution provider is not configured",
            )
        if provider.provider_id != bound_provider_id:
            return _status_failure(
                category=DelegatedExecutionStatusOutcomeCategory.PROVIDER_BINDING_MISMATCH,
                execution_id=binding.execution_id,
                run_id=binding.run_id,
                provider_id=bound_provider_id,
                failure_code="PROVIDER_BINDING_MISMATCH",
                failure_message=_BINDING_MISMATCH_MESSAGE,
            )

        capabilities = provider.capabilities
        if not capabilities.supports_status_read:
            return _status_failure(
                category=DelegatedExecutionStatusOutcomeCategory.UNSUPPORTED,
                execution_id=binding.execution_id,
                run_id=binding.run_id,
                provider_id=bound_provider_id,
                failure_code="STATUS_UNSUPPORTED",
                failure_message=_UNSUPPORTED_MESSAGE,
            )
        if not isinstance(provider, DelegatedExecutionStatusProvider):
            return _status_failure(
                category=DelegatedExecutionStatusOutcomeCategory.STATUS_CONTRACT_MISSING,
                execution_id=binding.execution_id,
                run_id=binding.run_id,
                provider_id=bound_provider_id,
                failure_code="STATUS_CONTRACT_MISSING",
                failure_message=_CONTRACT_MESSAGE,
            )

        request = DelegatedExecutionStatusRequest(invocation_binding=binding)
        return await _dispatch_status_read(
            provider,
            request,
            bound_provider_id=bound_provider_id,
            observed_at=self._clock(),
        )


async def _dispatch_status_read(
    provider: DelegatedExecutionStatusProvider,
    request: DelegatedExecutionStatusRequest,
    *,
    bound_provider_id: str,
    observed_at: datetime,
) -> DelegatedExecutionStatusOutcome:
    binding = request.invocation_binding
    inv = binding.provider_invocation
    try:
        observation = await provider.read_delegated_execution_status(request)
    except DelegatedExecutionTransportError:
        return _status_failure(
            category=DelegatedExecutionStatusOutcomeCategory.TRANSPORT_FAILURE,
            execution_id=binding.execution_id,
            run_id=binding.run_id,
            provider_id=bound_provider_id,
            failure_code="TRANSPORT_FAILURE",
            failure_message=_TRANSPORT_MESSAGE,
        )
    except Exception:
        return _status_failure(
            category=DelegatedExecutionStatusOutcomeCategory.PROVIDER_FAILURE,
            execution_id=binding.execution_id,
            run_id=binding.run_id,
            provider_id=bound_provider_id,
            failure_code="PROVIDER_STATUS_FAILED",
            failure_message="delegated execution provider status read failed",
        )

    return _finalize_status_observation(
        observation,
        request=request,
        bound_provider_id=bound_provider_id,
        observed_at=observed_at,
    )


def _finalize_status_observation(
    observation: DelegatedExecutionProviderStatusObservation,
    *,
    request: DelegatedExecutionStatusRequest,
    bound_provider_id: str,
    observed_at: datetime,
) -> DelegatedExecutionStatusOutcome:
    binding = request.invocation_binding
    inv = binding.provider_invocation
    if not provider_status_observation_matches_binding(
        observation=observation,
        request=request,
        bound_provider_id=bound_provider_id,
    ):
        return _status_failure(
            category=DelegatedExecutionStatusOutcomeCategory.STATUS_OUTCOME_CONTRACT_MISMATCH,
            execution_id=binding.execution_id,
            run_id=binding.run_id,
            provider_id=bound_provider_id,
            failure_code="STATUS_OUTCOME_CONTRACT_MISMATCH",
            failure_message=_OUTCOME_MISMATCH_MESSAGE,
        )
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    view = DelegatedExecutionStatusView(
        execution_id=binding.execution_id,
        run_id=binding.run_id,
        provider_id=bound_provider_id,
        invocation_id=inv.invocation_id,
        provider_request_id=inv.provider_request_id,
        provider_operation_id=inv.provider_operation_id,
        physical_status=observation.physical_status,
        provider_external_status=observation.provider_external_status,
        provider_completed_at=observation.provider_completed_at,
        observed_at=observed_at,
    )
    return DelegatedExecutionStatusOutcome(
        category=DelegatedExecutionStatusOutcomeCategory.AVAILABLE,
        execution_id=binding.execution_id,
        run_id=binding.run_id,
        provider_id=bound_provider_id,
        view=view,
    )


def _status_failure(
    *,
    category: DelegatedExecutionStatusOutcomeCategory,
    failure_code: str,
    failure_message: str,
    execution_id: ExecutionId | None = None,
    run_id: RunId | None = None,
    provider_id: str | None = None,
) -> DelegatedExecutionStatusOutcome:
    return DelegatedExecutionStatusOutcome(
        category=category,
        execution_id=execution_id,
        run_id=run_id,
        provider_id=provider_id,
        failure_code=failure_code,
        failure_message=failure_message,
    )


__all__ = ["DelegatedExecutionStatusReadService"]
