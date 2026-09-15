# © Artur Czarnecki. All rights reserved.

"""Execution-owned delegated provider reattachment (P2.1-S2C4)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from intergrax.contracts.delegated_execution_continuation import (
    DelegatedExecutionContinuationOutcome,
    DelegatedExecutionContinuationOutcomeCategory,
    DelegatedExecutionContinuationOutcomeUnknownError,
    DelegatedExecutionContinuationRequest,
    DelegatedExecutionContinuationView,
    DelegatedExecutionProviderReattachmentObservation,
    DelegatedExecutionReattachmentKind,
    DelegatedExecutionReattachmentProvider,
    continuation_outcome_category_for_kind,
    provider_reattachment_observation_matches_binding,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionTransportError,
)
from intergrax.contracts.delegated_execution_provider_resolver import (
    DelegatedExecutionProviderResolver,
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

_UNSUPPORTED_MESSAGE = "provider does not advertise reattachment capability"
_BINDING_MISMATCH_MESSAGE = "resolved provider does not match bound provider_id"
_CONTRACT_MESSAGE = (
    "provider advertises reattachment without implementing reattachment contract"
)
_TRANSPORT_MESSAGE = "delegated execution reattachment transport failed"
_OUTCOME_MISMATCH_MESSAGE = (
    "delegated reattachment observation failed platform correlation checks"
)
_PROVIDER_FAILURE_MESSAGE = "delegated execution provider reattachment failed"
_UNKNOWN_OUTCOME_MESSAGE = (
    "delegated execution reattachment outcome could not be determined"
)


class DelegatedExecutionContinuationService:
    """
    Durable ExecutionId → binding → provider reattachment → typed outcome.

    Does not mint ExecutionId, persist correlation, or mutate canonical lifecycle.
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

    async def reattach_by_execution_id(
        self,
        execution_id: ExecutionId,
    ) -> DelegatedExecutionContinuationOutcome:
        normalized = validate_execution_id(execution_id)
        try:
            binding = self._correlation_lookup.load_binding_by_execution_id(normalized)
        except DelegatedInvocationCorrelationNotFoundError:
            return _continuation_failure(
                category=DelegatedExecutionContinuationOutcomeCategory.CORRELATION_NOT_FOUND,
                execution_id=normalized,
                failure_code="CORRELATION_NOT_FOUND",
                failure_message=DELEGATED_INVOCATION_CORRELATION_NOT_FOUND_MESSAGE,
            )
        except DelegatedInvocationCorrelationIntegrityError:
            return _continuation_failure(
                category=DelegatedExecutionContinuationOutcomeCategory.CORRELATION_INTEGRITY_FAILURE,
                execution_id=normalized,
                failure_code="CORRELATION_INTEGRITY_FAILURE",
                failure_message=DELEGATED_INVOCATION_CORRELATION_INTEGRITY_FAILURE_MESSAGE,
            )
        except DelegatedInvocationCorrelationPersistenceError:
            return _continuation_failure(
                category=(
                    DelegatedExecutionContinuationOutcomeCategory.CORRELATION_PERSISTENCE_UNAVAILABLE
                ),
                execution_id=normalized,
                failure_code="CORRELATION_PERSISTENCE_UNAVAILABLE",
                failure_message=DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
            )

        bound_provider_id = binding.provider_invocation.provider_id
        provider = self._resolver.resolve(bound_provider_id)
        if provider is None:
            return _continuation_failure(
                category=DelegatedExecutionContinuationOutcomeCategory.PROVIDER_UNAVAILABLE,
                execution_id=binding.execution_id,
                run_id=binding.run_id,
                provider_id=bound_provider_id,
                failure_code="PROVIDER_UNAVAILABLE",
                failure_message="delegated execution provider is not configured",
            )
        if provider.provider_id != bound_provider_id:
            return _continuation_failure(
                category=DelegatedExecutionContinuationOutcomeCategory.PROVIDER_BINDING_MISMATCH,
                execution_id=binding.execution_id,
                run_id=binding.run_id,
                provider_id=bound_provider_id,
                failure_code="PROVIDER_BINDING_MISMATCH",
                failure_message=_BINDING_MISMATCH_MESSAGE,
            )

        capabilities = provider.capabilities
        if not capabilities.supports_reattachment:
            return _continuation_failure(
                category=DelegatedExecutionContinuationOutcomeCategory.UNSUPPORTED,
                execution_id=binding.execution_id,
                run_id=binding.run_id,
                provider_id=bound_provider_id,
                failure_code="UNSUPPORTED",
                failure_message=_UNSUPPORTED_MESSAGE,
            )
        if not isinstance(provider, DelegatedExecutionReattachmentProvider):
            return _continuation_failure(
                category=DelegatedExecutionContinuationOutcomeCategory.CONTINUATION_CONTRACT_MISSING,
                execution_id=binding.execution_id,
                run_id=binding.run_id,
                provider_id=bound_provider_id,
                failure_code="CONTINUATION_CONTRACT_MISSING",
                failure_message=_CONTRACT_MESSAGE,
            )

        request = DelegatedExecutionContinuationRequest(invocation_binding=binding)
        return await _dispatch_reattachment(
            provider,
            request,
            bound_provider_id=bound_provider_id,
            observed_at=self._clock(),
        )


async def _dispatch_reattachment(
    provider: DelegatedExecutionReattachmentProvider,
    request: DelegatedExecutionContinuationRequest,
    *,
    bound_provider_id: str,
    observed_at: datetime,
) -> DelegatedExecutionContinuationOutcome:
    binding = request.invocation_binding
    try:
        observation = await provider.reattach_delegated_execution(request)
    except DelegatedExecutionContinuationOutcomeUnknownError:
        return _continuation_failure(
            category=DelegatedExecutionContinuationOutcomeCategory.CONTINUATION_OUTCOME_UNKNOWN,
            execution_id=binding.execution_id,
            run_id=binding.run_id,
            provider_id=bound_provider_id,
            failure_code="CONTINUATION_OUTCOME_UNKNOWN",
            failure_message=_UNKNOWN_OUTCOME_MESSAGE,
        )
    except DelegatedExecutionTransportError:
        return _continuation_failure(
            category=DelegatedExecutionContinuationOutcomeCategory.TRANSPORT_FAILURE,
            execution_id=binding.execution_id,
            run_id=binding.run_id,
            provider_id=bound_provider_id,
            failure_code="TRANSPORT_FAILURE",
            failure_message=_TRANSPORT_MESSAGE,
        )
    except Exception:
        return _continuation_failure(
            category=DelegatedExecutionContinuationOutcomeCategory.PROVIDER_FAILURE,
            execution_id=binding.execution_id,
            run_id=binding.run_id,
            provider_id=bound_provider_id,
            failure_code="PROVIDER_FAILURE",
            failure_message=_PROVIDER_FAILURE_MESSAGE,
        )

    return _finalize_reattachment_observation(
        observation,
        request=request,
        bound_provider_id=bound_provider_id,
        observed_at=observed_at,
    )


def _finalize_reattachment_observation(
    observation: DelegatedExecutionProviderReattachmentObservation,
    *,
    request: DelegatedExecutionContinuationRequest,
    bound_provider_id: str,
    observed_at: datetime,
) -> DelegatedExecutionContinuationOutcome:
    binding = request.invocation_binding
    inv = binding.provider_invocation
    if not provider_reattachment_observation_matches_binding(
        observation=observation,
        request=request,
        bound_provider_id=bound_provider_id,
    ):
        return _continuation_failure(
            category=(
                DelegatedExecutionContinuationOutcomeCategory.CONTINUATION_OUTCOME_CONTRACT_MISMATCH
            ),
            execution_id=binding.execution_id,
            run_id=binding.run_id,
            provider_id=bound_provider_id,
            failure_code="CONTINUATION_OUTCOME_CONTRACT_MISMATCH",
            failure_message=_OUTCOME_MISMATCH_MESSAGE,
        )

    if observation.kind is DelegatedExecutionReattachmentKind.OPERATION_NOT_FOUND:
        return _continuation_failure(
            category=DelegatedExecutionContinuationOutcomeCategory.PROVIDER_OPERATION_NOT_FOUND,
            execution_id=binding.execution_id,
            run_id=binding.run_id,
            provider_id=bound_provider_id,
            failure_code="PROVIDER_OPERATION_NOT_FOUND",
            failure_message="provider-side delegated operation was not found",
        )

    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")

    category = continuation_outcome_category_for_kind(observation.kind)
    view = DelegatedExecutionContinuationView(
        execution_id=binding.execution_id,
        run_id=binding.run_id,
        provider_id=bound_provider_id,
        invocation_id=inv.invocation_id,
        provider_request_id=inv.provider_request_id,
        provider_operation_id=inv.provider_operation_id,
        reattachment_kind=observation.kind,
        physical_status=observation.physical_status,
        provider_external_status=observation.provider_external_status,
        observed_at=observed_at,
    )
    return DelegatedExecutionContinuationOutcome(
        category=category,
        execution_id=binding.execution_id,
        run_id=binding.run_id,
        provider_id=bound_provider_id,
        view=view,
    )


def _continuation_failure(
    *,
    category: DelegatedExecutionContinuationOutcomeCategory,
    failure_code: str,
    failure_message: str,
    execution_id: ExecutionId | None = None,
    run_id: RunId | None = None,
    provider_id: str | None = None,
) -> DelegatedExecutionContinuationOutcome:
    return DelegatedExecutionContinuationOutcome(
        category=category,
        execution_id=execution_id,
        run_id=run_id,
        provider_id=provider_id,
        failure_code=failure_code,
        failure_message=failure_message,
    )


__all__ = ["DelegatedExecutionContinuationService"]
