# © Artur Czarnecki. All rights reserved.

"""Durable ExecutionId lookup adapter for delegated control (P2.1-S2C2)."""

from __future__ import annotations

from intergrax.contracts.delegated_execution_control import (
    DelegatedExecutionControlOperation,
    DelegatedExecutionControlOutcomeCategory,
    DelegatedExecutionControlRequest,
    DelegatedExecutionDurableControlOutcome,
    DelegatedExecutionDurableControlOutcomeCategory,
    delegated_control_outcome,
    durable_control_lookup_failure,
    durable_control_resolved,
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
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.runtime.execution.delegated_execution.control_service import (
    DelegatedExecutionControlService,
)
from intergrax.runtime.execution.delegated_execution.correlation_service import (
    DelegatedInvocationCorrelationLookup,
)

_PROVIDER_UNAVAILABLE_MESSAGE = "delegated execution provider is not configured"
_PROVIDER_BINDING_MISMATCH_MESSAGE = (
    "resolved provider does not match bound provider_id"
)


class DelegatedExecutionDurableControlService:
    """
    Thin adapter: ExecutionId → durable binding → existing control service.

    Does not duplicate provider control dispatch logic.
    """

    __slots__ = ("_correlation_lookup", "_resolver")

    def __init__(
        self,
        correlation_lookup: DelegatedInvocationCorrelationLookup,
        resolver: DelegatedExecutionProviderResolver,
    ) -> None:
        self._correlation_lookup = correlation_lookup
        self._resolver = resolver

    async def apply_control_by_execution_id(
        self,
        execution_id: ExecutionId,
        operation: DelegatedExecutionControlOperation,
        *,
        reason_code: str | None = None,
        reason_message: str | None = None,
    ) -> DelegatedExecutionDurableControlOutcome:
        normalized = validate_execution_id(execution_id)
        try:
            binding = self._correlation_lookup.load_binding_by_execution_id(normalized)
        except DelegatedInvocationCorrelationNotFoundError:
            return durable_control_lookup_failure(
                category=DelegatedExecutionDurableControlOutcomeCategory.CORRELATION_NOT_FOUND,
                requested_execution_id=normalized,
                requested_operation=operation,
                failure_code="CORRELATION_NOT_FOUND",
                failure_message=DELEGATED_INVOCATION_CORRELATION_NOT_FOUND_MESSAGE,
            )
        except DelegatedInvocationCorrelationIntegrityError:
            return durable_control_lookup_failure(
                category=(
                    DelegatedExecutionDurableControlOutcomeCategory.CORRELATION_INTEGRITY_FAILURE
                ),
                requested_execution_id=normalized,
                requested_operation=operation,
                failure_code="CORRELATION_INTEGRITY_FAILURE",
                failure_message=DELEGATED_INVOCATION_CORRELATION_INTEGRITY_FAILURE_MESSAGE,
            )
        except DelegatedInvocationCorrelationPersistenceError:
            return durable_control_lookup_failure(
                category=(
                    DelegatedExecutionDurableControlOutcomeCategory.CORRELATION_PERSISTENCE_UNAVAILABLE
                ),
                requested_execution_id=normalized,
                requested_operation=operation,
                failure_code="CORRELATION_PERSISTENCE_UNAVAILABLE",
                failure_message=DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
            )

        bound_provider_id = binding.provider_invocation.provider_id
        provider = self._resolver.resolve(bound_provider_id)
        request = DelegatedExecutionControlRequest(
            invocation_binding=binding,
            operation=operation,
            reason_code=reason_code,
            reason_message=reason_message,
        )
        if provider is None:
            return durable_control_resolved(
                requested_execution_id=normalized,
                requested_operation=operation,
                control_outcome=delegated_control_outcome(
                    category=DelegatedExecutionControlOutcomeCategory.PROVIDER_BINDING_MISMATCH,
                    request=request,
                    provider_id=bound_provider_id,
                    failure_code="PROVIDER_UNAVAILABLE",
                    failure_message=_PROVIDER_UNAVAILABLE_MESSAGE,
                ),
            )
        if provider.provider_id != bound_provider_id:
            return durable_control_resolved(
                requested_execution_id=normalized,
                requested_operation=operation,
                control_outcome=delegated_control_outcome(
                    category=DelegatedExecutionControlOutcomeCategory.PROVIDER_BINDING_MISMATCH,
                    request=request,
                    provider_id=provider.provider_id,
                    failure_code="PROVIDER_BINDING_MISMATCH",
                    failure_message=_PROVIDER_BINDING_MISMATCH_MESSAGE,
                ),
            )

        control = DelegatedExecutionControlService(provider)
        outcome = await control.apply_control(request)
        return durable_control_resolved(
            requested_execution_id=normalized,
            requested_operation=operation,
            control_outcome=outcome,
        )


__all__ = ["DelegatedExecutionDurableControlService"]
