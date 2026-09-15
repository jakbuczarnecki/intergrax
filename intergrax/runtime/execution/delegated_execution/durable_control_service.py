# © Artur Czarnecki. All rights reserved.

"""Durable ExecutionId lookup adapter for delegated control (P2.1-S2C2)."""

from __future__ import annotations


from intergrax.contracts.delegated_execution_control import (
    DelegatedExecutionControlOperation,
    DelegatedExecutionControlOutcome,
    DelegatedExecutionControlOutcomeCategory,
    DelegatedExecutionControlRequest,
    delegated_control_outcome,
)
from intergrax.contracts.delegated_execution_invocation_binding import (
    mint_delegated_execution_invocation_binding,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionBudgetMode,
    DelegatedExecutionBudgetProjection,
    DelegatedExecutionContext,
    DelegatedExecutionOperationMetadata,
)
from intergrax.contracts.delegated_execution_provider_resolver import (
    DelegatedExecutionProviderResolver,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationPersistenceError,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    validate_execution_id,
)
from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.runtime.execution.delegated_execution.control_service import (
    DelegatedExecutionControlService,
)
from intergrax.runtime.execution.delegated_execution.correlation_service import (
    DelegatedInvocationCorrelationLookup,
)

_CORRELATION_NOT_FOUND = "delegated invocation correlation not found"
_SYNTHETIC_DIGEST = "sha256:" + ("00" * 32)


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
    ) -> DelegatedExecutionControlOutcome:
        normalized = validate_execution_id(execution_id)
        try:
            binding = self._correlation_lookup.load_binding_by_execution_id(normalized)
        except DelegatedInvocationCorrelationIntegrityError as exc:
            request = _synthetic_control_request(
                execution_id=normalized,
                operation=operation,
            )
            if str(exc) == _CORRELATION_NOT_FOUND:
                return delegated_control_outcome(
                    category=DelegatedExecutionControlOutcomeCategory.NOT_FOUND,
                    request=request,
                    provider_id="unknown",
                )
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.PROVIDER_FAILURE,
                request=request,
                provider_id="unknown",
                failure_code="CORRELATION_INTEGRITY_FAILURE",
                failure_message=str(exc),
            )
        except DelegatedInvocationCorrelationPersistenceError as exc:
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.TRANSPORT_FAILURE,
                request=_synthetic_control_request(
                    execution_id=normalized,
                    operation=operation,
                ),
                provider_id="unknown",
                failure_code="CORRELATION_PERSISTENCE_UNAVAILABLE",
                failure_message=str(exc),
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
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.PROVIDER_BINDING_MISMATCH,
                request=request,
                provider_id=bound_provider_id,
                failure_code="PROVIDER_UNAVAILABLE",
                failure_message="delegated execution provider is not configured",
            )
        if provider.provider_id != bound_provider_id:
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.PROVIDER_BINDING_MISMATCH,
                request=request,
                provider_id=provider.provider_id,
                failure_code="PROVIDER_BINDING_MISMATCH",
                failure_message="resolved provider does not match bound provider_id",
            )

        control = DelegatedExecutionControlService(provider)
        return await control.apply_control(request)


def _synthetic_control_request(
    *,
    execution_id: ExecutionId,
    operation: DelegatedExecutionControlOperation,
) -> DelegatedExecutionControlRequest:
    """Envelope-only request for lookup failures (binding unavailable)."""
    parent = mint_execution_id()
    run = mint_run_id()
    ctx = DelegatedExecutionContext(
        execution_id=execution_id,
        parent_execution_id=parent,
        run_id=run,
        attempt_id=mint_attempt_id(),
        authority=ParentExecutionAuthority.scoped(("tools.read",)),
        budget=DelegatedExecutionBudgetProjection(
            allocation_mode=DelegatedExecutionBudgetMode.SHARED,
        ),
    )
    op = DelegatedExecutionOperationMetadata.model_validate(
        {"operation": "synthetic_control", "task_id": "synthetic"},
    )
    inv = ProviderInvocation.model_validate(
        {
            "invocation_id": "synthetic-invocation",
            "provider_id": "unknown",
            "operation": op.operation,
            "task_id": op.task_id,
            "run_id": str(run),
            "request_digest": _SYNTHETIC_DIGEST,
            "started_at": "2026-09-07T08:00:00+00:00",
        }
    )
    binding = mint_delegated_execution_invocation_binding(
        context=ctx,
        operation=op,
        payload_digest=_SYNTHETIC_DIGEST,
        provider_invocation=inv,
    )
    return DelegatedExecutionControlRequest(
        invocation_binding=binding,
        operation=operation,
    )


__all__ = ["DelegatedExecutionDurableControlService"]
