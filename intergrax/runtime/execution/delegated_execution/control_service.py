# © Artur Czarnecki. All rights reserved.

"""Execution-owned delegated provider control dispatch (P2.1-S2B)."""

from __future__ import annotations

from typing import TypeVar

from intergrax.contracts.delegated_execution_control import (
    DelegatedExecutionCancelProvider,
    DelegatedExecutionControlOperation,
    DelegatedExecutionControlOutcome,
    DelegatedExecutionControlOutcomeCategory,
    DelegatedExecutionControlRequest,
    DelegatedExecutionInterruptProvider,
    delegated_control_outcome,
    provider_control_outcome_matches_request,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionProvider,
    DelegatedExecutionTransportError,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")

_UNSUPPORTED_MESSAGE = "provider does not advertise this control capability"
_BINDING_MISMATCH_MESSAGE = "control request provider does not match bound provider"
_CAPABILITY_CONTRACT_MESSAGE = (
    "provider advertises control capability without implementing control contract"
)
_TRANSPORT_MESSAGE = "delegated execution control transport failed"
_OUTCOME_MISMATCH_MESSAGE = "delegated control outcome failed platform correlation checks"


class DelegatedExecutionControlService:
    """
    Capability-gated provider-native cancel/interrupt dispatch.

    Does not mint ExecutionId or mutate canonical Execution lifecycle.
    """

    __slots__ = ("_provider",)

    def __init__(
        self,
        provider: DelegatedExecutionProvider[RequestT, ResultT],
    ) -> None:
        self._provider = provider

    @property
    def bound_provider_id(self) -> str:
        return self._provider.provider_id

    async def apply_control(
        self,
        request: DelegatedExecutionControlRequest,
    ) -> DelegatedExecutionControlOutcome:
        bound_id = self._provider.provider_id
        invocation = request.provider_invocation
        if invocation.provider_id != bound_id:
            return delegated_control_outcome(
                category=DelegatedExecutionControlOutcomeCategory.PROVIDER_BINDING_MISMATCH,
                request=request,
                provider_id=bound_id,
                failure_code="PROVIDER_BINDING_MISMATCH",
                failure_message=_BINDING_MISMATCH_MESSAGE,
            )

        capabilities = self._provider.capabilities
        operation = request.operation

        if operation is DelegatedExecutionControlOperation.CANCEL:
            if not capabilities.supports_cancel:
                return delegated_control_outcome(
                    category=DelegatedExecutionControlOutcomeCategory.UNSUPPORTED,
                    request=request,
                    provider_id=bound_id,
                    failure_code="CONTROL_UNSUPPORTED",
                    failure_message=_UNSUPPORTED_MESSAGE,
                )
            if not isinstance(self._provider, DelegatedExecutionCancelProvider):
                return delegated_control_outcome(
                    category=DelegatedExecutionControlOutcomeCategory.UNSUPPORTED,
                    request=request,
                    provider_id=bound_id,
                    failure_code="CONTROL_CONTRACT_MISSING",
                    failure_message=_CAPABILITY_CONTRACT_MESSAGE,
                )
            return await _dispatch_cancel(
                self._provider,
                request,
                provider_id=bound_id,
            )

        if operation is DelegatedExecutionControlOperation.INTERRUPT:
            if not capabilities.supports_interrupt:
                return delegated_control_outcome(
                    category=DelegatedExecutionControlOutcomeCategory.UNSUPPORTED,
                    request=request,
                    provider_id=bound_id,
                    failure_code="CONTROL_UNSUPPORTED",
                    failure_message=_UNSUPPORTED_MESSAGE,
                )
            if not isinstance(self._provider, DelegatedExecutionInterruptProvider):
                return delegated_control_outcome(
                    category=DelegatedExecutionControlOutcomeCategory.UNSUPPORTED,
                    request=request,
                    provider_id=bound_id,
                    failure_code="CONTROL_CONTRACT_MISSING",
                    failure_message=_CAPABILITY_CONTRACT_MESSAGE,
                )
            return await _dispatch_interrupt(
                self._provider,
                request,
                provider_id=bound_id,
            )

        return delegated_control_outcome(
            category=DelegatedExecutionControlOutcomeCategory.UNSUPPORTED,
            request=request,
            provider_id=bound_id,
            failure_code="CONTROL_OPERATION_UNKNOWN",
            failure_message=_UNSUPPORTED_MESSAGE,
        )


async def _dispatch_cancel(
    provider: DelegatedExecutionCancelProvider,
    request: DelegatedExecutionControlRequest,
    *,
    provider_id: str,
) -> DelegatedExecutionControlOutcome:
    try:
        provider_outcome = await provider.cancel_delegated_execution(request)
    except DelegatedExecutionTransportError:
        return delegated_control_outcome(
            category=DelegatedExecutionControlOutcomeCategory.TRANSPORT_FAILURE,
            request=request,
            provider_id=provider_id,
            failure_code="TRANSPORT_FAILURE",
            failure_message=_TRANSPORT_MESSAGE,
        )
    return _finalize_provider_control_outcome(
        provider_outcome,
        request=request,
        provider_id=provider_id,
    )


async def _dispatch_interrupt(
    provider: DelegatedExecutionInterruptProvider,
    request: DelegatedExecutionControlRequest,
    *,
    provider_id: str,
) -> DelegatedExecutionControlOutcome:
    try:
        provider_outcome = await provider.interrupt_delegated_execution(request)
    except DelegatedExecutionTransportError:
        return delegated_control_outcome(
            category=DelegatedExecutionControlOutcomeCategory.TRANSPORT_FAILURE,
            request=request,
            provider_id=provider_id,
            failure_code="TRANSPORT_FAILURE",
            failure_message=_TRANSPORT_MESSAGE,
        )
    return _finalize_provider_control_outcome(
        provider_outcome,
        request=request,
        provider_id=provider_id,
    )


def _finalize_provider_control_outcome(
    provider_outcome: DelegatedExecutionControlOutcome,
    *,
    request: DelegatedExecutionControlRequest,
    provider_id: str,
) -> DelegatedExecutionControlOutcome:
    if provider_control_outcome_matches_request(
        outcome=provider_outcome,
        request=request,
        bound_provider_id=provider_id,
    ):
        return provider_outcome
    return delegated_control_outcome(
        category=DelegatedExecutionControlOutcomeCategory.CONTROL_OUTCOME_CONTRACT_MISMATCH,
        request=request,
        provider_id=provider_id,
        failure_code="CONTROL_OUTCOME_CONTRACT_MISMATCH",
        failure_message=_OUTCOME_MISMATCH_MESSAGE,
    )


__all__ = ["DelegatedExecutionControlService"]
