# © Artur Czarnecki. All rights reserved.

"""Provider-native delegated execution control contract (P2.1-S2B).

Control operations attempt to stop or interrupt **physical provider-side work**
for an already-admitted delegated child execution. They do **not** transition
canonical Execution lifecycle state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionContractError,
)
from intergrax.contracts.delegated_execution_invocation_binding import (
    DelegatedExecutionInvocationBinding,
)
from intergrax.contracts.execution_identity import ExecutionId, RunId
from intergrax.contracts.provider_invocation import ProviderInvocation

SCHEMA_DELEGATED_EXECUTION_CONTROL_REQUEST_V2: Final = (
    "delegated_execution_control_request.v2"
)


class DelegatedExecutionControlOperation(StrEnum):
    """Platform control-plane operations for provider-native work."""

    CANCEL = "cancel"
    INTERRUPT = "interrupt"


class DelegatedExecutionControlOutcomeCategory(StrEnum):
    """Neutral provider control outcome — not canonical Execution status."""

    ACCEPTED = "accepted"
    COMPLETED = "completed"
    UNSUPPORTED = "unsupported"
    NOT_FOUND = "not_found"
    ALREADY_TERMINAL = "already_terminal"
    PROVIDER_BINDING_MISMATCH = "provider_binding_mismatch"
    CONTROL_OUTCOME_CONTRACT_MISMATCH = "control_outcome_contract_mismatch"
    PROVIDER_FAILURE = "provider_failure"
    TRANSPORT_FAILURE = "transport_failure"


class DelegatedExecutionControlRequest(BaseModel):
    """Typed control carrier correlated via platform invocation binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_control_request.v2"] = (
        SCHEMA_DELEGATED_EXECUTION_CONTROL_REQUEST_V2
    )
    invocation_binding: DelegatedExecutionInvocationBinding
    operation: DelegatedExecutionControlOperation
    reason_code: str | None = None
    reason_message: str | None = None

    @property
    def execution_id(self) -> ExecutionId:
        return self.invocation_binding.execution_id

    @property
    def run_id(self) -> RunId:
        return self.invocation_binding.run_id

    @property
    def provider_invocation(self) -> ProviderInvocation:
        return self.invocation_binding.provider_invocation


@dataclass(frozen=True, slots=True)
class DelegatedExecutionControlOutcome:
    """Provider-neutral control result for runtime evidence and adaptation."""

    category: DelegatedExecutionControlOutcomeCategory
    operation: DelegatedExecutionControlOperation
    execution_id: ExecutionId
    run_id: RunId
    provider_id: str
    invocation_id: str
    provider_request_id: str | None = None
    provider_operation_id: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if self.category in {
            DelegatedExecutionControlOutcomeCategory.PROVIDER_FAILURE,
            DelegatedExecutionControlOutcomeCategory.TRANSPORT_FAILURE,
            DelegatedExecutionControlOutcomeCategory.UNSUPPORTED,
            DelegatedExecutionControlOutcomeCategory.PROVIDER_BINDING_MISMATCH,
            DelegatedExecutionControlOutcomeCategory.CONTROL_OUTCOME_CONTRACT_MISMATCH,
        } and not (self.failure_code and self.failure_code.strip()):
            raise DelegatedExecutionContractError(
                "control failure categories require failure_code",
            )
        if self.category in {
            DelegatedExecutionControlOutcomeCategory.ACCEPTED,
            DelegatedExecutionControlOutcomeCategory.COMPLETED,
            DelegatedExecutionControlOutcomeCategory.NOT_FOUND,
            DelegatedExecutionControlOutcomeCategory.ALREADY_TERMINAL,
        }:
            if self.failure_code is not None or self.failure_message is not None:
                raise DelegatedExecutionContractError(
                    "non-failure control outcomes cannot carry failure fields",
                )


class DelegatedExecutionDurableControlOutcomeCategory(StrEnum):
    """Durable control adapter outcome — lookup failures vs resolved S2B control."""

    RESOLVED_CONTROL = "resolved_control"
    CORRELATION_NOT_FOUND = "correlation_not_found"
    CORRELATION_INTEGRITY_FAILURE = "correlation_integrity_failure"
    CORRELATION_PERSISTENCE_UNAVAILABLE = "correlation_persistence_unavailable"


@dataclass(frozen=True, slots=True)
class DelegatedExecutionDurableControlOutcome:
    """ExecutionId lookup envelope without fabricating invocation binding."""

    category: DelegatedExecutionDurableControlOutcomeCategory
    requested_execution_id: ExecutionId
    requested_operation: DelegatedExecutionControlOperation
    control_outcome: DelegatedExecutionControlOutcome | None = None
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if self.category is DelegatedExecutionDurableControlOutcomeCategory.RESOLVED_CONTROL:
            if self.control_outcome is None:
                raise DelegatedExecutionContractError(
                    "resolved durable control requires control_outcome",
                )
            if self.failure_code is not None or self.failure_message is not None:
                raise DelegatedExecutionContractError(
                    "resolved durable control cannot carry failure fields",
                )
            return
        if self.control_outcome is not None:
            raise DelegatedExecutionContractError(
                "lookup-failure durable control cannot carry control_outcome",
            )
        if not (self.failure_code and self.failure_code.strip()):
            raise DelegatedExecutionContractError(
                "lookup-failure durable control requires failure_code",
            )
        if not (self.failure_message and self.failure_message.strip()):
            raise DelegatedExecutionContractError(
                "lookup-failure durable control requires failure_message",
            )


def durable_control_lookup_failure(
    *,
    category: DelegatedExecutionDurableControlOutcomeCategory,
    requested_execution_id: ExecutionId,
    requested_operation: DelegatedExecutionControlOperation,
    failure_code: str,
    failure_message: str,
) -> DelegatedExecutionDurableControlOutcome:
    """Typed durable control outcome when binding cannot be loaded."""
    return DelegatedExecutionDurableControlOutcome(
        category=category,
        requested_execution_id=requested_execution_id,
        requested_operation=requested_operation,
        failure_code=failure_code,
        failure_message=failure_message,
    )


def durable_control_resolved(
    *,
    requested_execution_id: ExecutionId,
    requested_operation: DelegatedExecutionControlOperation,
    control_outcome: DelegatedExecutionControlOutcome,
) -> DelegatedExecutionDurableControlOutcome:
    """Wrap an S2B control outcome after successful durable binding lookup."""
    return DelegatedExecutionDurableControlOutcome(
        category=DelegatedExecutionDurableControlOutcomeCategory.RESOLVED_CONTROL,
        requested_execution_id=requested_execution_id,
        requested_operation=requested_operation,
        control_outcome=control_outcome,
    )


def delegated_control_outcome(
    *,
    category: DelegatedExecutionControlOutcomeCategory,
    request: DelegatedExecutionControlRequest,
    provider_id: str,
    failure_code: str | None = None,
    failure_message: str | None = None,
) -> DelegatedExecutionControlOutcome:
    """Construct a platform-owned control outcome correlated to the request."""
    binding = request.invocation_binding
    inv = binding.provider_invocation
    return DelegatedExecutionControlOutcome(
        category=category,
        operation=request.operation,
        execution_id=binding.execution_id,
        run_id=binding.run_id,
        provider_id=provider_id,
        invocation_id=inv.invocation_id,
        provider_request_id=inv.provider_request_id,
        provider_operation_id=inv.provider_operation_id,
        failure_code=failure_code,
        failure_message=failure_message,
    )


def provider_control_outcome_matches_request(
    *,
    outcome: DelegatedExecutionControlOutcome,
    request: DelegatedExecutionControlRequest,
    bound_provider_id: str,
) -> bool:
    """Reject untrusted plugin outcomes that spoof platform correlation fields."""
    binding = request.invocation_binding
    inv = binding.provider_invocation
    return (
        outcome.execution_id == binding.execution_id
        and outcome.run_id == binding.run_id
        and outcome.provider_id == bound_provider_id
        and outcome.invocation_id == inv.invocation_id
        and outcome.provider_request_id == inv.provider_request_id
        and outcome.provider_operation_id == inv.provider_operation_id
        and outcome.operation == request.operation
    )


@runtime_checkable
class DelegatedExecutionCancelProvider(Protocol):
    """Optional cancel control surface — required when ``supports_cancel`` is True."""

    async def cancel_delegated_execution(
        self,
        request: DelegatedExecutionControlRequest,
    ) -> DelegatedExecutionControlOutcome:
        """Attempt to terminate provider-side work for the given invocation."""
        ...


@runtime_checkable
class DelegatedExecutionInterruptProvider(Protocol):
    """Optional interrupt control surface — required when ``supports_interrupt`` is True."""

    async def interrupt_delegated_execution(
        self,
        request: DelegatedExecutionControlRequest,
    ) -> DelegatedExecutionControlOutcome:
        """Attempt to stop current provider activity without implying canonical cancel."""
        ...


__all__ = [
    "DelegatedExecutionCancelProvider",
    "DelegatedExecutionControlOperation",
    "DelegatedExecutionControlOutcome",
    "DelegatedExecutionControlOutcomeCategory",
    "DelegatedExecutionControlRequest",
    "DelegatedExecutionDurableControlOutcome",
    "DelegatedExecutionDurableControlOutcomeCategory",
    "DelegatedExecutionInterruptProvider",
    "delegated_control_outcome",
    "durable_control_lookup_failure",
    "durable_control_resolved",
    "provider_control_outcome_matches_request",
]
