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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionContractError,
    assert_provider_native_ids_distinct_from_execution,
)
from intergrax.contracts.execution_identity import (
    ExecutionId,
    RunId,
    validate_execution_id,
    validate_run_id,
)
from intergrax.contracts.provider_invocation import ProviderInvocation

SCHEMA_DELEGATED_EXECUTION_CONTROL_REQUEST_V1: Final = (
    "delegated_execution_control_request.v1"
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
    PROVIDER_FAILURE = "provider_failure"
    TRANSPORT_FAILURE = "transport_failure"


class DelegatedExecutionControlRequest(BaseModel):
    """Typed control carrier for an existing provider invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_control_request.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_CONTROL_REQUEST_V1
    )
    execution_id: ExecutionId
    run_id: RunId
    provider_invocation: ProviderInvocation
    operation: DelegatedExecutionControlOperation
    reason_code: str | None = None
    reason_message: str | None = None

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId:
        return validate_run_id(value)

    @model_validator(mode="after")
    def _correlate_invocation(self) -> DelegatedExecutionControlRequest:
        inv = self.provider_invocation
        if str(self.run_id) != inv.run_id:
            raise ValueError("provider_invocation.run_id must match control request run_id")
        assert_provider_native_ids_distinct_from_execution(
            execution_id=self.execution_id,
            provider_request_id=inv.provider_request_id,
            provider_operation_id=inv.provider_operation_id,
            invocation_id=inv.invocation_id,
        )
        return self


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


def delegated_control_outcome(
    *,
    category: DelegatedExecutionControlOutcomeCategory,
    request: DelegatedExecutionControlRequest,
    provider_id: str,
    failure_code: str | None = None,
    failure_message: str | None = None,
) -> DelegatedExecutionControlOutcome:
    """Construct a validated control outcome correlated to the request."""
    inv = request.provider_invocation
    return DelegatedExecutionControlOutcome(
        category=category,
        operation=request.operation,
        execution_id=request.execution_id,
        run_id=request.run_id,
        provider_id=provider_id,
        invocation_id=inv.invocation_id,
        provider_request_id=inv.provider_request_id,
        provider_operation_id=inv.provider_operation_id,
        failure_code=failure_code,
        failure_message=failure_message,
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
    "DelegatedExecutionInterruptProvider",
    "delegated_control_outcome",
]
