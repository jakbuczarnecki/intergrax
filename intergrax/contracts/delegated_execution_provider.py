# © Artur Czarnecki. All rights reserved.

"""Provider-neutral delegated execution contract (P2.1-S1).

Runtime admits child execution identity, authority, and budget before any
``DelegatedExecutionProvider`` dispatch. Providers perform physical dispatch
only — they must not mint canonical ``ExecutionId`` or widen authority/budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.delegation_authority import (
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)

SCHEMA_DELEGATED_EXECUTION_CAPABILITIES_V1: Final = (
    "delegated_execution_capabilities.v1"
)
SCHEMA_DELEGATED_EXECUTION_CONTEXT_V1: Final = "delegated_execution_context.v1"
SCHEMA_DELEGATED_EXECUTION_BUDGET_PROJECTION_V1: Final = (
    "delegated_execution_budget_projection.v1"
)
SCHEMA_DELEGATED_EXECUTION_OPERATION_V1: Final = "delegated_execution_operation.v1"
_NON_EMPTY = Field(min_length=1)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class DelegatedExecutionProviderError(Exception):
    """Base error for delegated execution provider boundary failures."""


class DelegatedExecutionTransportError(DelegatedExecutionProviderError):
    """Transport or connectivity failure reaching the provider backend."""


class DelegatedExecutionCapabilityError(DelegatedExecutionProviderError):
    """Requested provider operation or capability is not supported."""


class DelegatedExecutionContractError(DelegatedExecutionProviderError):
    """Malformed provider request, context, or outcome contract."""


class DelegatedExecutionBudgetMode(StrEnum):
    """Readonly child budget participation mode (UER projection)."""

    SHARED = "shared"
    RESERVED = "reserved"


class DelegatedExecutionOutcomeCategory(StrEnum):
    """Neutral provider outcome categories for runtime adaptation."""

    SUCCESS = "success"
    PROVIDER_FAILURE = "provider_failure"
    TRANSPORT_FAILURE = "transport_failure"
    UNSUPPORTED = "unsupported"


class DelegatedExecutionBudgetBounds(BaseModel):
    """Readonly bounded budget allowance — not a ledger mutation surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_input_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    max_total_tokens: int | None = Field(default=None, ge=0)
    max_llm_calls: int | None = Field(default=None, ge=0)
    max_tool_calls: int | None = Field(default=None, ge=0)
    max_rag_invocations: int | None = Field(default=None, ge=0)
    max_websearch_invocations: int | None = Field(default=None, ge=0)
    max_wall_time_seconds: float | None = Field(default=None, ge=0.0)
    max_planner_iterations: int | None = Field(default=None, ge=0)
    max_replans: int | None = Field(default=None, ge=0)


class DelegatedExecutionBudgetProjection(BaseModel):
    """Effective child budget evidence already resolved by UER."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_budget_projection.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_BUDGET_PROJECTION_V1
    )
    allocation_mode: DelegatedExecutionBudgetMode
    reservation_allowance: DelegatedExecutionBudgetBounds | None = None


class DelegatedExecutionCapabilities(BaseModel):
    """Conservative, deterministic provider capability manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_capabilities.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_CAPABILITIES_V1
    )
    provider_id: str = _NON_EMPTY
    supports_cancel: bool = False
    supports_pause: bool = False
    supports_resume: bool = False
    supports_streaming: bool = False
    supports_interrupt: bool = False

    @field_validator("provider_id")
    @classmethod
    def _strip_provider_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("provider_id must be non-empty")
        return normalized


class DelegatedExecutionContext(BaseModel):
    """Immutable admitted child execution projection for provider dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_context.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_CONTEXT_V1
    )
    execution_id: ExecutionId
    parent_execution_id: ExecutionId
    run_id: RunId
    attempt_id: AttemptId
    authority: ParentExecutionAuthority
    effective_delegation: EffectiveDelegationAuthority | None = None
    budget: DelegatedExecutionBudgetProjection
    correlation_id: str | None = None

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("parent_execution_id", mode="before")
    @classmethod
    def _validate_parent_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)

    @model_validator(mode="after")
    def _reject_identity_alias(self) -> DelegatedExecutionContext:
        if self.execution_id == self.parent_execution_id:
            raise ValueError(
                "execution_id must differ from parent_execution_id for child delegation",
            )
        return self


class DelegatedExecutionOperationMetadata(BaseModel):
    """Typed delegation metadata — no authority or budget request fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_operation.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_OPERATION_V1
    )
    operation: str = _NON_EMPTY
    task_id: str = _NON_EMPTY
    correlation_id: str | None = None
    idempotency_key: str | None = None

    @field_validator("operation", "task_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


@dataclass(frozen=True, slots=True)
class DelegatedExecutionRequest(Generic[RequestT]):
    """Typed provider dispatch carrier for an already-admitted child execution."""

    context: DelegatedExecutionContext
    payload: RequestT
    operation: DelegatedExecutionOperationMetadata


@dataclass(frozen=True, slots=True)
class DelegatedExecutionOutcome(Generic[ResultT]):
    """Neutral provider outcome envelope for runtime adaptation."""

    category: DelegatedExecutionOutcomeCategory
    result: ResultT | None = None
    provider_invocation: ProviderInvocation | None = None
    provider_outcome: ProviderInvocationOutcome | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    provider_status: str | None = None

    def __post_init__(self) -> None:
        if self.category is DelegatedExecutionOutcomeCategory.SUCCESS:
            if self.failure_code is not None or self.failure_message is not None:
                raise DelegatedExecutionContractError(
                    "success outcome cannot carry failure fields",
                )
            return
        if self.result is not None:
            raise DelegatedExecutionContractError(
                "non-success outcome cannot carry result payload",
            )
        if self.category in {
            DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE,
            DelegatedExecutionOutcomeCategory.TRANSPORT_FAILURE,
        } and not (self.failure_code and self.failure_code.strip()):
            raise DelegatedExecutionContractError(
                "provider and transport failures require failure_code",
            )


@runtime_checkable
class DelegatedExecutionProvider(Protocol[RequestT, ResultT]):
    """Provider-neutral delegated execution dispatch surface."""

    @property
    def provider_id(self) -> str:
        """Stable provider identifier."""
        ...

    @property
    def provider_version(self) -> str:
        """Provider implementation version."""
        ...

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        """Declared provider capabilities."""
        ...

    async def execute(
        self,
        request: DelegatedExecutionRequest[RequestT],
    ) -> DelegatedExecutionOutcome[ResultT]:
        """Dispatch admitted child work through the provider backend."""
        ...


def validate_provider_identity(*, provider_id: str, provider_version: str) -> None:
    """Fail-closed validation for provider identity fields."""
    if not provider_id or not provider_id.strip():
        raise DelegatedExecutionContractError("provider_id must be non-empty")
    if not provider_version or not provider_version.strip():
        raise DelegatedExecutionContractError("provider_version must be non-empty")


def assert_provider_native_ids_distinct_from_execution(
    *,
    execution_id: ExecutionId,
    provider_request_id: str | None = None,
    provider_operation_id: str | None = None,
    invocation_id: str | None = None,
) -> None:
    """Reject aliases between provider-native ids and canonical execution identity."""
    canonical = str(execution_id)
    for label, value in (
        ("provider_request_id", provider_request_id),
        ("provider_operation_id", provider_operation_id),
        ("invocation_id", invocation_id),
    ):
        if value is not None and value.strip() == canonical:
            raise DelegatedExecutionContractError(
                f"{label} must not alias canonical execution_id",
            )


def digest_delegated_execution_request(
    *,
    context: DelegatedExecutionContext,
    operation: DelegatedExecutionOperationMetadata,
    payload_digest: str,
) -> str:
    """Stable request digest for provider invocation evidence."""
    material = {
        "execution_id": str(context.execution_id),
        "parent_execution_id": str(context.parent_execution_id),
        "run_id": str(context.run_id),
        "attempt_id": str(context.attempt_id),
        "operation": operation.operation,
        "task_id": operation.task_id,
        "payload_digest": payload_digest,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def mint_delegated_provider_invocation(
    *,
    context: DelegatedExecutionContext,
    operation: DelegatedExecutionOperationMetadata,
    provider_id: str,
    request_digest: str,
    started_at: datetime,
    invocation_id: str,
    provider_request_id: str | None = None,
    provider_operation_id: str | None = None,
) -> ProviderInvocation:
    """Create provider invocation evidence before physical dispatch."""
    if not provider_id or not provider_id.strip():
        raise DelegatedExecutionContractError("provider_id must be non-empty")
    assert_provider_native_ids_distinct_from_execution(
        execution_id=context.execution_id,
        provider_request_id=provider_request_id,
        provider_operation_id=provider_operation_id,
        invocation_id=invocation_id,
    )
    return ProviderInvocation(
        invocation_id=invocation_id,
        provider_id=provider_id,
        operation=operation.operation,
        task_id=operation.task_id,
        run_id=str(context.run_id),
        correlation_id=operation.correlation_id or context.correlation_id,
        idempotency_key=operation.idempotency_key,
        request_digest=request_digest,
        started_at=started_at,
        provider_request_id=provider_request_id,
        provider_operation_id=provider_operation_id,
    )


def delegated_success_outcome(
    *,
    result: ResultT,
    provider_invocation: ProviderInvocation,
    provider_outcome: ProviderInvocationOutcome,
) -> DelegatedExecutionOutcome[ResultT]:
    """Construct a validated success outcome."""
    if provider_outcome.status is not ProviderInvocationStatus.SUCCEEDED:
        raise DelegatedExecutionContractError(
            "success outcome requires SUCCEEDED provider_outcome status",
        )
    return DelegatedExecutionOutcome(
        category=DelegatedExecutionOutcomeCategory.SUCCESS,
        result=result,
        provider_invocation=provider_invocation,
        provider_outcome=provider_outcome,
    )


def delegated_failure_outcome(
    *,
    category: DelegatedExecutionOutcomeCategory,
    failure_code: str,
    failure_message: str,
    provider_invocation: ProviderInvocation | None = None,
    provider_outcome: ProviderInvocationOutcome | None = None,
    provider_status: str | None = None,
) -> DelegatedExecutionOutcome[ResultT]:
    """Construct a validated non-success outcome without leaking vendor exceptions."""
    if category is DelegatedExecutionOutcomeCategory.SUCCESS:
        raise DelegatedExecutionContractError("delegated_failure_outcome cannot be SUCCESS")
    return DelegatedExecutionOutcome(
        category=category,
        provider_invocation=provider_invocation,
        provider_outcome=provider_outcome,
        failure_code=failure_code,
        failure_message=failure_message,
        provider_status=provider_status,
    )


__all__ = [
    "DelegatedExecutionBudgetBounds",
    "DelegatedExecutionBudgetMode",
    "DelegatedExecutionBudgetProjection",
    "DelegatedExecutionCapabilities",
    "DelegatedExecutionCapabilityError",
    "DelegatedExecutionContext",
    "DelegatedExecutionContractError",
    "DelegatedExecutionOperationMetadata",
    "DelegatedExecutionOutcome",
    "DelegatedExecutionOutcomeCategory",
    "DelegatedExecutionProvider",
    "DelegatedExecutionProviderError",
    "DelegatedExecutionRequest",
    "DelegatedExecutionTransportError",
    "assert_provider_native_ids_distinct_from_execution",
    "delegated_failure_outcome",
    "delegated_success_outcome",
    "digest_delegated_execution_request",
    "mint_delegated_provider_invocation",
    "validate_provider_identity",
]
