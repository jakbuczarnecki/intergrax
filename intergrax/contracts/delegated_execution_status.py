# © Artur Czarnecki. All rights reserved.

"""Provider-native delegated execution status read contract (P2.1-S2C2).

Status observations describe **physical provider-side work** for a delegated
child execution. They do **not** transition or authority canonical Execution
lifecycle state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.delegated_execution_invocation_binding import (
    DelegatedExecutionInvocationBinding,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionContractError,
)
from intergrax.contracts.execution_identity import ExecutionId, RunId
from intergrax.contracts.provider_invocation import ProviderInvocation

SCHEMA_DELEGATED_EXECUTION_STATUS_REQUEST_V1: Final = (
    "delegated_execution_status_request.v1"
)
SCHEMA_DELEGATED_EXECUTION_STATUS_VIEW_V1: Final = (
    "delegated_execution_status_view.v1"
)
_NON_EMPTY = Field(min_length=1)


class DelegatedExecutionProviderPhysicalStatus(StrEnum):
    """Vendor-neutral physical operation status — not canonical Execution status."""

    UNKNOWN = "unknown"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DelegatedExecutionStatusOutcomeCategory(StrEnum):
    """Platform status read outcome — read-only, no lifecycle authority."""

    AVAILABLE = "available"
    UNSUPPORTED = "unsupported"
    CORRELATION_NOT_FOUND = "correlation_not_found"
    CORRELATION_INTEGRITY_FAILURE = "correlation_integrity_failure"
    CORRELATION_PERSISTENCE_UNAVAILABLE = "correlation_persistence_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_BINDING_MISMATCH = "provider_binding_mismatch"
    STATUS_CONTRACT_MISSING = "status_contract_missing"
    STATUS_OUTCOME_CONTRACT_MISMATCH = "status_outcome_contract_mismatch"
    PROVIDER_FAILURE = "provider_failure"
    TRANSPORT_FAILURE = "transport_failure"


class DelegatedExecutionStatusRequest(BaseModel):
    """Status carrier correlated via platform invocation binding (service-built)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_status_request.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_STATUS_REQUEST_V1
    )
    invocation_binding: DelegatedExecutionInvocationBinding

    @property
    def execution_id(self) -> ExecutionId:
        return self.invocation_binding.execution_id

    @property
    def run_id(self) -> RunId:
        return self.invocation_binding.run_id

    @property
    def provider_invocation(self) -> ProviderInvocation:
        return self.invocation_binding.provider_invocation


class DelegatedExecutionStatusView(BaseModel):
    """Typed platform read model for provider-side delegated operation status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_status_view.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_STATUS_VIEW_V1
    )
    execution_id: ExecutionId
    run_id: RunId
    provider_id: str = _NON_EMPTY
    invocation_id: str = _NON_EMPTY
    provider_request_id: str | None = None
    provider_operation_id: str | None = None
    physical_status: DelegatedExecutionProviderPhysicalStatus
    provider_external_status: str | None = None
    provider_completed_at: datetime | None = None
    observed_at: datetime
    availability: Literal["available"] = "available"


@dataclass(frozen=True, slots=True)
class DelegatedExecutionProviderStatusObservation:
    """Untrusted plugin status payload before platform validation."""

    physical_status: DelegatedExecutionProviderPhysicalStatus
    provider_id: str
    invocation_id: str
    provider_request_id: str | None = None
    provider_operation_id: str | None = None
    provider_external_status: str | None = None
    provider_completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class DelegatedExecutionStatusOutcome:
    """Neutral status read result for runtime consumers."""

    category: DelegatedExecutionStatusOutcomeCategory
    execution_id: ExecutionId | None = None
    run_id: RunId | None = None
    provider_id: str | None = None
    view: DelegatedExecutionStatusView | None = None
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if self.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE:
            if self.view is None:
                raise DelegatedExecutionContractError(
                    "AVAILABLE status outcome requires view",
                )
            if self.failure_code is not None or self.failure_message is not None:
                raise DelegatedExecutionContractError(
                    "AVAILABLE status outcome cannot carry failure fields",
                )
            return
        if self.view is not None:
            raise DelegatedExecutionContractError(
                "non-AVAILABLE status outcome cannot carry view",
            )
        if self.category in {
            DelegatedExecutionStatusOutcomeCategory.UNSUPPORTED,
            DelegatedExecutionStatusOutcomeCategory.CORRELATION_NOT_FOUND,
            DelegatedExecutionStatusOutcomeCategory.CORRELATION_INTEGRITY_FAILURE,
            DelegatedExecutionStatusOutcomeCategory.CORRELATION_PERSISTENCE_UNAVAILABLE,
            DelegatedExecutionStatusOutcomeCategory.PROVIDER_UNAVAILABLE,
            DelegatedExecutionStatusOutcomeCategory.PROVIDER_BINDING_MISMATCH,
            DelegatedExecutionStatusOutcomeCategory.STATUS_CONTRACT_MISSING,
            DelegatedExecutionStatusOutcomeCategory.STATUS_OUTCOME_CONTRACT_MISMATCH,
            DelegatedExecutionStatusOutcomeCategory.PROVIDER_FAILURE,
            DelegatedExecutionStatusOutcomeCategory.TRANSPORT_FAILURE,
        } and not (self.failure_code and self.failure_code.strip()):
            raise DelegatedExecutionContractError(
                "status read failure categories require failure_code",
            )


def provider_status_observation_matches_binding(
    *,
    observation: DelegatedExecutionProviderStatusObservation,
    request: DelegatedExecutionStatusRequest,
    bound_provider_id: str,
) -> bool:
    """Reject untrusted plugin observations that spoof platform correlation fields."""
    binding = request.invocation_binding
    inv = binding.provider_invocation
    return (
        observation.provider_id == bound_provider_id
        and observation.invocation_id == inv.invocation_id
        and observation.provider_request_id == inv.provider_request_id
        and observation.provider_operation_id == inv.provider_operation_id
    )


@runtime_checkable
class DelegatedExecutionStatusProvider(Protocol):
    """Optional status read surface — required when ``supports_status_read`` is True."""

    async def read_delegated_execution_status(
        self,
        request: DelegatedExecutionStatusRequest,
    ) -> DelegatedExecutionProviderStatusObservation:
        """Return provider-side physical status for the bound invocation."""
        ...


__all__ = [
    "DelegatedExecutionProviderPhysicalStatus",
    "DelegatedExecutionProviderStatusObservation",
    "DelegatedExecutionStatusOutcome",
    "DelegatedExecutionStatusOutcomeCategory",
    "DelegatedExecutionStatusProvider",
    "DelegatedExecutionStatusRequest",
    "DelegatedExecutionStatusView",
    "provider_status_observation_matches_binding",
]
