# © Artur Czarnecki. All rights reserved.

"""Provider-native delegated execution reattachment contract (P2.1-S2C4).

Reattachment re-establishes platform observation/control over an **existing**
provider-side delegated operation. It does **not** mint canonical execution
identity, mutate durable correlation, or transition Execution lifecycle.
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
    DelegatedExecutionProviderError,
)
from intergrax.contracts.delegated_execution_status import (
    DelegatedExecutionProviderPhysicalStatus,
)
from intergrax.contracts.execution_identity import ExecutionId, RunId
from intergrax.contracts.provider_invocation import ProviderInvocation

SCHEMA_DELEGATED_EXECUTION_CONTINUATION_REQUEST_V1: Final = (
    "delegated_execution_continuation_request.v1"
)
SCHEMA_DELEGATED_EXECUTION_CONTINUATION_VIEW_V1: Final = (
    "delegated_execution_continuation_view.v1"
)
_NON_EMPTY = Field(min_length=1)


class DelegatedExecutionReattachmentKind(StrEnum):
    """Untrusted provider reattachment observation before platform validation."""

    REATTACHED = "reattached"
    ALREADY_ATTACHED = "already_attached"
    OBSERVED_TERMINAL = "observed_terminal"
    OPERATION_NOT_FOUND = "operation_not_found"


class DelegatedExecutionContinuationOutcomeCategory(StrEnum):
    """Platform reattachment outcome — provider-plane facts only."""

    REATTACHED = "reattached"
    ALREADY_ATTACHED = "already_attached"
    OBSERVED_TERMINAL = "observed_terminal"
    UNSUPPORTED = "unsupported"
    CORRELATION_NOT_FOUND = "correlation_not_found"
    CORRELATION_INTEGRITY_FAILURE = "correlation_integrity_failure"
    CORRELATION_PERSISTENCE_UNAVAILABLE = "correlation_persistence_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_BINDING_MISMATCH = "provider_binding_mismatch"
    CONTINUATION_CONTRACT_MISSING = "continuation_contract_missing"
    CONTINUATION_OUTCOME_CONTRACT_MISMATCH = "continuation_outcome_contract_mismatch"
    PROVIDER_OPERATION_NOT_FOUND = "provider_operation_not_found"
    PROVIDER_FAILURE = "provider_failure"
    TRANSPORT_FAILURE = "transport_failure"
    CONTINUATION_OUTCOME_UNKNOWN = "continuation_outcome_unknown"


class DelegatedExecutionContinuationRequest(BaseModel):
    """Reattachment carrier built from durable invocation binding (service-owned)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_continuation_request.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_CONTINUATION_REQUEST_V1
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


class DelegatedExecutionContinuationView(BaseModel):
    """Typed platform read model after validated provider reattachment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_continuation_view.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_CONTINUATION_VIEW_V1
    )
    execution_id: ExecutionId
    run_id: RunId
    provider_id: str = _NON_EMPTY
    invocation_id: str = _NON_EMPTY
    provider_request_id: str | None = None
    provider_operation_id: str | None = None
    reattachment_kind: DelegatedExecutionReattachmentKind
    physical_status: DelegatedExecutionProviderPhysicalStatus | None = None
    provider_external_status: str | None = None
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class DelegatedExecutionProviderReattachmentObservation:
    """Untrusted plugin reattachment payload before platform validation."""

    kind: DelegatedExecutionReattachmentKind
    provider_id: str
    invocation_id: str
    provider_request_id: str | None = None
    provider_operation_id: str | None = None
    physical_status: DelegatedExecutionProviderPhysicalStatus | None = None
    provider_external_status: str | None = None


class DelegatedExecutionContinuationOutcomeUnknownError(DelegatedExecutionProviderError):
    """Continuation completed without sufficient evidence to classify outcome."""


@dataclass(frozen=True, slots=True)
class DelegatedExecutionContinuationOutcome:
    """Neutral reattachment result for runtime consumers."""

    category: DelegatedExecutionContinuationOutcomeCategory
    execution_id: ExecutionId | None = None
    run_id: RunId | None = None
    provider_id: str | None = None
    view: DelegatedExecutionContinuationView | None = None
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        success_categories = {
            DelegatedExecutionContinuationOutcomeCategory.REATTACHED,
            DelegatedExecutionContinuationOutcomeCategory.ALREADY_ATTACHED,
            DelegatedExecutionContinuationOutcomeCategory.OBSERVED_TERMINAL,
        }
        if self.category in success_categories:
            if self.view is None:
                raise DelegatedExecutionContractError(
                    "successful continuation outcome requires view",
                )
            if self.failure_code is not None or self.failure_message is not None:
                raise DelegatedExecutionContractError(
                    "successful continuation outcome cannot carry failure fields",
                )
            return
        if self.view is not None:
            raise DelegatedExecutionContractError(
                "failure continuation outcome cannot carry view",
            )
        if self.category in {
            DelegatedExecutionContinuationOutcomeCategory.UNSUPPORTED,
            DelegatedExecutionContinuationOutcomeCategory.CORRELATION_NOT_FOUND,
            DelegatedExecutionContinuationOutcomeCategory.CORRELATION_INTEGRITY_FAILURE,
            DelegatedExecutionContinuationOutcomeCategory.CORRELATION_PERSISTENCE_UNAVAILABLE,
            DelegatedExecutionContinuationOutcomeCategory.PROVIDER_UNAVAILABLE,
            DelegatedExecutionContinuationOutcomeCategory.PROVIDER_BINDING_MISMATCH,
            DelegatedExecutionContinuationOutcomeCategory.CONTINUATION_CONTRACT_MISSING,
            DelegatedExecutionContinuationOutcomeCategory.CONTINUATION_OUTCOME_CONTRACT_MISMATCH,
            DelegatedExecutionContinuationOutcomeCategory.PROVIDER_OPERATION_NOT_FOUND,
            DelegatedExecutionContinuationOutcomeCategory.PROVIDER_FAILURE,
            DelegatedExecutionContinuationOutcomeCategory.TRANSPORT_FAILURE,
            DelegatedExecutionContinuationOutcomeCategory.CONTINUATION_OUTCOME_UNKNOWN,
        } and not (self.failure_code and self.failure_code.strip()):
            raise DelegatedExecutionContractError(
                "continuation failure categories require failure_code",
            )


def provider_reattachment_observation_matches_binding(
    *,
    observation: DelegatedExecutionProviderReattachmentObservation,
    request: DelegatedExecutionContinuationRequest,
    bound_provider_id: str,
) -> bool:
    """Reject untrusted plugin observations that spoof platform correlation fields."""
    inv = request.provider_invocation
    return (
        observation.provider_id == bound_provider_id
        and observation.invocation_id == inv.invocation_id
        and observation.provider_request_id == inv.provider_request_id
        and observation.provider_operation_id == inv.provider_operation_id
    )


def continuation_outcome_category_for_kind(
    kind: DelegatedExecutionReattachmentKind,
) -> DelegatedExecutionContinuationOutcomeCategory:
    """Map validated provider kind to platform outcome category."""
    if kind is DelegatedExecutionReattachmentKind.REATTACHED:
        return DelegatedExecutionContinuationOutcomeCategory.REATTACHED
    if kind is DelegatedExecutionReattachmentKind.ALREADY_ATTACHED:
        return DelegatedExecutionContinuationOutcomeCategory.ALREADY_ATTACHED
    if kind is DelegatedExecutionReattachmentKind.OBSERVED_TERMINAL:
        return DelegatedExecutionContinuationOutcomeCategory.OBSERVED_TERMINAL
    if kind is DelegatedExecutionReattachmentKind.OPERATION_NOT_FOUND:
        return DelegatedExecutionContinuationOutcomeCategory.PROVIDER_OPERATION_NOT_FOUND
    raise DelegatedExecutionContractError(
        f"unsupported reattachment kind: {kind}",
    )


@runtime_checkable
class DelegatedExecutionReattachmentProvider(Protocol):
    """Optional reattachment surface — required when ``supports_reattachment`` is True."""

    async def reattach_delegated_execution(
        self,
        request: DelegatedExecutionContinuationRequest,
    ) -> DelegatedExecutionProviderReattachmentObservation:
        """Reconnect to the existing provider-side operation for the bound invocation."""
        ...


__all__ = [
    "DelegatedExecutionContinuationOutcome",
    "DelegatedExecutionContinuationOutcomeCategory",
    "DelegatedExecutionContinuationOutcomeUnknownError",
    "DelegatedExecutionContinuationRequest",
    "DelegatedExecutionContinuationView",
    "DelegatedExecutionProviderReattachmentObservation",
    "DelegatedExecutionReattachmentKind",
    "DelegatedExecutionReattachmentProvider",
    "continuation_outcome_category_for_kind",
    "provider_reattachment_observation_matches_binding",
]
