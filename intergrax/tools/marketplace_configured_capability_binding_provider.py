# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Configured Marketplace Tool capability binding — no activation or business I/O."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.capability_qualification.configured_capability_execution_subject import (
    ConfiguredCapabilityExecutionSubject,
    derive_configured_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingReasonCode,
    QualifiedCapabilityBindingResult,
    QualifiedCapabilityExecutionTarget,
)
from intergrax.tools.marketplace_tool_execution_routing import (
    MARKETPLACE_TOOL_CONFIGURED_CAPABILITY_BINDING_PROVIDER_ID,
    build_marketplace_tool_execution_target,
    derive_marketplace_configured_tool_execution_target_reference,
)


class MarketplaceConfiguredCapabilityBindingRequest:
    """Thin binding envelope for configured execution subject."""

    def __init__(
        self,
        *,
        binding_operation_id: str,
        subject: ConfiguredCapabilityExecutionSubject,
        requested_at: datetime | None = None,
    ) -> None:
        self.binding_operation_id = binding_operation_id
        self.subject = subject
        self.requested_at = requested_at or datetime.now(UTC)


class MarketplaceConfiguredCapabilityBindingProvider:
    """Produce opaque configured execution targets — truthful configured binding provenance."""

    @property
    def provider_id(self) -> str:
        return MARKETPLACE_TOOL_CONFIGURED_CAPABILITY_BINDING_PROVIDER_ID

    def bind(
        self,
        request: MarketplaceConfiguredCapabilityBindingRequest,
    ) -> QualifiedCapabilityBindingResult:
        subject = request.subject
        target_reference = derive_marketplace_configured_tool_execution_target_reference(
            request.binding_operation_id,
        )
        target = build_marketplace_tool_execution_target(
            execution_target_reference=target_reference,
            binding_provider_id=self.provider_id,
            qualified_subject_reference=subject.subject_reference,
        )
        timestamp = request.requested_at
        return QualifiedCapabilityBindingResult(
            binding_operation_id=request.binding_operation_id,
            outcome=QualifiedCapabilityBindingOutcome.BOUND,
            reason_code=QualifiedCapabilityBindingReasonCode.NONE,
            provider_id=self.provider_id,
            execution_target=target,
            started_at=timestamp,
            completed_at=timestamp,
        )


def derive_marketplace_configured_binding_operation_id(
    *,
    subject: ConfiguredCapabilityExecutionSubject,
) -> str:
    from intergrax.contracts.capability_qualification.configured_capability_execution_subject import (
        derive_configured_capability_execution_operation_id,
    )

    execution_operation_id = derive_configured_capability_execution_operation_id(
        recovery_decision_id=subject.recovery_decision_id,
        decision_id=subject.decision_id,
    )
    return derive_configured_capability_binding_operation_id(
        configured_execution_operation_id=execution_operation_id,
        subject_reference=subject.subject_reference,
    )


__all__ = [
    "MarketplaceConfiguredCapabilityBindingProvider",
    "MarketplaceConfiguredCapabilityBindingRequest",
    "derive_marketplace_configured_binding_operation_id",
]
