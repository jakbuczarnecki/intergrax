# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production TOOL host-available binding — resolves identity to EE execution target."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.autonomous_work.worker_host_available_capability_binding import (
    HostAvailableCapabilityBindingRequest,
    HostAvailableCapabilityBindingResult,
    host_available_subject_reference_for_identity,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingReasonCode,
    QualifiedCapabilityExecutionTarget,
)

HOST_AVAILABLE_TOOL_CAPABILITY_BINDING_PROVIDER_ID = (
    "host_available.tool_capability_binding.v1"
)


def execution_target_reference_for_host_tool(
    capability_identity_sort_key: tuple[str, str, str, str],
) -> str:
    kind, source_kind, source_id, logical_id = capability_identity_sort_key
    return f"host-available-tool:{kind}:{source_kind}:{source_id}:{logical_id}"


class HostAvailableToolCapabilityBindingProvider:
    """Bind host-available TOOL identities — no ToolRuntime invocation."""

    @property
    def provider_id(self) -> str:
        return HOST_AVAILABLE_TOOL_CAPABILITY_BINDING_PROVIDER_ID

    def supports(self, request: HostAvailableCapabilityBindingRequest) -> bool:
        return request.capability_identity.kind is CapabilityKind.TOOL

    def bind(
        self,
        request: HostAvailableCapabilityBindingRequest,
    ) -> HostAvailableCapabilityBindingResult:
        started_at = request.requested_at or datetime.now(tz=UTC)
        completed_at = datetime.now(tz=UTC)
        subject_ref = host_available_subject_reference_for_identity(
            request.capability_identity,
        )
        target = QualifiedCapabilityExecutionTarget(
            execution_target_reference=execution_target_reference_for_host_tool(
                request.capability_identity.sort_key,
            ),
            binding_provider_id=self.provider_id,
            qualified_subject_reference=subject_ref,
        )
        return HostAvailableCapabilityBindingResult(
            binding_operation_id=request.binding_operation_id,
            outcome=QualifiedCapabilityBindingOutcome.BOUND,
            reason_code=QualifiedCapabilityBindingReasonCode.NONE,
            provider_id=self.provider_id,
            execution_target=target,
            host_subject_reference=subject_ref,
            started_at=started_at,
            completed_at=completed_at,
        )


__all__ = [
    "HOST_AVAILABLE_TOOL_CAPABILITY_BINDING_PROVIDER_ID",
    "HostAvailableToolCapabilityBindingProvider",
    "execution_target_reference_for_host_tool",
]
