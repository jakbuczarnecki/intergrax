# © Artur Czarnecki. All rights reserved.

"""Tool effective availability → capability health facts (P1.5)."""

from __future__ import annotations

from intergrax.applications.contracts.capability_dependency import CapabilityDependencyKind
from intergrax.applications.contracts.capability_health import (
    CapabilityHealthConditionKind,
    CapabilityHealthFact,
    CapabilityHealthFactStatus,
    CapabilityHealthProjectionContext,
    CapabilityHealthProvider,
    CapabilityHealthReason,
)
from intergrax.skills.registry.tool_requirements import available_tool_ids_for_profile

_PROVIDER_ID = "tool_effective_availability"
_SOURCE_PROVENANCE = "tool_profile"


class ToolEffectiveAvailabilityHealthProvider:
    """Expose host ToolProfile availability as one health fact for tool capabilities."""

    @property
    def provider_id(self) -> str:
        return _PROVIDER_ID

    @property
    def source_provenance(self) -> str:
        return _SOURCE_PROVENANCE

    def health_facts_for(
        self,
        context: CapabilityHealthProjectionContext,
    ) -> tuple[CapabilityHealthFact, ...]:
        if context.capability.kind is not CapabilityDependencyKind.TOOL:
            return ()
        if context.environment_profile is None:
            return (
                CapabilityHealthFact(
                    capability=context.capability,
                    source=self.source_provenance,
                    condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
                    condition_ref=context.capability.capability_id,
                    scope_application_id=context.scope_application_id,
                    scope_tenant_id=context.scope_tenant_id,
                    status=CapabilityHealthFactStatus.UNKNOWN,
                    blocking=True,
                    reason=CapabilityHealthReason(
                        reason_code="tool.availability.unknown",
                        source=self.source_provenance,
                        subject_ref=context.capability.canonical_key,
                        detail="environment profile is not available for tool availability projection",
                    ),
                    provider_id=self.provider_id,
                ),
            )

        available = available_tool_ids_for_profile(
            context.environment_profile.tool_profile,
        )
        tool_id = context.capability.capability_id
        if tool_id in available:
            return (
                CapabilityHealthFact(
                    capability=context.capability,
                    source=self.source_provenance,
                    condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
                    condition_ref=tool_id,
                    scope_application_id=context.scope_application_id,
                    scope_tenant_id=context.scope_tenant_id,
                    status=CapabilityHealthFactStatus.SATISFIED,
                    blocking=True,
                    reason=CapabilityHealthReason(
                        reason_code="tool.availability.satisfied",
                        source=self.source_provenance,
                        subject_ref=context.capability.canonical_key,
                        detail="tool is effectively available on host ToolProfile",
                    ),
                    provider_id=self.provider_id,
                ),
            )
        return (
            CapabilityHealthFact(
                capability=context.capability,
                source=self.source_provenance,
                condition_kind=CapabilityHealthConditionKind.TOOL_EFFECTIVE_AVAILABILITY,
                condition_ref=tool_id,
                scope_application_id=context.scope_application_id,
                scope_tenant_id=context.scope_tenant_id,
                status=CapabilityHealthFactStatus.UNSATISFIED,
                blocking=True,
                reason=CapabilityHealthReason(
                    reason_code="tool.availability.unsatisfied",
                    source=self.source_provenance,
                    subject_ref=context.capability.canonical_key,
                    detail=f"tool {tool_id!r} is not effectively available on host ToolProfile",
                ),
                provider_id=self.provider_id,
            ),
        )


def tool_effective_availability_health_provider() -> CapabilityHealthProvider:
    return ToolEffectiveAvailabilityHealthProvider()
