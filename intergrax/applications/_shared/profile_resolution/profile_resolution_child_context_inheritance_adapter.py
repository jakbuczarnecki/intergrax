# © Artur Czarnecki. All rights reserved.

"""Profile Resolution adapter for neutral child execution context inheritance."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.profile_resolution.execution_pinning import (
    inherit_child_execution_pinned_revision,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionPinningStore,
)
from intergrax.contracts.child_execution_context_inheritance import (
    ChildExecutionContextInheritancePort,
    ChildExecutionContextInheritanceRequest,
)


@dataclass(frozen=True, slots=True)
class ProfileResolutionChildContextInheritanceAdapter(ChildExecutionContextInheritancePort):
    """Tenant-bound profile pinning inheritance for canonical child executions."""

    tenant_id: str
    pinning_store: EffectiveProfileExecutionPinningStore

    def inherit_child_context(self, request: ChildExecutionContextInheritanceRequest) -> None:
        tenant = self.tenant_id.strip()
        if not tenant:
            raise ValueError("tenant_id must be non-empty")
        inherit_child_execution_pinned_revision(
            tenant_id=tenant,
            parent_execution_id=request.parent_execution_id,
            child_execution_id=request.child_execution_id,
            pinning_store=self.pinning_store,
        )


def build_profile_resolution_child_context_inheritance_adapter(
    *,
    tenant_id: str,
    pinning_store: EffectiveProfileExecutionPinningStore,
) -> ProfileResolutionChildContextInheritanceAdapter:
    """Single composition helper for profile-aware host child context propagation."""
    return ProfileResolutionChildContextInheritanceAdapter(
        tenant_id=tenant_id,
        pinning_store=pinning_store,
    )


__all__ = [
    "ProfileResolutionChildContextInheritanceAdapter",
    "build_profile_resolution_child_context_inheritance_adapter",
]
