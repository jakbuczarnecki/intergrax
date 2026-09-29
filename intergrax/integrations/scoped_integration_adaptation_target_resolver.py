# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integrations-owned scoped adaptation target resolution (AW-7C-P3)."""

from __future__ import annotations

from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationRequest,
    ScopedIntegrationAdaptationTarget,
)


class IntegrationIdentityScopedIntegrationAdaptationTargetResolver:
    """Resolve target truth from Integrations identity semantics (INT-CONFIG analogue)."""

    def resolve_target(
        self,
        request: ScopedIntegrationAdaptationRequest,
    ) -> ScopedIntegrationAdaptationTarget:
        scope = request.scope
        return ScopedIntegrationAdaptationTarget(
            tenant_id=request.tenant_id,
            integration_category=request.integration_category,
            provider_id=request.provider_id,
            resource_scope=request.resource_scope,
            current_revision=scope.candidate_revision,
        )


__all__ = [
    "IntegrationIdentityScopedIntegrationAdaptationTargetResolver",
]
