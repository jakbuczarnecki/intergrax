# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integrations-owned scoped adaptation target resolution (AW-7C-P3/P4)."""

from __future__ import annotations

from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationError,
    ScopedIntegrationAdaptationFailureReason,
    ScopedIntegrationAdaptationRequest,
    ScopedIntegrationAdaptationTarget,
    ScopedIntegrationAdaptationTargetLookupKey,
    ScopedIntegrationAdaptationTargetSource,
)


class SourceBackedScopedIntegrationAdaptationTargetResolver:
    """Resolve target via Integrations-owned source — never echo request as truth."""

    def __init__(
        self,
        *,
        target_source: ScopedIntegrationAdaptationTargetSource,
    ) -> None:
        self._target_source = target_source

    def resolve_target(
        self,
        request: ScopedIntegrationAdaptationRequest,
    ) -> ScopedIntegrationAdaptationTarget:
        key = ScopedIntegrationAdaptationTargetLookupKey(
            tenant_id=request.tenant_id,
            integration_category=request.integration_category,
            provider_id=request.provider_id,
            resource_scope=request.resource_scope,
        )
        try:
            return self._target_source.resolve(key)
        except ScopedIntegrationAdaptationError:
            raise
        except KeyError as exc:
            raise ScopedIntegrationAdaptationError(
                ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
                detail=f"target source miss: {exc}",
            ) from exc
        except LookupError as exc:
            raise ScopedIntegrationAdaptationError(
                ScopedIntegrationAdaptationFailureReason.IDENTITY_MISMATCH,
                detail=str(exc),
            ) from exc


__all__ = [
    "SourceBackedScopedIntegrationAdaptationTargetResolver",
]
