# © Artur Czarnecki. All rights reserved.

"""G-09 gate fixtures — tenant-scoped online evaluation registry handles."""

from __future__ import annotations

from intergrax.runtime.architecture.online_evaluation_registry import (
    InMemoryOnlineEvaluationRegistry,
    TenantScopedOnlineEvaluationRegistry,
)

__all__ = [
    "CI_SMOKE_PRODUCT_TENANT",
    "tenant_scoped_online_evaluation_registry",
]

# Shared with DF5 destructive proof and harness architecture gates.
CI_SMOKE_PRODUCT_TENANT = "df5-destructive"


def tenant_scoped_online_evaluation_registry(
    tenant_id: str = CI_SMOKE_PRODUCT_TENANT,
) -> TenantScopedOnlineEvaluationRegistry:
    """In-memory registry view bound to one tenant (G-09 fail-closed contract)."""
    return TenantScopedOnlineEvaluationRegistry(
        InMemoryOnlineEvaluationRegistry(),
        tenant_id,
    )
