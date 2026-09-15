# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Tenant identity resolution for functional evidence (fail closed)."""

from __future__ import annotations

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext


class FunctionalEvidenceTenantIdentityError(ValueError):
    """Raised when tenant identity cannot be resolved for functional evidence."""


def require_tenant_id_from_exec_ctx(exec_ctx: RuntimeExecutionContext) -> str:
    """Resolve tenant from canonical request identity; never guess or default."""
    identity = exec_ctx.canonical_request_identity
    if identity is None:
        raise FunctionalEvidenceTenantIdentityError(
            "functional evidence requires canonical_request_identity with tenant_id",
        )
    tenant_id = identity.tenant_id
    if type(tenant_id) is not str:
        raise FunctionalEvidenceTenantIdentityError("tenant_id must be str")
    normalized = tenant_id.strip()
    if not normalized or tenant_id != normalized:
        raise FunctionalEvidenceTenantIdentityError("tenant_id must be non-empty and normalized")
    return normalized


__all__ = [
    "FunctionalEvidenceTenantIdentityError",
    "require_tenant_id_from_exec_ctx",
]
