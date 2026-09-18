# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Consistency checks between active governance identity and non-authoritative projections."""

from __future__ import annotations

from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
)


class GovernanceIdentityProjectionMismatchError(RuntimeError):
    """Supplied projection fields disagree with active execution governance identity."""


def validate_governance_identity_projection(
    identity: ActiveExecutionGovernanceIdentity,
    *,
    tenant_id: str | None = None,
    workspace_id: str | None = None,
    principal_id: str | None = None,
) -> None:
    """Fail closed when a projection field is present and differs from the active carrier."""
    if tenant_id is not None:
        projected = tenant_id.strip()
        if projected and projected != identity.tenant_id:
            raise GovernanceIdentityProjectionMismatchError(
                "governance tenant_id projection mismatch",
            )
    if workspace_id is not None:
        projected = workspace_id.strip()
        if projected and projected != identity.workspace_id:
            raise GovernanceIdentityProjectionMismatchError(
                "governance workspace_id projection mismatch",
            )
    if principal_id is not None:
        projected = principal_id.strip()
        if projected and projected != identity.principal_id:
            raise GovernanceIdentityProjectionMismatchError(
                "governance principal_id projection mismatch",
            )


__all__ = [
    "GovernanceIdentityProjectionMismatchError",
    "validate_governance_identity_projection",
]
