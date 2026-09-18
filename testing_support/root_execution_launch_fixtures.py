# © Artur Czarnecki. All rights reserved.

"""Shared admitted identity fixtures for root launch tests."""

from __future__ import annotations

from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)


def sample_admitted_root_governance_identity(
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "workspace-x",
    principal_id: str = "principal-1",
) -> AdmittedRootGovernanceIdentity:
    return AdmittedRootGovernanceIdentity(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
    )


__all__ = ["sample_admitted_root_governance_identity"]
