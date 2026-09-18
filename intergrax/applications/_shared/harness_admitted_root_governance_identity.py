# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lab/harness composition — explicit admitted root governance identity (not production auth)."""

from __future__ import annotations

from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.runtime.task.task import Task


def admit_harness_root_governance_identity(
    task: Task,
) -> AdmittedRootGovernanceIdentity:
    """Map harness task fields to admitted identity after explicit lab composition rules."""
    tenant = task.tenant_id.strip()
    if not tenant:
        raise ValueError("harness admitted identity requires task.tenant_id")
    principal = (task.user_id or "").strip()
    if not principal:
        raise ValueError("harness admitted identity requires task.user_id")
    workspace_raw = task.metadata.get("workspace_id")
    workspace = workspace_raw.strip() if isinstance(workspace_raw, str) else ""
    if not workspace:
        workspace = f"harness-workspace:{tenant}"
    return AdmittedRootGovernanceIdentity(
        tenant_id=tenant,
        workspace_id=workspace,
        principal_id=principal,
    )


__all__ = ["admit_harness_root_governance_identity"]
