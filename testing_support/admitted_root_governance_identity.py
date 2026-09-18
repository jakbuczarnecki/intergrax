# © Artur Czarnecki. All rights reserved.

"""Explicit admitted identity fixtures for qualification harnesses (not production)."""

from __future__ import annotations

from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.runtime.task.task import Task


def lab_admitted_root_governance_identity(
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
) -> AdmittedRootGovernanceIdentity:
    return AdmittedRootGovernanceIdentity(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
    )


def lab_admitted_root_governance_identity_for_task(
    task: Task,
) -> AdmittedRootGovernanceIdentity:
    """Deterministic harness mapping — requires explicit tenant_id and user_id on the task."""
    tenant = task.tenant_id.strip()
    principal = (task.user_id or "").strip()
    if not tenant or not principal:
        raise ValueError("lab admitted identity requires task tenant_id and user_id")
    workspace_raw = task.metadata.get("workspace_id")
    workspace = workspace_raw.strip() if isinstance(workspace_raw, str) else ""
    if not workspace:
        workspace = f"lab-workspace:{tenant}"
    return AdmittedRootGovernanceIdentity(
        tenant_id=tenant,
        workspace_id=workspace,
        principal_id=principal,
    )


__all__ = [
    "lab_admitted_root_governance_identity",
    "lab_admitted_root_governance_identity_for_task",
]
