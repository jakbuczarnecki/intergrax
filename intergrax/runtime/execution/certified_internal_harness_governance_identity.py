# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CERTIFIED INTERNAL harness admission — not production Host/Admission auth.

Used only by certified internal composition (nexus worker, debug, unit tests)
that call :func:`build_host_task_execution` without Tier-3 host wiring.
Production application hosts must inject real admission via applications wiring.
"""

from __future__ import annotations

from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.runtime.task.task import Task


def admit_certified_internal_harness_root_governance_identity(
    task: Task,
) -> AdmittedRootGovernanceIdentity:
    tenant = task.tenant_id.strip()
    if not tenant:
        raise ValueError("certified harness admitted identity requires task.tenant_id")
    principal = (task.user_id or "").strip()
    if not principal:
        raise ValueError("certified harness admitted identity requires task.user_id")
    workspace_raw = task.metadata.get("workspace_id")
    workspace = workspace_raw.strip() if isinstance(workspace_raw, str) else ""
    if not workspace:
        workspace = f"certified-harness-workspace:{tenant}"
    return AdmittedRootGovernanceIdentity(
        tenant_id=tenant,
        workspace_id=workspace,
        principal_id=principal,
    )


__all__ = ["admit_certified_internal_harness_root_governance_identity"]
