# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PRE_MODEL governance principal resolution (GR-10-R2-C1)."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.runtime.governance.active_execution_governance_identity import (
    peek_active_execution_governance_identity,
)
from intergrax.runtime.task.task import Task


def principal_id_from_request_identity(identity: RequestIdentity) -> str:
    """Map admitted run identity to a non-empty governance principal when possible."""
    auth_subject = (identity.auth_subject or "").strip()
    if auth_subject:
        return auth_subject
    user_id = (identity.user_id or "").strip()
    return user_id


def principal_id_for_orchestration_task(task: Task) -> str:
    """Resolve PRE_MODEL principal from active governance identity or task admission."""
    governance = peek_active_execution_governance_identity()
    if governance is not None:
        return governance.principal_id
    user_id = (task.user_id or "").strip()
    if user_id:
        return user_id
    if task.canonical_identity is not None:
        return principal_id_from_request_identity(task.canonical_identity)
    return ""


__all__ = [
    "principal_id_for_orchestration_task",
    "principal_id_from_request_identity",
]
