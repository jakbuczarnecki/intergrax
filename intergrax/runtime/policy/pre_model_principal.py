# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PRE_MODEL governance principal resolution (GR-10-R2-C1 / GR-10-R2-R1)."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    peek_active_execution_governance_identity,
    require_active_execution_governance_identity,
)
from intergrax.runtime.governance.governance_identity_projection import (
    GovernanceIdentityProjectionMismatchError,
    validate_governance_identity_projection,
)
from intergrax.runtime.policy.pre_model_policy_errors import PreModelPolicyConfigurationError
from intergrax.runtime.task.task import Task


def principal_id_from_request_identity(identity: RequestIdentity) -> str:
    """Map admitted run identity to a non-empty governance principal when possible."""
    auth_subject = (identity.auth_subject or "").strip()
    if auth_subject:
        return auth_subject
    user_id = (identity.user_id or "").strip()
    return user_id


def _projection_mismatch_error(exc: GovernanceIdentityProjectionMismatchError) -> PreModelPolicyConfigurationError:
    return PreModelPolicyConfigurationError(str(exc))


def require_orchestration_pre_model_governance_scope(task: Task) -> ActiveExecutionGovernanceIdentity:
    """Atomic tenant/workspace/principal for orchestration PRE_MODEL from active governance identity."""
    try:
        identity = require_active_execution_governance_identity()
    except RuntimeError as exc:
        raise PreModelPolicyConfigurationError(
            "pre_model governance identity unavailable",
        ) from exc
    workspace_projection = task.metadata.get("workspace_id")
    workspace_value = workspace_projection if isinstance(workspace_projection, str) else None
    try:
        validate_governance_identity_projection(
            identity,
            tenant_id=task.tenant_id,
            workspace_id=workspace_value,
            principal_id=task.user_id,
        )
        if task.canonical_identity is not None:
            validate_governance_identity_projection(
                identity,
                principal_id=principal_id_from_request_identity(task.canonical_identity),
            )
    except GovernanceIdentityProjectionMismatchError as exc:
        raise _projection_mismatch_error(exc) from exc
    return identity


def principal_id_for_orchestration_task(task: Task) -> str:
    """Resolve PRE_MODEL principal from active governance identity with task projection checks."""
    return require_orchestration_pre_model_governance_scope(task).principal_id


def resolve_agentic_pre_model_scope(
    *,
    tenant_id: str,
    workspace_id: str | None,
    request_principal_id: str,
    production_mode: bool,
) -> ActiveExecutionGovernanceIdentity:
    """Resolve agentic PRE_MODEL scope from active governance identity or harness projection."""
    active = peek_active_execution_governance_identity()
    if active is not None:
        try:
            validate_governance_identity_projection(
                active,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                principal_id=request_principal_id,
            )
        except GovernanceIdentityProjectionMismatchError as exc:
            raise _projection_mismatch_error(exc) from exc
        return active
    raise PreModelPolicyConfigurationError(
        "pre_model governance identity unavailable",
    )


__all__ = [
    "principal_id_for_orchestration_task",
    "principal_id_from_request_identity",
    "require_orchestration_pre_model_governance_scope",
    "resolve_agentic_pre_model_scope",
]
