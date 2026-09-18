# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Active execution governance identity for PRE_MODEL evaluation and evidence (GR-10-R2)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActiveExecutionGovernanceIdentity:
    """Runtime-only tenant/workspace/principal carrier for governed model invocation."""

    tenant_id: str
    workspace_id: str
    principal_id: str


_active_execution_governance_identity: ContextVar[
    ActiveExecutionGovernanceIdentity | None
] = ContextVar(
    "active_execution_governance_identity",
    default=None,
)


def bind_active_execution_governance_identity(
    identity: ActiveExecutionGovernanceIdentity,
) -> Token:
    tenant_id = identity.tenant_id.strip()
    workspace_id = identity.workspace_id.strip()
    principal_id = identity.principal_id.strip()
    if not tenant_id or not workspace_id or not principal_id:
        raise ValueError(
            "active execution governance identity requires non-empty "
            "tenant_id, workspace_id, and principal_id",
        )
    return _active_execution_governance_identity.set(
        ActiveExecutionGovernanceIdentity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            principal_id=principal_id,
        ),
    )


def reset_active_execution_governance_identity(token: Token) -> None:
    _active_execution_governance_identity.reset(token)


def peek_active_execution_governance_identity() -> ActiveExecutionGovernanceIdentity | None:
    return _active_execution_governance_identity.get()


def require_active_execution_governance_identity() -> ActiveExecutionGovernanceIdentity:
    identity = peek_active_execution_governance_identity()
    if identity is None:
        raise RuntimeError("active execution governance identity required")
    return identity


__all__ = [
    "ActiveExecutionGovernanceIdentity",
    "bind_active_execution_governance_identity",
    "peek_active_execution_governance_identity",
    "require_active_execution_governance_identity",
    "reset_active_execution_governance_identity",
]
