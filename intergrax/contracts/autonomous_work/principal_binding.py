# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker→Collaborative Principal durable binding contract (AW-3A).

The binding identifies which canonical Collaborative Principal a Worker acts as
within explicit tenant/workspace scope. It does not carry authority, permissions,
roles, or policy semantics. Binding mutation is a control-plane responsibility —
not Worker self-service.

Scoped identity is not authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.autonomous_work._validation import (
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.ids import (
    WorkerInstanceId,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.revision import (
    Revision,
    validate_revision,
)


def validate_collaborative_tenant_id(value: object) -> str:
    """Validate a canonical Collaborative Work tenant identifier reference."""
    return require_non_empty_text(value, label="tenant_id")


def validate_collaborative_workspace_id(value: object) -> str:
    """Validate a canonical Collaborative Work workspace identifier reference."""
    return require_non_empty_text(value, label="workspace_id")


def validate_collaborative_principal_id(value: object) -> str:
    """Validate a canonical Collaborative Work principal identifier reference."""
    return require_non_empty_text(value, label="principal_id")


@dataclass(frozen=True, slots=True)
class ResolvedWorkerPrincipal:
    """Provider-neutral scoped Principal identity resolved from a Worker binding.

    Identity coordinates only — no authority, permissions, roles, or policy fields.
    """

    tenant_id: str
    workspace_id: str
    principal_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", validate_collaborative_tenant_id(self.tenant_id))
        object.__setattr__(
            self,
            "workspace_id",
            validate_collaborative_workspace_id(self.workspace_id),
        )
        object.__setattr__(
            self,
            "principal_id",
            validate_collaborative_principal_id(self.principal_id),
        )


@dataclass(frozen=True, slots=True)
class WorkerPrincipalBinding:
    """Immutable durable Worker→Principal identity binding in tenant/workspace scope."""

    worker_instance_id: WorkerInstanceId
    tenant_id: str
    workspace_id: str
    principal_id: str
    created_at: datetime
    revision: Revision

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(self, "tenant_id", validate_collaborative_tenant_id(self.tenant_id))
        object.__setattr__(
            self,
            "workspace_id",
            validate_collaborative_workspace_id(self.workspace_id),
        )
        object.__setattr__(
            self,
            "principal_id",
            validate_collaborative_principal_id(self.principal_id),
        )
        object.__setattr__(
            self,
            "created_at",
            require_aware_utc(self.created_at, label="created_at"),
        )
        if type(self.revision) is not Revision:
            raise TypeError("revision must be Revision")
        validate_revision(self.revision)

    def to_resolved_principal(self) -> ResolvedWorkerPrincipal:
        """Project binding identity coordinates without authority semantics."""
        return ResolvedWorkerPrincipal(
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            principal_id=self.principal_id,
        )
