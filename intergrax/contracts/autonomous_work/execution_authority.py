# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker execution authority admission contracts (AW-3B).

Admission correlates Worker identity binding with Collaborative Work effective
authority for canonical Execution intake. Snapshot semantics apply per admission
only — not a durable Worker permission cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.autonomous_work._validation import (
    freeze_tuple,
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.ids import (
    WorkerInstanceId,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.principal_binding import ResolvedWorkerPrincipal
from intergrax.contracts.collaborative_work import (
    EffectiveAuthorityDecision,
    EffectiveAuthorityRequest,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority


def validate_authority_scopes(value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate a non-empty tuple of non-empty authority scope identifiers."""
    scopes = freeze_tuple(value, label="requested_authority_scopes")
    if not scopes:
        raise ValueError("requested_authority_scopes must contain at least one scope")
    normalized = tuple(require_non_empty_text(scope, label="authority_scope") for scope in scopes)
    return normalized


@dataclass(frozen=True, slots=True)
class WorkerExecutionAuthorityRequest:
    """Authority required for a prospective Worker execution admission.

    Identity coordinates come exclusively from AW-3A binding resolution.
    """

    worker_instance_id: WorkerInstanceId
    requested_authority_scopes: tuple[str, ...]
    resource_scope: str | None = None
    delegator_principal_id: str | None = None
    delegation_id: str | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "requested_authority_scopes",
            validate_authority_scopes(self.requested_authority_scopes),
        )
        if self.resource_scope is not None:
            object.__setattr__(
                self,
                "resource_scope",
                require_non_empty_text(self.resource_scope, label="resource_scope"),
            )
        if self.delegator_principal_id is not None:
            object.__setattr__(
                self,
                "delegator_principal_id",
                require_non_empty_text(
                    self.delegator_principal_id,
                    label="delegator_principal_id",
                ),
            )
        if self.delegation_id is not None:
            object.__setattr__(
                self,
                "delegation_id",
                require_non_empty_text(self.delegation_id, label="delegation_id"),
            )
        if self.delegator_principal_id is not None and self.delegation_id is None:
            raise ValueError("delegation_id is required when delegator_principal_id is provided")
        if self.delegation_id is not None and self.delegator_principal_id is None:
            raise ValueError("delegator_principal_id is required when delegation_id is provided")


@dataclass(frozen=True, slots=True)
class WorkerExecutionAuthorityContext:
    """Immutable admission authority snapshot for one Worker execution intake."""

    worker_instance_id: WorkerInstanceId
    resolved_principal: ResolvedWorkerPrincipal
    requested_authority_scopes: tuple[str, ...]
    approved_authority_scopes: tuple[str, ...]
    effective_authority_request: EffectiveAuthorityRequest
    effective_authority_decision: EffectiveAuthorityDecision
    evaluated_at: datetime

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "requested_authority_scopes",
            validate_authority_scopes(self.requested_authority_scopes),
        )
        approved = validate_authority_scopes(self.approved_authority_scopes)
        object.__setattr__(self, "approved_authority_scopes", approved)
        if not set(approved).issubset(set(self.requested_authority_scopes)):
            raise ValueError("approved_authority_scopes must be a subset of requested scopes")
        if type(self.effective_authority_request) is not EffectiveAuthorityRequest:
            raise TypeError("effective_authority_request must be EffectiveAuthorityRequest")
        if type(self.effective_authority_decision) is not EffectiveAuthorityDecision:
            raise TypeError("effective_authority_decision must be EffectiveAuthorityDecision")
        object.__setattr__(
            self,
            "evaluated_at",
            require_aware_utc(self.evaluated_at, label="evaluated_at"),
        )

    def to_parent_execution_authority(self) -> ParentExecutionAuthority:
        """Project approved scopes to canonical runtime execution authority carrier."""
        return ParentExecutionAuthority.scoped(self.approved_authority_scopes)
