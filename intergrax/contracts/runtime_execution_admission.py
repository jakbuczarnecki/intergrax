# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime/Governance root execution authority admission contracts (AW-5A seam).

Trusted ``ParentExecutionAuthority`` minting belongs to Runtime/Governance.
Autonomous Work consumes this port after AW-3B collaborative admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.execution_authority import (
    validate_authority_scopes,
)
from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.runtime_policy import PolicyDecision


class RootExecutionAuthorityAdmissionDisposition(StrEnum):
    """Runtime/Governance admission outcome for trusted root authority."""

    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    REQUIRE_HUMAN = "REQUIRE_HUMAN"
    ESCALATE = "ESCALATE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class RootExecutionAuthorityAdmissionRequest:
    """Inputs for trusted root execution authority admission."""

    tenant_id: str
    workspace_id: str
    principal_id: str
    collaborative_authority_scopes: tuple[str, ...]
    effective_authority_decision: EffectiveAuthorityDecision

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not self.workspace_id.strip():
            raise ValueError("workspace_id must be non-empty")
        if not self.principal_id.strip():
            raise ValueError("principal_id must be non-empty")
        object.__setattr__(
            self,
            "collaborative_authority_scopes",
            validate_authority_scopes(self.collaborative_authority_scopes),
        )
        if type(self.effective_authority_decision) is not EffectiveAuthorityDecision:
            raise TypeError("effective_authority_decision must be EffectiveAuthorityDecision")


@dataclass(frozen=True, slots=True)
class RootExecutionAuthorityAdmissionResult:
    """Trusted root authority admission result — fail closed when not ALLOWED."""

    disposition: RootExecutionAuthorityAdmissionDisposition
    trusted_parent_execution_authority: ParentExecutionAuthority | None = None
    policy_decision: PolicyDecision | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not RootExecutionAuthorityAdmissionDisposition:
            raise TypeError("disposition must be RootExecutionAuthorityAdmissionDisposition")
        if self.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED:
            if self.trusted_parent_execution_authority is None:
                raise ValueError("ALLOWED admission requires trusted_parent_execution_authority")
            if type(self.trusted_parent_execution_authority) is not ParentExecutionAuthority:
                raise TypeError(
                    "trusted_parent_execution_authority must be ParentExecutionAuthority"
                )
        elif self.trusted_parent_execution_authority is not None:
            raise ValueError("non-ALLOWED admission must not expose trusted authority")


@runtime_checkable
class RootExecutionAuthorityAdmissionPort(Protocol):
    """Trusted root execution authority admission owned by Runtime/Governance."""

    def authorize(
        self,
        request: RootExecutionAuthorityAdmissionRequest,
    ) -> RootExecutionAuthorityAdmissionResult:
        """Evaluate runtime admission and mint trusted root authority when allowed."""
        ...
