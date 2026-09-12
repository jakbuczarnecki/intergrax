# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain-neutral governance disposition before execution lifecycle (ERL)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.enterprise_reliability.hitl_requirement import HumanApprovalRequirement


class GovernanceDisposition(StrEnum):
    """Whether a proposed lifecycle action may run automatically."""

    ALLOW = "allow"
    DENY = "deny"
    APPROVAL_REQUIRED = "approval_required"


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    """Typed governance outcome — execution runtime acts only after this boundary."""

    disposition: GovernanceDisposition
    rationale: str = ""
    hitl_requirement: HumanApprovalRequirement | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not GovernanceDisposition:
            raise TypeError("GovernanceDecision.disposition must be GovernanceDisposition")
        if self.disposition is GovernanceDisposition.APPROVAL_REQUIRED:
            if type(self.hitl_requirement) is not HumanApprovalRequirement:
                raise TypeError(
                    "GovernanceDecision.hitl_requirement required when disposition is "
                    "APPROVAL_REQUIRED",
                )


def _platform_hitl_requirement(
    *,
    tenant_id: str,
    correlation_id: str,
    contract_id: str,
    requirement_ref: str,
    rationale: str,
) -> HumanApprovalRequirement:
    return HumanApprovalRequirement(
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        contract_id=contract_id,
        requirement_ref=requirement_ref,
        rationale=rationale,
    )


def missing_governance_strategy_decision(
    *,
    tenant_id: str,
    correlation_id: str,
    contract_id: str,
) -> GovernanceDecision:
    """Fail closed when no governance plugin is registered — never automatic allow."""
    rationale = "governance_strategy_missing"
    return GovernanceDecision(
        disposition=GovernanceDisposition.APPROVAL_REQUIRED,
        rationale=rationale,
        hitl_requirement=_platform_hitl_requirement(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            contract_id=contract_id,
            requirement_ref="erl:governance:strategy_missing",
            rationale=rationale,
        ),
    )


def abstained_governance_strategy_decision(
    *,
    tenant_id: str,
    correlation_id: str,
    contract_id: str,
) -> GovernanceDecision:
    """Strategy registered but returned no decision — fail closed without allow."""
    rationale = "governance_strategy_abstained"
    return GovernanceDecision(
        disposition=GovernanceDisposition.APPROVAL_REQUIRED,
        rationale=rationale,
        hitl_requirement=_platform_hitl_requirement(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            contract_id=contract_id,
            requirement_ref="erl:governance:strategy_abstained",
            rationale=rationale,
        ),
    )


def invalid_governance_strategy_decision(
    *,
    tenant_id: str,
    correlation_id: str,
    contract_id: str,
) -> GovernanceDecision:
    """Unknown or malformed strategy outcome — deny automatic execution."""
    rationale = "governance_strategy_invalid"
    return GovernanceDecision(
        disposition=GovernanceDisposition.DENY,
        rationale=rationale,
        hitl_requirement=None,
    )


__all__ = [
    "GovernanceDecision",
    "GovernanceDisposition",
    "abstained_governance_strategy_decision",
    "invalid_governance_strategy_decision",
    "missing_governance_strategy_decision",
]
