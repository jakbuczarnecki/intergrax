# © Artur Czarnecki. All rights reserved.

"""Host-owned orchestration DecisionRequirementPolicy for legal harness / production MSE (GR-10-R10-R2)."""

from __future__ import annotations

from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)

from legal_application.host.settings import LegalBackendSettings


def default_legal_harness_orchestration_decision_requirement_policy() -> DecisionRequirementPolicy:
    """Explicit host strategy — orchestration MSE uses permissive classification until legal rules are configured."""
    return PermissiveDecisionRequirementPolicy()


def resolve_legal_harness_orchestration_decision_requirement_policy(
    settings: LegalBackendSettings,
) -> DecisionRequirementPolicy:
    override = settings.orchestration_decision_requirement_policy
    if override is not None:
        return override
    return default_legal_harness_orchestration_decision_requirement_policy()


__all__ = [
    "default_legal_harness_orchestration_decision_requirement_policy",
    "resolve_legal_harness_orchestration_decision_requirement_policy",
]
