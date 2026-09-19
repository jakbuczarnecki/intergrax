# © Artur Czarnecki. All rights reserved.

"""Host-owned orchestration DecisionRequirementPolicy for LKW harness / production MSE (GR-10-R10-R2)."""

from __future__ import annotations

from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)

from local_workspace_application.host.settings import LocalWorkspaceBackendSettings


def default_local_workspace_harness_orchestration_decision_requirement_policy() -> (
    DecisionRequirementPolicy
):
    """Explicit host strategy for strict product orchestration MSE."""
    return PermissiveDecisionRequirementPolicy()


def resolve_local_workspace_harness_orchestration_decision_requirement_policy(
    settings: LocalWorkspaceBackendSettings,
) -> DecisionRequirementPolicy:
    override = settings.orchestration_decision_requirement_policy
    if override is not None:
        return override
    return default_local_workspace_harness_orchestration_decision_requirement_policy()


__all__ = [
    "default_local_workspace_harness_orchestration_decision_requirement_policy",
    "resolve_local_workspace_harness_orchestration_decision_requirement_policy",
]
