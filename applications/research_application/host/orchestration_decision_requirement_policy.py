# © Artur Czarnecki. All rights reserved.

"""Host-owned orchestration DecisionRequirementPolicy for research harness / production MSE (GR-10-R10-R2)."""

from __future__ import annotations

from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)

def default_research_harness_orchestration_decision_requirement_policy() -> DecisionRequirementPolicy:
    """Explicit host strategy — no domain decision-bound actions classified today."""
    return PermissiveDecisionRequirementPolicy()


def resolve_research_harness_orchestration_decision_requirement_policy(
    runtime_override: DecisionRequirementPolicy | None = None,
) -> DecisionRequirementPolicy:
    if runtime_override is not None:
        return runtime_override
    return default_research_harness_orchestration_decision_requirement_policy()


__all__ = [
    "default_research_harness_orchestration_decision_requirement_policy",
    "resolve_research_harness_orchestration_decision_requirement_policy",
]
