# © Artur Czarnecki. All rights reserved.

"""Host-owned orchestration DecisionRequirementPolicy for harness / production composition (GR-10-R10-R1)."""

from __future__ import annotations

from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)


def default_governed_contractor_harness_orchestration_decision_requirement_policy() -> (
    DecisionRequirementPolicy
):
    """Explicit host strategy — generic orchestration runtime does not invent domain decision rules."""
    return PermissiveDecisionRequirementPolicy()


__all__ = [
    "default_governed_contractor_harness_orchestration_decision_requirement_policy",
]
