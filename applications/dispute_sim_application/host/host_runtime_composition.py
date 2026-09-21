# © Artur Czarnecki. All rights reserved.

"""EBH-2D-D-R3 — dispute_sim host runtime composition (not declarative settings)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_requirement_policy import DecisionRequirementPolicy


@dataclass(frozen=True, slots=True)
class DisputeSimHostRuntimeComposition:
    """Host-scoped runtime overrides for dispute_sim_application."""

    orchestration_decision_requirement_policy: DecisionRequirementPolicy | None = None


__all__ = ["DisputeSimHostRuntimeComposition"]
