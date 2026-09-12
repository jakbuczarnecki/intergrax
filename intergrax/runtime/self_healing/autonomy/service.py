# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy control service — evaluation and optional persistence (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.decision import AutonomyControlDecision
from intergrax.contracts.self_healing.autonomy.engine import AutonomyControlEngine
from intergrax.contracts.self_healing.autonomy.repository import AutonomyRepository
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest


@dataclass(frozen=True, slots=True)
class AutonomyControlService:
    engine: AutonomyControlEngine
    repository: AutonomyRepository | None = None

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyControlDecision:
        decision = self.engine.evaluate(request)
        if self.repository is not None:
            return self.repository.append_decision(decision)
        return decision


__all__ = ["AutonomyControlService"]
