# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy control engine port (SELF-HEALING R6.1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.decision import AutonomyControlDecision
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest


@runtime_checkable
class AutonomyControlEngine(Protocol):
    @property
    def engine_id(self) -> str: ...

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyControlDecision:
        """
        Classify whether a recommendation may proceed under enterprise autonomy rules.

        Must not execute strategies, mutate lifecycle, or bypass decision authority.
        """
        ...


__all__ = ["AutonomyControlEngine"]
