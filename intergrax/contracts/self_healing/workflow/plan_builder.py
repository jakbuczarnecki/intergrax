# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing plan builder SPI (SELF-HEALING R2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.decision import SelfHealingDecision
from intergrax.contracts.self_healing.workflow.plan import SelfHealingPlan


@runtime_checkable
class SelfHealingPlanBuilder(Protocol):
    """
    Builds declarative plans from strategy decisions.

    Must not execute or bypass governance.
    """

    @property
    def builder_id(self) -> str: ...

    def build_plan(
        self,
        decision: SelfHealingDecision,
        context: SelfHealingContext,
    ) -> SelfHealingPlan:
        ...


__all__ = ["SelfHealingPlanBuilder"]
