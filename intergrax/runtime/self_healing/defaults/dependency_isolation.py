# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Platform default — external dependency isolation (SELF-HEALING R1)."""

from __future__ import annotations

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.decision import SelfHealingDecision
from intergrax.contracts.self_healing.strategy import SelfHealingStrategyDescriptor
from intergrax.runtime.self_healing.defaults._helpers import mint_default_decision

_ACTION = "self_healing.dependency.isolation"
_OPERATION = "dependency.circuit.open"
_CAPABILITY = "dependency.degradation"


class ExternalDependencyIsolationStrategy:
    strategy_id = "platform.default.dependency_isolation"
    version = "1"

    @property
    def descriptor(self) -> SelfHealingStrategyDescriptor:
        return SelfHealingStrategyDescriptor(
            strategy_id=self.strategy_id,
            version=self.version,
            owner="platform",
            capabilities=(_CAPABILITY,),
            tenant_scope=None,
            priority=30,
            specificity=1,
            timeout_seconds=1.0,
            resource_budget_tokens=128,
        )

    def evaluate(self, context: SelfHealingContext) -> SelfHealingDecision | None:
        return mint_default_decision(
            strategy_id=self.strategy_id,
            context=context,
            action_type=_ACTION,
            operation_kind=_OPERATION,
            confidence=0.65,
            justification="isolate failing external dependency to protect tenant workload",
            required_approval=True,
        )


__all__ = ["ExternalDependencyIsolationStrategy"]
