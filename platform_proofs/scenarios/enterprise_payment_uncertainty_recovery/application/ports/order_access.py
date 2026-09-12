"""Order read port — application never imports storage technology."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.entities import (
    OrderSnapshot,
)


class OrderAccessPort(Protocol):
    def load_order_for_execution(self, context: ScenarioExecutionContext) -> OrderSnapshot:
        """Return the order that anchors this scenario run."""
        ...
