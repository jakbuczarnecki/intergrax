"""Wired application dependencies — supplied by composition, not constructed in services."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.observability import (
    ScenarioApplicationObservability,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.ports.order_access import (
    OrderAccessPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.ports.payment_workflow import (
    PaymentWorkflowPort,
)


@dataclass(frozen=True, slots=True)
class ApplicationDependencies:
    order_access: OrderAccessPort
    payment_workflow: PaymentWorkflowPort
    observability: ScenarioApplicationObservability
