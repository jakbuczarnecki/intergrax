"""Application-level observability boundary — not ERL evidence packaging."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.workflow import (
    PaymentWorkflowOutcome,
)


class BusinessActionKind(StrEnum):
    ORDER_LOADED = "order_loaded"
    PAYMENT_CAPTURE_REQUESTED = "payment_capture_requested"


class ScenarioApplicationObservability(Protocol):
    def scenario_started(self, context: ScenarioExecutionContext) -> None:
        ...

    def business_action_executed(
        self,
        context: ScenarioExecutionContext,
        *,
        kind: BusinessActionKind,
        detail: dict[str, Any],
    ) -> None:
        ...

    def scenario_state_prepared(
        self,
        context: ScenarioExecutionContext,
        outcome: PaymentWorkflowOutcome,
    ) -> None:
        ...


@dataclass
class RecordingScenarioApplicationObservability:
    """In-memory recorder for tests and local diagnostics."""

    started: list[ScenarioExecutionContext] = field(default_factory=list)
    actions: list[tuple[ScenarioExecutionContext, BusinessActionKind, dict[str, Any]]] = field(
        default_factory=list
    )
    prepared: list[tuple[ScenarioExecutionContext, PaymentWorkflowOutcome]] = field(
        default_factory=list
    )

    def scenario_started(self, context: ScenarioExecutionContext) -> None:
        self.started.append(context)

    def business_action_executed(
        self,
        context: ScenarioExecutionContext,
        *,
        kind: BusinessActionKind,
        detail: dict[str, Any],
    ) -> None:
        self.actions.append((context, kind, detail))

    def scenario_state_prepared(
        self,
        context: ScenarioExecutionContext,
        outcome: PaymentWorkflowOutcome,
    ) -> None:
        self.prepared.append((context, outcome))
