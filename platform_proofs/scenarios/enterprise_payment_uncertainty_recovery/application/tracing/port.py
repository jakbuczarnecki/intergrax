"""Scenario execution trace port — business lifecycle steps only."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from intergrax.runtime.nexus.tracing.trace_models import TraceEvent


class ScenarioExecutionTraceStepId(StrEnum):
    SCENARIO_EXECUTION_STARTED = "scenario_execution_started"
    PAYMENT_WORKFLOW_STARTED = "payment_workflow_started"
    EXTERNAL_PAYMENT_EFFECT_CREATED = "external_payment_effect_created"
    UNKNOWN_DETECTED = "unknown_detected"
    RELIABILITY_CASE_CREATED = "reliability_case_created"
    RECONCILIATION_EXECUTED = "reconciliation_executed"
    EVIDENCE_EVALUATED = "evidence_evaluated"
    RESOLUTION_DECIDED = "resolution_decided"
    GOVERNANCE_EVALUATED = "governance_evaluated"
    RECOVERY_EXECUTED = "recovery_executed"
    SCENARIO_COMPLETED = "scenario_completed"


class ScenarioExecutionTracePort(Protocol):
    def emit_lifecycle_step(
        self,
        step_id: ScenarioExecutionTraceStepId,
        *,
        outcome: str,
        component_identity: str,
        business_detail: dict[str, Any] | None = None,
    ) -> None:
        ...

    def snapshot(self) -> tuple[TraceEvent, ...]:
        ...
