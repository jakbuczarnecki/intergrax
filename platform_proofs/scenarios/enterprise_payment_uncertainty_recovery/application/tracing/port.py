"""Scenario execution trace port — business lifecycle steps only."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol

from intergrax.contracts.tracing import TraceEvent

TraceAttributeValue = str | int | float | bool | None
TraceBusinessDetail = Mapping[str, TraceAttributeValue]


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
    def begin_execution(
        self,
        *,
        correlation_id: str,
        scenario_id: str,
        variant_id: str,
    ) -> None:
        ...

    def emit_lifecycle_step(
        self,
        step_id: ScenarioExecutionTraceStepId,
        *,
        outcome: str,
        component_identity: str,
        business_detail: TraceBusinessDetail | None = None,
    ) -> None:
        ...

    def snapshot(self) -> tuple[TraceEvent, ...]:
        ...
