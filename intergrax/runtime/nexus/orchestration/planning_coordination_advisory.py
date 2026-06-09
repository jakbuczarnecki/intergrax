# © Artur Czarnecki. All rights reserved.

"""Coordination pattern advisory trace (ORCH-5.3)."""

from __future__ import annotations

from intergrax.runtime.architecture.multi_agent_coordination import (
    PlanningConstraints,
    PlanningDimensionLevel,
    select_coordination_pattern,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType


def build_coordination_advisory_event(
    *,
    task_id: str,
    tenant_id: str,
) -> RuntimeEvent:
    report = select_coordination_pattern(
        constraints=PlanningConstraints(
            complexity_level=PlanningDimensionLevel.MEDIUM,
            risk_level=PlanningDimensionLevel.MEDIUM,
            cost_level=PlanningDimensionLevel.MEDIUM,
            latency_level=PlanningDimensionLevel.MEDIUM,
        ),
    )
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_PROGRESS,
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=task_id,
        phase=ExecutionPhase.PLANNING,
        payload={
            "event_kind": "COORDINATION_PATTERN_ADVISORY",
            "selected_pattern": report.decision.selected_pattern.value,
            "reasons": list(report.decision.reasons),
        },
    )
