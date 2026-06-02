from __future__ import annotations

from intergrax.runtime.architecture.multi_agent_coordination import (
    PlanningConstraints,
    PlanningDimensionLevel,
)
from intergrax.runtime.architecture.runtime_governance_bridge import RuntimeArchitectureGovernanceBridge


def test_runtime_bridge_emits_coordination_pattern_metadata() -> None:
    bridge = RuntimeArchitectureGovernanceBridge()
    metadata = bridge.build_trace_metadata(
        constraints=PlanningConstraints(
            risk_level=PlanningDimensionLevel.HIGH,
            latency_level=PlanningDimensionLevel.LOW,
            cost_level=PlanningDimensionLevel.LOW,
            complexity_level=PlanningDimensionLevel.LOW,
        )
    )
    assert metadata.coordination_pattern == "supervisor_worker"
    assert metadata.adaptive_governance_passed is True
