from __future__ import annotations

from intergrax.runtime.architecture.multi_agent_coordination import (
    CoordinationPattern,
    PlanningConstraints,
    PlanningDimensionLevel,
    build_default_coordination_catalog,
    select_coordination_pattern,
)


def test_coordination_catalog_contains_core_patterns() -> None:
    catalog = build_default_coordination_catalog()
    patterns = {item.pattern for item in catalog.patterns}
    assert CoordinationPattern.SWARM in patterns
    assert CoordinationPattern.EVALUATOR_LOOP in patterns


def test_pattern_selection_prefers_supervisor_for_high_risk() -> None:
    report = select_coordination_pattern(
        constraints=PlanningConstraints(
            risk_level=PlanningDimensionLevel.HIGH,
            latency_level=PlanningDimensionLevel.LOW,
            cost_level=PlanningDimensionLevel.LOW,
            complexity_level=PlanningDimensionLevel.LOW,
        )
    )
    assert report.decision.selected_pattern == CoordinationPattern.SUPERVISOR_WORKER
