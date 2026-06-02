# © Artur Czarnecki. All rights reserved.

"""Multi-agent coordination pattern catalog and selection matrix (Phase V-MA.1/V-MA.2)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CoordinationPattern(str, Enum):
    HIERARCHICAL = "hierarchical"
    ORCHESTRATOR_WORKER = "orchestrator_worker"
    SUPERVISOR_WORKER = "supervisor_worker"
    PEER_TO_PEER = "peer_to_peer"
    SWARM = "swarm"
    EVALUATOR_LOOP = "evaluator_loop"


class PlanningDimensionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PlanningConstraints(BaseModel):
    risk_level: PlanningDimensionLevel
    latency_level: PlanningDimensionLevel
    cost_level: PlanningDimensionLevel
    complexity_level: PlanningDimensionLevel


class CoordinationPatternDefinition(BaseModel):
    pattern: CoordinationPattern
    description: str
    min_agents: int = 1
    max_agents: int = 32
    supports_human_gate: bool = False


class CoordinationPatternCatalog(BaseModel):
    schema_version: str = "1.0.0"
    patterns: list[CoordinationPatternDefinition] = Field(default_factory=list)


class PatternSelectionDecision(BaseModel):
    selected_pattern: CoordinationPattern
    reasons: list[str] = Field(default_factory=list)


class PatternSelectionMatrixReport(BaseModel):
    schema_version: str = "1.0.0"
    constraints: PlanningConstraints
    decision: PatternSelectionDecision


def build_default_coordination_catalog() -> CoordinationPatternCatalog:
    return CoordinationPatternCatalog(
        patterns=[
            CoordinationPatternDefinition(
                pattern=CoordinationPattern.HIERARCHICAL,
                description="Top-down planner delegates to specialized workers.",
                min_agents=2,
                max_agents=16,
                supports_human_gate=True,
            ),
            CoordinationPatternDefinition(
                pattern=CoordinationPattern.ORCHESTRATOR_WORKER,
                description="Single orchestrator coordinates deterministic worker steps.",
                min_agents=2,
                max_agents=24,
            ),
            CoordinationPatternDefinition(
                pattern=CoordinationPattern.SUPERVISOR_WORKER,
                description="Supervisor monitors worker outputs and re-plans on failure.",
                min_agents=2,
                max_agents=20,
                supports_human_gate=True,
            ),
            CoordinationPatternDefinition(
                pattern=CoordinationPattern.PEER_TO_PEER,
                description="Agents negotiate directly with shared task contract.",
                min_agents=2,
                max_agents=12,
            ),
            CoordinationPatternDefinition(
                pattern=CoordinationPattern.SWARM,
                description="Many lightweight agents explore in parallel and aggregate.",
                min_agents=3,
                max_agents=32,
            ),
            CoordinationPatternDefinition(
                pattern=CoordinationPattern.EVALUATOR_LOOP,
                description="Generator and evaluator iterate until quality gate passes.",
                min_agents=2,
                max_agents=8,
            ),
        ]
    )


def select_coordination_pattern(
    *,
    constraints: PlanningConstraints,
    catalog: CoordinationPatternCatalog | None = None,
) -> PatternSelectionMatrixReport:
    _ = catalog or build_default_coordination_catalog()
    reasons: list[str] = []
    if constraints.complexity_level == PlanningDimensionLevel.HIGH:
        selected = CoordinationPattern.HIERARCHICAL
        reasons.append("High complexity favors hierarchical decomposition")
    elif constraints.risk_level == PlanningDimensionLevel.HIGH:
        selected = CoordinationPattern.SUPERVISOR_WORKER
        reasons.append("High risk favors supervised worker pattern")
    elif constraints.cost_level == PlanningDimensionLevel.HIGH:
        selected = CoordinationPattern.ORCHESTRATOR_WORKER
        reasons.append("High cost pressure favors orchestrator-worker efficiency")
    elif constraints.latency_level == PlanningDimensionLevel.HIGH:
        selected = CoordinationPattern.SWARM
        reasons.append("High latency tolerance allows parallel swarm exploration")
    else:
        selected = CoordinationPattern.EVALUATOR_LOOP
        reasons.append("Balanced constraints favor evaluator loop quality gate")
    return PatternSelectionMatrixReport(
        constraints=constraints,
        decision=PatternSelectionDecision(selected_pattern=selected, reasons=reasons),
    )
