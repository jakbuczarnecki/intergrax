# © Artur Czarnecki. All rights reserved.

"""Pattern-specific multi-agent acceptance contracts (Phase V-MA.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern


class MultiAgentAcceptanceCase(BaseModel):
    case_id: str
    pattern: CoordinationPattern
    agent_count: int
    completed_steps: int
    expected_steps: int
    human_gate_satisfied: bool = True


class MultiAgentAcceptanceResult(BaseModel):
    case_id: str
    pattern: CoordinationPattern
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class MultiAgentAcceptanceReport(BaseModel):
    schema_version: str = "1.0.0"
    results: list[MultiAgentAcceptanceResult] = Field(default_factory=list)
    passed: bool


def evaluate_multi_agent_acceptance(
    cases: list[MultiAgentAcceptanceCase],
) -> MultiAgentAcceptanceReport:
    results: list[MultiAgentAcceptanceResult] = []
    for case in cases:
        reasons: list[str] = []
        if case.agent_count < _min_agents_for_pattern(case.pattern):
            reasons.append("Agent count below pattern minimum")
        if case.completed_steps < case.expected_steps:
            reasons.append("Not all coordination steps completed")
        if case.pattern in {
            CoordinationPattern.SUPERVISOR_WORKER,
            CoordinationPattern.HIERARCHICAL,
        } and not case.human_gate_satisfied:
            reasons.append("Human gate requirement not satisfied")
        results.append(
            MultiAgentAcceptanceResult(
                case_id=case.case_id,
                pattern=case.pattern,
                passed=not reasons,
                reasons=reasons,
            )
        )
    return MultiAgentAcceptanceReport(
        results=results,
        passed=all(result.passed for result in results),
    )


def _min_agents_for_pattern(pattern: CoordinationPattern) -> int:
    if pattern == CoordinationPattern.SWARM:
        return 3
    return 2
