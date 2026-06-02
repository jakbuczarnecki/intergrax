from __future__ import annotations

from intergrax.runtime.architecture.multi_agent_acceptance import (
    MultiAgentAcceptanceCase,
    evaluate_multi_agent_acceptance,
)
from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern


def test_multi_agent_acceptance_passes_valid_supervisor_case() -> None:
    report = evaluate_multi_agent_acceptance(
        [
            MultiAgentAcceptanceCase(
                case_id="case-1",
                pattern=CoordinationPattern.SUPERVISOR_WORKER,
                agent_count=3,
                completed_steps=4,
                expected_steps=4,
                human_gate_satisfied=True,
            )
        ]
    )
    assert report.passed is True


def test_multi_agent_acceptance_fails_when_human_gate_missing() -> None:
    report = evaluate_multi_agent_acceptance(
        [
            MultiAgentAcceptanceCase(
                case_id="case-2",
                pattern=CoordinationPattern.HIERARCHICAL,
                agent_count=2,
                completed_steps=3,
                expected_steps=3,
                human_gate_satisfied=False,
            )
        ]
    )
    assert report.passed is False
