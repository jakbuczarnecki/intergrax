# © Artur Czarnecki. All rights reserved.

"""CVL contract tests (Phase CRIT-V-1.2)."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticRequest,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
    RubricSpec,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_critic_request_defaults_to_l0_layer() -> None:
    request = CriticRequest(
        scope=CriticScope.NODE_PARTIAL,
        run_id="run-1",
        agent_id="agent-1",
    )
    assert request.enabled_layers == (CriticLayer.L0_DETERMINISTIC,)


def test_critic_verdict_carries_layer_results() -> None:
    verdict = CriticVerdict(
        scope=CriticScope.GRAPH_FINAL,
        passed=False,
        layers=[
            LayerVerdict(
                layer=CriticLayer.L0_DETERMINISTIC,
                passed=True,
                score=1.0,
            ),
            LayerVerdict(
                layer=CriticLayer.L1_SEMANTIC,
                passed=False,
                score=0.4,
                errors=["score below threshold"],
            ),
        ],
        recommended_action=CriticAction.REVISE,
        failure_reasons=["score below threshold"],
    )
    assert verdict.recommended_action is CriticAction.REVISE
    assert len(verdict.layers) == 2


def test_rubric_spec_enforces_score_bounds() -> None:
    rubric = RubricSpec(
        rubric_id="legal.summary",
        criteria=[" cites sources", "non-empty"],
        min_score=0.8,
    )
    assert rubric.min_score == 0.8


def test_critic_request_accepts_execution_payload() -> None:
    execution = AgentExecutionResult(
        agent_id="worker",
        run_id="run-worker",
        status=AgentExecutionStatus.COMPLETED,
        summary="done",
    )
    request = CriticRequest(
        scope=CriticScope.OFFLINE_CASE,
        run_id="eval-1",
        agent_id="worker",
        execution=execution,
        enabled_layers=(CriticLayer.L0_DETERMINISTIC, CriticLayer.L1_SEMANTIC),
        rubric=RubricSpec(rubric_id="case.rubric"),
    )
    assert request.execution is execution
    assert CriticLayer.L1_SEMANTIC in request.enabled_layers
