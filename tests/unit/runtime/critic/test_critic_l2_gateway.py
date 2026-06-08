# © Artur Czarnecki. All rights reserved.

"""L2 gateway and orchestrator integration tests (Phase CRIT-V-FOLLOWUP)."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.critic.contracts import CriticAction, CriticLayer, CriticRequest, CriticScope
from intergrax.runtime.critic.critic_orchestrator import CriticOrchestrator

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_critic_orchestrator_l2_escalates_hitl() -> None:
    orchestrator = CriticOrchestrator()
    request = CriticRequest(
        scope=CriticScope.GRAPH_FINAL,
        run_id="run-1",
        agent_id="worker",
        execution=AgentExecutionResult(
            agent_id="worker",
            run_id="run-1",
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
        ),
        enabled_layers=(
            CriticLayer.L0_DETERMINISTIC,
            CriticLayer.L2_HUMAN,
        ),
    )
    verdict = orchestrator.verify_final(request)
    assert verdict.passed is False
    assert verdict.recommended_action is CriticAction.ESCALATE_HITL
    assert verdict.layers[-1].layer is CriticLayer.L2_HUMAN
