# © Artur Czarnecki. All rights reserved.

"""CriticOrchestrator unit tests (Phase CRIT-V-3.1)."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticRequest,
    CriticScope,
    RubricSpec,
)
from intergrax.runtime.critic.critic_orchestrator import CriticOrchestrator
from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient
from intergrax.runtime.critic.l1_gateway import L1Gateway
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput, EvalTrajectoryInput, EvalTrajectoryOutput

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeEvalClient(CriticEvalToolClient):
    def __init__(
        self,
        *,
        judge_passed: bool = True,
        judge_score: float = 0.9,
        trajectory_passed: bool = True,
        trajectory_score: float = 0.85,
    ) -> None:
        self._judge_passed = judge_passed
        self._judge_score = judge_score
        self._trajectory_passed = trajectory_passed
        self._trajectory_score = trajectory_score
        self.judge_calls = 0
        self.trajectory_calls = 0

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        self.judge_calls += 1
        return EvalJudgeOutput(
            rubric_id=params.rubric_id,
            score=self._judge_score,
            passed=self._judge_passed,
            reasons=[] if self._judge_passed else ["below threshold"],
        )

    def trajectory(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        self.trajectory_calls += 1
        return EvalTrajectoryOutput(
            run_id=params.run_id,
            score=self._trajectory_score,
            passed=self._trajectory_passed,
            reasons=[] if self._trajectory_passed else ["trajectory anomaly"],
        )


def _completed_execution(summary: str = "valid output") -> AgentExecutionResult:
    return AgentExecutionResult(
        agent_id="worker",
        run_id="run-1",
        status=AgentExecutionStatus.COMPLETED,
        summary=summary,
    )


def test_critic_orchestrator_l0_only_passes() -> None:
    orchestrator = CriticOrchestrator()
    request = CriticRequest(
        scope=CriticScope.NODE_PARTIAL,
        run_id="run-1",
        agent_id="worker",
        execution=_completed_execution(),
        enabled_layers=(CriticLayer.L0_DETERMINISTIC,),
    )
    verdict = orchestrator.verify(request)
    assert verdict.passed is True
    assert verdict.recommended_action is CriticAction.CONTINUE
    assert len(verdict.layers) == 1
    assert verdict.layers[0].layer is CriticLayer.L0_DETERMINISTIC


def test_critic_orchestrator_l0_fail_short_circuits_before_l1() -> None:
    client = _FakeEvalClient()
    orchestrator = CriticOrchestrator(l1_gateway=L1Gateway(tool_client=client))
    request = CriticRequest(
        scope=CriticScope.NODE_PARTIAL,
        run_id="run-1",
        agent_id="worker",
        execution=AgentExecutionResult(
            agent_id="worker",
            run_id="run-1",
            status=AgentExecutionStatus.FAILED,
            summary="",
            errors=["boom"],
        ),
        enabled_layers=(
            CriticLayer.L0_DETERMINISTIC,
            CriticLayer.L1_SEMANTIC,
        ),
        rubric=RubricSpec(rubric_id="case.a"),
    )
    verdict = orchestrator.verify(request)
    assert verdict.passed is False
    assert verdict.recommended_action is CriticAction.RETRY
    assert client.judge_calls == 0


def test_critic_orchestrator_runs_layers_in_order() -> None:
    client = _FakeEvalClient()
    orchestrator = CriticOrchestrator(l1_gateway=L1Gateway(tool_client=client))
    request = CriticRequest(
        scope=CriticScope.GRAPH_FINAL,
        run_id="run-1",
        agent_id="worker",
        execution=_completed_execution(),
        enabled_layers=(
            CriticLayer.L0_DETERMINISTIC,
            CriticLayer.L1_SEMANTIC,
            CriticLayer.L1_TRAJECTORY,
        ),
        rubric=RubricSpec(rubric_id="case.a", min_score=0.7),
        context={"tenant_id": "tenant-1"},
    )
    verdict = orchestrator.verify(request)
    assert verdict.passed is True
    assert [layer.layer for layer in verdict.layers] == [
        CriticLayer.L0_DETERMINISTIC,
        CriticLayer.L1_SEMANTIC,
        CriticLayer.L1_TRAJECTORY,
    ]
    assert client.judge_calls == 1
    assert client.trajectory_calls == 1


def test_critic_orchestrator_l1_fail_recommends_revise() -> None:
    client = _FakeEvalClient(judge_passed=False, judge_score=0.2)
    orchestrator = CriticOrchestrator(l1_gateway=L1Gateway(tool_client=client))
    request = CriticRequest(
        scope=CriticScope.NODE_PARTIAL,
        run_id="run-1",
        agent_id="worker",
        execution=_completed_execution(),
        enabled_layers=(CriticLayer.L0_DETERMINISTIC, CriticLayer.L1_SEMANTIC),
        rubric=RubricSpec(rubric_id="case.a"),
    )
    verdict = orchestrator.verify_partial(request)
    assert verdict.passed is False
    assert verdict.recommended_action is CriticAction.REVISE
    assert verdict.scope is CriticScope.NODE_PARTIAL


def test_critic_orchestrator_l0_fail_on_final_recommends_fail() -> None:
    orchestrator = CriticOrchestrator()
    request = CriticRequest(
        scope=CriticScope.GRAPH_FINAL,
        run_id="run-1",
        agent_id="worker",
        execution=AgentExecutionResult(
            agent_id="worker",
            run_id="run-1",
            status=AgentExecutionStatus.FAILED,
            summary="",
        ),
        enabled_layers=(CriticLayer.L0_DETERMINISTIC,),
    )
    verdict = orchestrator.verify_final(request)
    assert verdict.passed is False
    assert verdict.recommended_action is CriticAction.FAIL


def test_critic_orchestrator_never_emits_l2_layer() -> None:
    orchestrator = CriticOrchestrator()
    request = CriticRequest(
        scope=CriticScope.GRAPH_FINAL,
        run_id="run-1",
        agent_id="worker",
        execution=_completed_execution(),
        enabled_layers=(
            CriticLayer.L0_DETERMINISTIC,
            CriticLayer.L1_SEMANTIC,
            CriticLayer.L1_TRAJECTORY,
        ),
        rubric=RubricSpec(rubric_id="case.a"),
        context={"tenant_id": "tenant-1"},
    )
    client = _FakeEvalClient()
    orchestrator = CriticOrchestrator(l1_gateway=L1Gateway(tool_client=client))
    verdict = orchestrator.verify(request)
    emitted_layers = {layer.layer for layer in verdict.layers}
    assert CriticLayer.L0_DETERMINISTIC in emitted_layers or len(verdict.layers) >= 1
    assert all(
        layer.layer in (
            CriticLayer.L0_DETERMINISTIC,
            CriticLayer.L1_SEMANTIC,
            CriticLayer.L1_TRAJECTORY,
        )
        for layer in verdict.layers
    )
    assert verdict.recommended_action in (
        CriticAction.CONTINUE,
        CriticAction.RETRY,
        CriticAction.REVISE,
        CriticAction.FAIL,
    )


def test_critic_orchestrator_uses_contract_from_context() -> None:
    orchestrator = CriticOrchestrator()
    contract = AgentContract(
        id="worker",
        name="Worker",
        description="test",
        validation_rules=["no_errors"],
    )
    request = CriticRequest(
        scope=CriticScope.NODE_PARTIAL,
        run_id="run-1",
        agent_id="worker",
        execution=AgentExecutionResult(
            agent_id="worker",
            run_id="run-1",
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
            errors=["leftover"],
        ),
        enabled_layers=(CriticLayer.L0_DETERMINISTIC,),
        context={"contract": contract},
    )
    verdict = orchestrator.verify(request)
    assert verdict.passed is False
    assert any("no_errors" in reason for reason in verdict.failure_reasons)
