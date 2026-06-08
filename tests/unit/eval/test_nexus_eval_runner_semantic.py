# © Artur Czarnecki. All rights reserved.

"""CRIT-V-5 semantic NexusEvalRunner tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.eval.eval_case import EvalCase
from intergrax.eval.nexus_eval_runner import NexusEvalRunner
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput
from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeSemanticClient(CriticEvalToolClient):
    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        passed = "equivalent" in params.output_text.lower()
        return EvalJudgeOutput(
            rubric_id=params.rubric_id,
            score=0.9 if passed else 0.2,
            passed=passed,
            reasons=[] if passed else ["not equivalent"],
        )

    def trajectory(self, params):  # noqa: ANN001
        raise NotImplementedError


class _StubTaskRunner:
    async def run_runtime_request(self, request, *, tenant_id, user_id, capability=None):  # noqa: ANN001
        _ = tenant_id, user_id, capability
        return TaskResult(
            task_id="task-1",
            state=TaskState.COMPLETED,
            answer="semantically equivalent answer",
            agent_id=request.agent_id,
            run_id="run-semantic-1",
            execution_result=AgentExecutionResult(
                agent_id=request.agent_id or "agent",
                run_id="run-semantic-1",
                status=AgentExecutionStatus.COMPLETED,
                summary="semantically equivalent answer",
            ),
        )


@pytest.mark.asyncio
async def test_nexus_eval_runner_semantic_mode_passes_non_exact_output() -> None:
    case = EvalCase(
        case_id="semantic-1",
        runtime_request=RuntimeRequest(
            message="evaluate",
            tenant_id="t1",
            user_id="u1",
            session_id="session-1",
            agent_id="agent-1",
        ),
        expected_output="exact canonical answer",
        semantic_match_enabled=True,
        rubric_ref="prompt.rubric.default",
        semantic_threshold=0.75,
    )
    runner = NexusEvalRunner(_StubTaskRunner(), semantic_client=_FakeSemanticClient())
    result = await runner.run_case(case)
    assert result.success is True


@pytest.mark.asyncio
async def test_nexus_eval_runner_exact_mode_still_default() -> None:
    case = EvalCase(
        case_id="exact-1",
        runtime_request=RuntimeRequest(
            message="evaluate",
            tenant_id="t1",
            user_id="u1",
            session_id="session-1",
            agent_id="agent-1",
        ),
        expected_output="exact canonical answer",
    )
    runner = NexusEvalRunner(_StubTaskRunner(), semantic_client=_FakeSemanticClient())
    result = await runner.run_case(case)
    assert result.success is False
    assert result.error == "output_mismatch"
