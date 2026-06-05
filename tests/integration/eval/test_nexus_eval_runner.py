# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from intergrax.eval.eval_case import EvalCase
from intergrax.eval.nexus_eval_runner import NexusEvalRunner
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskState


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_eval_runner_runs_echo_case():
    registry = AgentRegistry()
    registry.register(EchoAgent())
    runner = NexusEvalRunner.from_nexus_loop(NexusLoop(registry))

    case = EvalCase(
        case_id="echo-1",
        runtime_request=RuntimeRequest(
            agent_id="echo",
            user_id="eval-user",
            session_id="eval-session",
            message="hello eval",
            tenant_id="eval-tenant",
            metadata={"capability": "echo.basic"},
        ),
        expected_output="echo: hello eval",
    )

    result = await runner.run_case(case)

    assert result.success is True
    assert result.final_answer == "echo: hello eval"
    assert result.error is None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_eval_runner_reports_output_mismatch():
    registry = AgentRegistry()
    registry.register(EchoAgent())
    runner = NexusEvalRunner.from_nexus_loop(NexusLoop(registry))

    case = EvalCase(
        case_id="echo-mismatch",
        runtime_request=RuntimeRequest(
            agent_id="echo",
            user_id="eval-user",
            session_id="eval-session",
            message="hello eval",
            tenant_id="eval-tenant",
            metadata={"capability": "echo.basic"},
        ),
        expected_output="wrong answer",
    )

    result = await runner.run_case(case)

    assert result.success is False
    assert result.error == "output_mismatch"
    assert result.final_answer == "echo: hello eval"
