# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from intergrax.eval.eval_case import EvalCase
from intergrax.eval.nexus_eval_runner import NexusEvalRunner
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.execution.harness_task_execution_port import (
    build_harness_root_task_execution_port,
)
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from testing_support.admitted_root_governance_identity import (
    lab_admitted_root_governance_identity_for_task,
)
from testing_support.builder import build_runtime_request_for_tests
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskState


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_eval_runner_runs_echo_case():
    registry = AgentRegistry()
    registry.register(EchoAgent())
    runner = NexusEvalRunner(
        UnifiedTaskRunner(
            build_harness_root_task_execution_port(NexusLoop(registry)),
            admitted_governance_identity_for_task=lab_admitted_root_governance_identity_for_task,
        )
    )

    case = EvalCase(
        case_id="echo-1",
        runtime_request=build_runtime_request_for_tests(
            seed="eval-echo-1",
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
    runner = NexusEvalRunner(
        UnifiedTaskRunner(
            build_harness_root_task_execution_port(NexusLoop(registry)),
            admitted_governance_identity_for_task=lab_admitted_root_governance_identity_for_task,
        )
    )

    case = EvalCase(
        case_id="echo-mismatch",
        runtime_request=build_runtime_request_for_tests(
            seed="eval-echo-mismatch",
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
