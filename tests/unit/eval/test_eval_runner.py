# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.eval.eval_runner import EvalRunner
from intergrax.eval.eval_case import EvalCase
from intergrax.eval.eval_result import EvalResult

from intergrax.runtime.nexus.responses.response_schema import (
    RuntimeRequest,
    RuntimeAnswer,
    RuntimeStats,
)

from intergrax.runtime.replay.models import (
    ReconstructedRun,
)

from intergrax.runtime.replay.metrics import ExecutionMetrics


# -----------------------------
# Test Stubs (minimal, explicit)
# -----------------------------

class StubRuntimeEngine:

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        return RuntimeAnswer(
            answer="expected",
            run_id="run-123",
            trace_events=[],
            stats=RuntimeStats(
                total_tokens=42,
                duration_ms=1,
            ),
            tool_calls=[],
            llm_usage_report=None,
        )


class StubReplayEngine:

    def reconstruct(self, run_id: str) -> ReconstructedRun:
        return ReconstructedRun(
            run_id=run_id,
            steps=[],
            artifacts=[],
            tool_calls=[],
            llm_calls=[],
            final_answer="expected",
        )


class StubMetricsEngine:

    def compute(self, reconstructed: ReconstructedRun) -> ExecutionMetrics:
        return ExecutionMetrics(
            step_count=1,
            total_llm_calls=1,
            total_tool_calls=0,
            total_artifacts=0,
            total_tokens=42,
            duration=0.001,
            tool_steps_ratio=0.0,
            llm_steps_ratio=1.0,
        )


# -----------------------------
# Test
# -----------------------------

@pytest.mark.asyncio
async def test_run_case_success_exact_match():

    runtime = StubRuntimeEngine()
    replay = StubReplayEngine()
    metrics = StubMetricsEngine()

    runner = EvalRunner(
        runtime_engine=runtime,
        replay_engine=replay,
        metrics_engine=metrics,
    )

    request = RuntimeRequest(
        tenant_id="test-tenant",
        agent_id="agent",
        user_id="user",
        session_id="session",
        message="prompt",
    )

    case = EvalCase(
        case_id="case-1",
        runtime_request=request,
        expected_output="expected",
    )

    result: EvalResult = await runner.run_case(case)

    assert result.case_id == "case-1"
    assert result.success is True
    assert result.final_answer == "expected"
    assert result.total_tokens == 42
    assert result.total_cost == 42.0
    assert result.tool_calls_count == 0
    assert result.error is None


@pytest.mark.asyncio
async def test_run_case_failure_exact_mismatch():

    class StubReplayEngineMismatch:

        def reconstruct(self, run_id: str) -> ReconstructedRun:
            return ReconstructedRun(
                run_id=run_id,
                steps=[],
                artifacts=[],
                tool_calls=[],
                llm_calls=[],
                final_answer="wrong",
            )

    runtime = StubRuntimeEngine()
    replay = StubReplayEngineMismatch()
    metrics = StubMetricsEngine()

    runner = EvalRunner(
        runtime_engine=runtime,
        replay_engine=replay,
        metrics_engine=metrics,
    )

    request = RuntimeRequest(
        tenant_id="test-tenant",
        agent_id="agent",
        user_id="user",
        session_id="session",
        message="prompt",
    )

    case = EvalCase(
        case_id="case-2",
        runtime_request=request,
        expected_output="expected",
    )

    result: EvalResult = await runner.run_case(case)

    assert result.case_id == "case-2"
    assert result.success is False
    assert result.final_answer == "wrong"
    assert result.total_tokens == 42
    assert result.total_cost == 42.0
    assert result.tool_calls_count == 0
    assert result.error is None


@pytest.mark.asyncio
async def test_run_case_missing_run_id():

    class StubRuntimeEngineNoRunId:

        async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
            return RuntimeAnswer(
                answer="expected",
                run_id=None,
                trace_events=[],
                stats=RuntimeStats(
                    total_tokens=42,
                    duration_ms=1,
                ),
                tool_calls=[],
                llm_usage_report=None,
            )

    runtime = StubRuntimeEngineNoRunId()
    replay = StubReplayEngine()
    metrics = StubMetricsEngine()

    runner = EvalRunner(
        runtime_engine=runtime,
        replay_engine=replay,
        metrics_engine=metrics,
    )

    request = RuntimeRequest(
        tenant_id="test-tenant",
        agent_id="agent",
        user_id="user",
        session_id="session",
        message="prompt",
    )

    case = EvalCase(
        case_id="case-3",
        runtime_request=request,
        expected_output="expected",
    )

    result: EvalResult = await runner.run_case(case)

    assert result.case_id == "case-3"
    assert result.success is False
    assert result.final_answer == ""
    assert result.total_tokens == 0
    assert result.total_cost == 0.0
    assert result.tool_calls_count == 0
    assert result.error == "Missing run_id in RuntimeAnswer"