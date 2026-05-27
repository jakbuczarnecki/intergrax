# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.runtime_cost import (
    aggregate_execution_metrics,
    extract_cost_from_runtime_answer,
    extract_duration_seconds_from_runtime_answer,
    tokens_to_cost_units,
)
from intergrax.contracts.runtime_mapping import runtime_answer_to_agent_result
from intergrax.llm_adapters.contracts.llm_adapter import LLMRunStats
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageReport
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeStats

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_tokens_to_cost_units():
    assert tokens_to_cost_units(42) == 42.0
    assert tokens_to_cost_units(-1) == 0.0


def test_extract_cost_from_runtime_answer_prefers_llm_usage_report():
    answer = RuntimeAnswer(
        answer="ok",
        llm_usage_report=LLMUsageReport(
            run_id="run-1",
            total=LLMRunStats(total_tokens=17, duration_ms=250),
            entries=[],
            by_provider_model={},
            adapter_instance_ids={},
        ),
    )
    assert extract_cost_from_runtime_answer(answer) == 17.0
    assert extract_duration_seconds_from_runtime_answer(answer) == 0.25


def test_runtime_answer_to_agent_result_populates_cost_and_duration():
    answer = RuntimeAnswer(
        answer="hello",
        run_id="run-1",
        stats=RuntimeStats(duration_ms=500, extra={"cost": 3.5}),
    )
    result = runtime_answer_to_agent_result(answer, agent_id="echo", valid=True)
    assert result.cost == 3.5
    assert result.duration_seconds == 0.5


def test_aggregate_execution_metrics_sums_cost():
    executions = [
        AgentExecutionResult(
            agent_id="a",
            run_id="r1",
            status=AgentExecutionStatus.COMPLETED,
            cost=2.0,
            duration_seconds=0.1,
        ),
        AgentExecutionResult(
            agent_id="b",
            run_id="r2",
            status=AgentExecutionStatus.COMPLETED,
            cost=5.0,
            duration_seconds=0.4,
        ),
    ]
    metrics = aggregate_execution_metrics(executions)
    assert metrics.cost == 7.0
    assert metrics.total_tokens == 7
    assert metrics.duration_ms == 400
    assert metrics.as_llm_usage()["cost"] == 7.0
