# © Artur Czarnecki. All rights reserved.

"""UE-11B — canonical agentic spine integration with real LLM (synthetic workload)."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.execution_identity import validate_execution_id
from intergrax.runtime.execution import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from testing_support.ue_11b_real_root_execution import (
    assert_completed_agentic_result,
    budget_evidence,
    build_agentic_stack,
    correlated_agentic_inputs,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.external_proof,
    pytest.mark.no_ci,
]

_REPEAT_RUNS = 3


@pytest.mark.parametrize("run_index", range(_REPEAT_RUNS))
async def test_ue_11b_canonical_agentic_spine_with_real_llm(run_index: int) -> None:
    del run_index
    stack = build_agentic_stack()
    run_id, options, runtime_request = correlated_agentic_inputs()
    request = ExecutionRequest(
        input=runtime_request,
        output_type=AgentExecutionResult,
        capabilities=frozenset({ExecutionCapability.TOOLS, ExecutionCapability.AGENT}),
    )

    assert StrategyResolver().resolve(request) is ExecutionStrategy.AGENTIC

    result = await stack.execution.execute(request, options=options)
    agent_result = assert_completed_agentic_result(result)

    assert agent_result.agent_id == stack.agent_id
    assert agent_result.run_id == run_id
    assert stack.admission_hook.captured_execution_id is not None
    assert validate_execution_id(stack.admission_hook.captured_execution_id)
    assert stack.admission_hook.captured_run_id == run_id

    attempt_id = stack.admission_hook.captured_attempt_id
    assert attempt_id is not None
    evidence = budget_evidence(stack.ledger, attempt_id=attempt_id)
    assert evidence.llm_calls > 0
    assert "[tool:" in agent_result.summary
