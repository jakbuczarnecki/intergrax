# © Artur Czarnecki. All rights reserved.

"""UE-11B — canonical orchestration spine integration with real LLM (synthetic workload)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import validate_execution_id
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.task.task import TaskState
from testing_support.ue_11b_real_root_execution import (
    budget_evidence,
    build_orchestration_stack,
    child_execution_records,
    orchestration_request,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.external_proof,
    pytest.mark.no_ci,
]

_REPEAT_RUNS = 2


@pytest.mark.parametrize("run_index", range(_REPEAT_RUNS))
async def test_ue_11b_canonical_orchestration_spine_with_real_llm(run_index: int) -> None:
    del run_index
    stack = build_orchestration_stack()
    request = orchestration_request(stack.task)

    assert StrategyResolver().resolve(request) is ExecutionStrategy.ORCHESTRATION

    result = await stack.execution.execute(request, options=stack.options)

    assert result.state == TaskState.COMPLETED
    assert result.answer
    assert stack.admission_hook.captured_execution_id is not None
    root_execution_id = stack.admission_hook.captured_execution_id
    assert validate_execution_id(root_execution_id)

    child_ids = child_execution_records(
        stack.ledger,
        attempt_id=stack.admission_hook.captured_attempt_id or stack.options.attempt_id,
        root_execution_id=root_execution_id,
    )
    assert child_ids
    assert all(child_id != root_execution_id for child_id in child_ids)
    assert all(validate_execution_id(child_id) for child_id in child_ids)

    evidence = budget_evidence(
        stack.ledger,
        attempt_id=stack.admission_hook.captured_attempt_id or stack.options.attempt_id,
    )
    assert evidence.llm_calls > 0

    agent_ids = result.metadata.get("agent_ids")
    assert isinstance(agent_ids, list)
    assert len(agent_ids) >= 2
