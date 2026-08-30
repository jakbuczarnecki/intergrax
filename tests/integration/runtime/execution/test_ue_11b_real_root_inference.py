# © Artur Czarnecki. All rights reserved.

"""UE-11B — real root inference end-to-end proof."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import validate_execution_id
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.execution import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from testing_support.ue_11b_real_root_execution import (
    UE_11B_INFERENCE_PROMPT,
    TextCategoryClassification,
    assert_completed_inference_result,
    budget_evidence,
    build_inference_stack,
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
async def test_ue_11b_real_root_inference_end_to_end(run_index: int) -> None:
    del run_index
    stack = build_inference_stack()
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content=UE_11B_INFERENCE_PROMPT),),
        output_type=TextCategoryClassification,
    )

    assert StrategyResolver().resolve(request) is ExecutionStrategy.INFERENCE
    assert ExecutionCapability.ORCHESTRATION not in request.capabilities

    result = await stack.execution.execute(request, options=stack.options)
    parsed = assert_completed_inference_result(result)

    assert parsed.category in {"platform", "validation"}
    assert stack.admission_hook.captured_execution_id is not None
    assert validate_execution_id(stack.admission_hook.captured_execution_id)
    assert stack.admission_hook.captured_run_id is not None
    assert stack.admission_hook.captured_attempt_id is not None

    evidence = budget_evidence(
        stack.ledger,
        attempt_id=stack.admission_hook.captured_attempt_id,
    )
    assert evidence.llm_calls > 0
    assert evidence.total_tokens > 0
