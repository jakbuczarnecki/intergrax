# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

import pytest

from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

from local_workspace_application.benchmarks.local_model_qualification.corpus import case_by_id
from local_workspace_application.benchmarks.local_model_qualification.protocols import (
    PROTOCOL_SINGLE_PLAN_TOOL,
    SUBMIT_DRAFT_TOOL_SCHEMA,
    build_protocol_messages,
    run_protocol_attempt,
)
from local_workspace_application.conversation.interaction_draft_models import ConversationInteractionDraft
from local_workspace_application.conversation.interaction_models import ConversationInteractionPlan
from local_workspace_application.benchmarks.local_model_qualification.config import BenchmarkConfig

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.no_ci,
]


def test_full_schema_tool_protocol_live() -> None:
    if os.environ.get("INTERGRAX_LKW_MODEL_QUALIFICATION_TOOL_E2E") != "1":
        pytest.skip("Set INTERGRAX_LKW_MODEL_QUALIFICATION_TOOL_E2E=1 to run live tool proof")

    model_name = os.environ.get("INTERGRAX_LLM_MODEL", "qwen2.5:14b")
    adapter = LLMAdapterRegistry.create(
        LLMProvider.OLLAMA,
        model=model_name,
        base_url="http://localhost:11434",
        keep_alive="10m",
    )
    benchmark = BenchmarkConfig(
        repetitions=1,
        warmup_runs=0,
        temperature=0.0,
        max_tokens=8192,
        randomize_case_order=False,
    )
    request = case_by_id("planner.workspace_list").request
    messages = build_protocol_messages(request, PROTOCOL_SINGLE_PLAN_TOOL)
    assert SUBMIT_DRAFT_TOOL_SCHEMA["function"]["parameters"]

    attempt = run_protocol_attempt(
        adapter=adapter,
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=request,
        benchmark=benchmark,
        run_id="lkw-tool-e2e",
    )

    if not attempt.ok:
        pytest.fail(
            "STATUS: BLOCKED "
            f"category={attempt.failure_category} "
            f"phase={attempt.failure_phase} "
            f"safe_error_code={attempt.safe_error_code}"
        )

    assert attempt.failure_category is None
    assert isinstance(attempt.draft, ConversationInteractionDraft)
    assert isinstance(attempt.plan, ConversationInteractionPlan)
