# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

import pytest

from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

from local_workspace_application.benchmarks.local_model_qualification.config import load_config
from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    StructuralFailureCategory,
)
from local_workspace_application.benchmarks.local_model_qualification.corpus import case_by_id
from local_workspace_application.benchmarks.local_model_qualification.protocols import (
    PROTOCOL_SINGLE_PLAN_TOOL,
    SUBMIT_DRAFT_TOOL_SCHEMA,
    build_protocol_messages,
    is_expected_schema_incompatibility,
    run_protocol_attempt,
)
from local_workspace_application.benchmarks.local_model_qualification.provisioning import (
    provision_ollama_runtime,
)
from local_workspace_application.conversation.interaction_draft_models import ConversationInteractionDraft
from local_workspace_application.conversation.interaction_models import ConversationInteractionPlan

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_E2E_FAILURE_CATEGORIES = frozenset(
    {
        StructuralFailureCategory.PROVIDER_ERROR.value,
        StructuralFailureCategory.RESOURCE_LIMIT.value,
        StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value,
    }
)


def _build_e2e_config(model_name: str):
    config = load_config()
    matching = [model for model in config.models if model.name == model_name]
    if len(matching) != 1:
        pytest.fail(
            f"Expected exactly one configured model named {model_name!r}, found {len(matching)}"
        )
    models = tuple(
        model.model_copy(update={"enabled": model.name == model_name})
        for model in config.models
    )
    return config.model_copy(update={"models": models})


def _assert_no_raw_provider_message(attempt) -> None:
    if attempt.safe_error_code is not None:
        pytest.fail(
            "unsafe raw provider error serialization: "
            f"safe_error_code={attempt.safe_error_code}"
        )


def test_full_schema_tool_protocol_live_classifies_model_compatibility() -> None:
    if os.environ.get("INTERGRAX_LKW_MODEL_QUALIFICATION_TOOL_E2E") != "1":
        pytest.skip("Set INTERGRAX_LKW_MODEL_QUALIFICATION_TOOL_E2E=1 to run live tool proof")

    model_name = os.environ.get("INTERGRAX_LLM_MODEL", "qwen2.5:14b")
    e2e_config = _build_e2e_config(model_name)

    provision_ollama_runtime(
        e2e_config,
        progress=lambda message: print(message, flush=True),
    )

    adapter = LLMAdapterRegistry.create(
        LLMProvider.OLLAMA,
        model=model_name,
        base_url=e2e_config.ollama.host,
        keep_alive=e2e_config.ollama.keep_alive,
    )
    tools_declared = adapter.supports_tools()

    request = case_by_id("planner.workspace_list").request
    messages = build_protocol_messages(request, PROTOCOL_SINGLE_PLAN_TOOL)
    assert SUBMIT_DRAFT_TOOL_SCHEMA["function"]["parameters"]

    attempt = run_protocol_attempt(
        adapter=adapter,
        protocol=PROTOCOL_SINGLE_PLAN_TOOL,
        request=request,
        benchmark=e2e_config.benchmark,
        run_id="lkw-tool-e2e",
    )

    if attempt.ok:
        assert attempt.failure_category is None
        assert attempt.failure_phase is None
        assert isinstance(attempt.draft, ConversationInteractionDraft)
        assert isinstance(attempt.plan, ConversationInteractionPlan)
        print("tool_schema_compatibility=PASS", flush=True)
        return

    if attempt.failure_category is None or not attempt.failure_phase or not attempt.failure_phase.strip():
        pytest.fail(
            "missing failure classification: "
            f"category={attempt.failure_category} phase={attempt.failure_phase}"
        )

    if attempt.failure_category in _E2E_FAILURE_CATEGORIES:
        if (
            attempt.failure_category == StructuralFailureCategory.PROTOCOL_UNSUPPORTED.value
            and tools_declared
        ):
            pytest.fail(
                "capability-resolution defect after successful provisioning: "
                f"category={attempt.failure_category} phase={attempt.failure_phase}"
            )
        pytest.fail(
            "provider or capability failure: "
            f"category={attempt.failure_category} "
            f"phase={attempt.failure_phase} "
            f"safe_error_code={attempt.safe_error_code}"
        )

    if not is_expected_schema_incompatibility(attempt.failure_category):
        pytest.fail(
            "unexpected failure category: "
            f"category={attempt.failure_category} "
            f"phase={attempt.failure_phase} "
            f"safe_error_code={attempt.safe_error_code}"
        )

    _assert_no_raw_provider_message(attempt)
    print("tool_schema_compatibility=INCOMPATIBLE", flush=True)
    print(f"tool_schema_failure_category={attempt.failure_category}", flush=True)
    print(f"tool_schema_failure_phase={attempt.failure_phase}", flush=True)
    print(f"tool_schema_safe_error_code={attempt.safe_error_code or 'none'}", flush=True)
