# © Artur Czarnecki. All rights reserved.

"""Gated live probe for Ollama canonical ConversationInteractionPlan schema."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "applications"))
from local_workspace_application.conversation.interaction_models import (  # noqa: E402
    ConversationInteractionPlan,
    WorkspaceActivatePlannedAction,
)
from local_workspace_application.conversation.interaction_prompt import (  # noqa: E402
    build_planning_messages,
)
from local_workspace_application.conversation.interaction_models import (  # noqa: E402
    ConversationPlanningRequest,
    ConversationPlanningWorkspace,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_E2E_FLAG = "INTERGRAX_OLLAMA_SCHEMA_COMPAT_E2E"


def _e2e_enabled() -> bool:
    return os.environ.get(_E2E_FLAG, "").strip() == "1"


def _require_model() -> str:
    model = os.environ.get("INTERGRAX_LLM_MODEL", "").strip()
    if not model:
        pytest.fail(f"INTERGRAX_LLM_MODEL is required when {_E2E_FLAG}=1")
    return model


@pytest.fixture(scope="module")
def live_adapter() -> LangChainOllamaAdapter:
    if not _e2e_enabled():
        pytest.skip(f"{_E2E_FLAG} is not set")
    return LangChainOllamaAdapter(model=_require_model())


def test_live_ollama_canonical_plan_schema_compilation(live_adapter: LangChainOllamaAdapter) -> None:
    request = ConversationPlanningRequest(
        message_text="Switch to workspace magazyn.",
        available_workspaces=(
            ConversationPlanningWorkspace(
                workspace_id="ws-1",
                name="magazyn",
                is_active=False,
            ),
        ),
    )
    messages = build_planning_messages(request)

    result = live_adapter.generate_structured(
        messages,
        ConversationInteractionPlan,
        temperature=0,
        max_tokens=8192,
        run_id="ollama-plan-schema-compat",
    )

    assert isinstance(result.parsed, ConversationInteractionPlan)
    assert result.parsed.plan_version == "2"
    assert result.parsed.response_mode == "aggregate"
    assert any(
        isinstance(action, WorkspaceActivatePlannedAction) for action in result.parsed.actions
    )
    assert result.response.provider == LLMProvider.OLLAMA.value
    assert result.response.model == live_adapter.model
