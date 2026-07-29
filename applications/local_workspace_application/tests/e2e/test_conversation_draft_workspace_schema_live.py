# © Artur Czarnecki. All rights reserved.

"""Gated live probe for ConversationInteractionDraft workspace-reference schema."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.llm.messages import ChatMessage

from local_workspace_application.conversation.interaction_draft_models import (
    ActiveDraftWorkspaceReference,
    ConversationInteractionDraft,
    WorkspaceAskDraftAction,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_E2E_FLAG = "INTERGRAX_LKW_DRAFT_WORKSPACE_SCHEMA_E2E"
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _e2e_enabled() -> bool:
    return os.environ.get(_E2E_FLAG, "").strip() == "1"


def _load_lkw_env() -> None:
    if not _ENV_FILE.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ENV_FILE, override=False)


def _require_e2e_config() -> tuple[str, str]:
    _load_lkw_env()
    provider = os.environ.get("INTERGRAX_LLM_PROVIDER", "").strip()
    model = os.environ.get("INTERGRAX_LLM_MODEL", "").strip()
    if not provider:
        pytest.fail(f"INTERGRAX_LLM_PROVIDER is required when {_E2E_FLAG}=1")
    if not model:
        pytest.fail(f"INTERGRAX_LLM_MODEL is required when {_E2E_FLAG}=1")
    return provider, model


@pytest.fixture(scope="module")
def live_adapter():
    if not _e2e_enabled():
        pytest.skip(f"{_E2E_FLAG} is not set")
    provider, model = _require_e2e_config()
    adapter = resolve_llm_adapter(None)
    if not adapter.supports_structured_output():
        pytest.fail(
            f"configured adapter does not support structured output "
            f"(provider={provider}, model={model})"
        )
    return adapter


def test_live_ollama_draft_workspace_schema_compilation(live_adapter) -> None:
    messages = [
        ChatMessage(
            role="user",
            content=(
                "Return a semantic interaction draft with exactly one action: "
                "workspace.ask using the active workspace reference and question "
                "'what is in the active workspace?'"
            ),
        ),
    ]

    result = live_adapter.generate_structured(
        messages,
        ConversationInteractionDraft,
        temperature=0,
        max_tokens=4096,
        run_id="ollama-draft-workspace-schema",
    )

    assert isinstance(result.parsed, ConversationInteractionDraft)
    assert result.parsed.actions
    action = result.parsed.actions[0]
    assert isinstance(action, WorkspaceAskDraftAction)
    assert isinstance(action.workspace, ActiveDraftWorkspaceReference)
    assert action.workspace.kind.value == "active"
    assert action.workspace.value is None
