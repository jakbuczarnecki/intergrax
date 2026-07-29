# © Artur Czarnecki. All rights reserved.

"""Gated live probe for Ollama native tool calling."""

from __future__ import annotations

import json
import os

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter

pytestmark = [
    pytest.mark.unit,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_E2E_FLAG = "INTERGRAX_OLLAMA_TOOL_CALLING_E2E"

_RECORD_WORKSPACE_NUMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "record_workspace_number",
        "description": (
            "Record the positive workspace number explicitly requested by the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_number": {
                    "type": "string",
                    "pattern": "^[1-9][0-9]*$",
                }
            },
            "required": ["workspace_number"],
            "additionalProperties": False,
        },
    },
}


def _e2e_enabled() -> bool:
    return os.environ.get(_E2E_FLAG, "").strip() == "1"


def _require_model() -> str:
    model = os.environ.get("INTERGRAX_LLM_MODEL", "").strip()
    if not model:
        pytest.fail(f"INTERGRAX_LLM_MODEL is required when {_E2E_FLAG}=1")
    return model


def _require_ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        pytest.fail(f"OLLAMA_HOST is required when {_E2E_FLAG}=1")
    return host


@pytest.fixture(scope="module")
def live_adapter() -> LangChainOllamaAdapter:
    if not _e2e_enabled():
        pytest.skip(f"{_E2E_FLAG} is not set")
    _require_ollama_host()
    model = _require_model()
    adapter = LangChainOllamaAdapter(model=model)
    if not adapter.supports_tools():
        pytest.fail("LangChainOllamaAdapter must support native tool calling")
    return adapter


def test_live_ollama_native_tool_calling(live_adapter: LangChainOllamaAdapter) -> None:
    messages = [
        ChatMessage(
            role="user",
            content=(
                "Call record_workspace_number with workspace_number 2. "
                "Do not answer in plain text."
            ),
        )
    ]

    result = live_adapter.generate_with_tools(
        messages,
        [_RECORD_WORKSPACE_NUMBER_TOOL],
        temperature=0,
        tool_choice="auto",
        run_id="ollama-tool-calling-probe",
    )

    assert isinstance(result, LLMAdapterResponse)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "record_workspace_number"
    args = json.loads(result.tool_calls[0].arguments_json)
    assert args["workspace_number"] == "2"
    assert result.finish_reason == LLMFinishReason.TOOL_CALLS
    assert result.provider == LLMProvider.OLLAMA.value
    assert result.model == live_adapter.model
