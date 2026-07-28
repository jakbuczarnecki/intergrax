# © Artur Czarnecki. All rights reserved.

"""Gated live probe for Ollama native JSON Schema structured output."""

from __future__ import annotations

import os
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter

pytestmark = [
    pytest.mark.unit,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_E2E_FLAG = "INTERGRAX_OLLAMA_STRUCTURED_E2E"


class OllamaStructuredProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    count: int


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
    model = _require_model()
    adapter = LangChainOllamaAdapter(model=model)
    if not adapter.supports_structured_output():
        pytest.fail("LangChainOllamaAdapter must support structured output")
    return adapter


def test_live_ollama_native_structured_output(live_adapter: LangChainOllamaAdapter) -> None:
    messages = [ChatMessage(role="user", content="Return status ok and count 2.")]

    result = live_adapter.generate_structured(
        messages,
        OllamaStructuredProbe,
        temperature=0,
        max_tokens=256,
        run_id="ollama-structured-probe",
    )

    assert result.parsed.status == "ok"
    assert result.parsed.count == 2
    assert result.response.content
    assert result.response.provider == LLMProvider.OLLAMA.value
    assert result.response.model == live_adapter.model
