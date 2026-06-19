# © Artur Czarnecki. All rights reserved.
"""
Optional live API smoke tests — not part of PR regression gate.

  uv run pytest tests/unit/llm_adapters/test_network_smoke.py -m network -q
"""

from __future__ import annotations

import os

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

pytestmark = [pytest.mark.unit, pytest.mark.network]


def _one_shot(llm_provider: LLMProvider, *, model: str | None = None, **kwargs) -> None:
    create_kwargs = dict(kwargs)
    if model:
        create_kwargs["model"] = model
    llm = LLMAdapterRegistry.create(llm_provider, **create_kwargs)
    response = llm.generate_messages(
        [ChatMessage(role="user", content="Reply with exactly: pong")],
        max_tokens=16,
        run_id=f"network-{llm_provider.value}",
    )
    assert isinstance(response.content, str) and len(response.content.strip()) > 0


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_groq_live_one_shot() -> None:
    _one_shot(LLMProvider.GROQ)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_live_one_shot() -> None:
    _one_shot(LLMProvider.OPENAI, model="gpt-4o-mini")


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_live_one_shot() -> None:
    _one_shot(LLMProvider.CLAUDE)


@pytest.mark.skipif(
    not os.getenv("INTERGRAX_DEFAULT_AWS_REGION") or not os.getenv("INTERGRAX_DEFAULT_BEDROCK_MODEL_ID"),
    reason="Bedrock env not set",
)
def test_bedrock_live_one_shot() -> None:
    _one_shot(LLMProvider.AWS_BEDROCK, use_converse=True)


@pytest.mark.skipif(
    not os.getenv("INTERGRAX_VERTEX_PROJECT"),
    reason="INTERGRAX_VERTEX_PROJECT not set",
)
def test_vertex_gemini_live_one_shot() -> None:
    _one_shot(LLMProvider.VERTEX_GEMINI)


def test_vllm_live_one_shot() -> None:
    from testing_support.builder import require_vllm_reachable

    require_vllm_reachable()
    model = os.getenv("INTERGRAX_DEFAULT_VLLM_MODEL", "").strip() or None
    _one_shot(LLMProvider.VLLM, model=model)


def test_llama_cpp_live_one_shot() -> None:
    from testing_support.builder import require_llama_cpp_reachable

    require_llama_cpp_reachable()
    model = os.getenv("INTERGRAX_DEFAULT_LLAMA_CPP_MODEL", "").strip() or None
    _one_shot(LLMProvider.LLAMA_CPP, model=model)
