# © Artur Czarnecki. All rights reserved.
"""
Optional live API smoke tests — not part of PR regression gate.

Run manually or via `.github/workflows/llm-network-smoke.yml`:

  uv run pytest tests/unit/llm_adapters/test_network_smoke.py -m network -q
"""

from __future__ import annotations

import os

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

pytestmark = [pytest.mark.unit, pytest.mark.network]


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_groq_live_one_shot() -> None:
    llm = LLMAdapterRegistry.create(LLMProvider.GROQ)
    text = llm.generate_messages(
        [ChatMessage(role="user", content="Reply with exactly: pong")],
        max_tokens=16,
        run_id="network-groq",
    )
    assert isinstance(text, str) and len(text.strip()) > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_live_one_shot() -> None:
    llm = LLMAdapterRegistry.create(LLMProvider.OPENAI, model="gpt-4o-mini")
    text = llm.generate_messages(
        [ChatMessage(role="user", content="Reply with exactly: pong")],
        max_tokens=16,
        run_id="network-openai",
    )
    assert isinstance(text, str) and len(text.strip()) > 0
