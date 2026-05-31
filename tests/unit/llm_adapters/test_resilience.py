# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.resilience import (
    LLMCircuitOpenError,
    LLMRateLimitError,
    execute_with_resilience,
    reset_provider_resilience,
)
from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit


def test_rate_limit_blocks_excess_calls() -> None:
    reset_provider_resilience("openai")
    cfg = LLMCallConfig(calls_per_minute=2)
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        return "ok"

    execute_with_resilience(fn, provider="openai", config=cfg, retry_fn=lambda f: f())
    execute_with_resilience(fn, provider="openai", config=cfg, retry_fn=lambda f: f())
    with pytest.raises(LLMRateLimitError):
        execute_with_resilience(fn, provider="openai", config=cfg, retry_fn=lambda f: f())
    reset_provider_resilience("openai")


def test_circuit_opens_after_failures() -> None:
    reset_provider_resilience("openai")
    cfg = LLMCallConfig(circuit_breaker_threshold=2, circuit_breaker_cooldown_sec=60)

    def boom() -> str:
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        execute_with_resilience(boom, provider="openai", config=cfg, retry_fn=lambda f: f())
    with pytest.raises(RuntimeError):
        execute_with_resilience(boom, provider="openai", config=cfg, retry_fn=lambda f: f())
    with pytest.raises(LLMCircuitOpenError):
        execute_with_resilience(boom, provider="openai", config=cfg, retry_fn=lambda f: f())
    reset_provider_resilience("openai")


def test_adapter_execute_uses_resilience() -> None:
    client = MagicMock()
    usage = MagicMock(input_tokens=1, output_tokens=1)
    response = MagicMock(usage=usage, output_text="x", output=[], status="completed")
    client.responses.create.return_value = response

    adapter = OpenAIChatResponsesAdapter(
        client=client,
        model="gpt-4o-mini",
        calls_per_minute=100,
    )
    from intergrax.llm.messages import ChatMessage

    text = adapter.generate_messages([ChatMessage(role="user", content="hi")], run_id="r1")
    assert text == "x"
