# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider, llm_provider_slug
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response


class _HttpStatusError(RuntimeError):
    status_code: int

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _StubAdapter(BaseLLMAdapter):
    provider: LLMProvider | str = LLMProvider.OPENAI
    model: str = ""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        fail: bool = False,
        status_code: int = 429,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.model = model
        self._fail = fail
        self._status_code = status_code

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(self, messages: object, **kwargs: object) -> LLMAdapterResponse:
        del messages, kwargs
        if self._fail:
            raise _HttpStatusError("rate limited", status_code=self._status_code)
        return LLMAdapterResponse(
            content=f"ok-{self.model}",
            usage=LLMTokenUsage(input_tokens=3, output_tokens=2),
            model=self.model,
            provider=llm_provider_slug(self.provider),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_failover_adapter_uses_secondary_on_retriable_error() -> None:
    primary = _StubAdapter(provider=LLMProvider.OPENAI, model="gpt-4o", fail=True)
    secondary = _StubAdapter(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile")
    adapter = FailoverLLMAdapter(
        [primary, secondary],
        profile_ids=("openai:gpt-4o", "groq:llama-3.3-70b-versatile"),
    )
    response = adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert response.content == "ok-llama-3.3-70b-versatile"
    assert len(adapter.routing_attempts) == 1
    assert adapter.routing_attempts[0].profile_id == "openai:gpt-4o"


def test_failover_adapter_raises_when_all_profiles_fail() -> None:
    primary = _StubAdapter(provider=LLMProvider.OPENAI, model="gpt-4o", fail=True)
    secondary = _StubAdapter(provider=LLMProvider.GROQ, model="backup", fail=True)
    adapter = FailoverLLMAdapter([primary, secondary])
    with pytest.raises(RuntimeError, match="rate limited"):
        adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert len(adapter.routing_attempts) == 2


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_failover_adapter_advances_on_retriable_http_status(status_code: int) -> None:
    primary = _StubAdapter(
        provider=LLMProvider.OPENAI, model="gpt-4o", fail=True, status_code=status_code
    )
    secondary = _StubAdapter(provider=LLMProvider.GROQ, model="backup")
    adapter = FailoverLLMAdapter([primary, secondary])
    response = adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert response.content == "ok-backup"


def test_failover_adapter_stops_on_non_retriable_error() -> None:
    primary = _StubAdapter(
        provider=LLMProvider.OPENAI, model="gpt-4o", fail=True, status_code=400
    )
    secondary = _StubAdapter(provider=LLMProvider.GROQ, model="backup")
    adapter = FailoverLLMAdapter([primary, secondary])
    with pytest.raises(RuntimeError, match="rate limited"):
        adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert len(adapter.routing_attempts) == 1


class _FrameworkFailAdapter(BaseLLMAdapter):
    provider: LLMProvider | str = LLMProvider.OPENAI
    model: str = "gpt-4o"

    def __init__(self, *, fail: bool, status_code: int) -> None:
        super().__init__()
        self._fail = fail
        self._status_code = status_code

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(self, messages, **kwargs):
        del messages, kwargs
        if self._fail:
            raise _HttpStatusError("provider error", status_code=self._status_code)
        return build_adapter_response(content="framework-ok", model=self.model)


def test_failover_adapter_uses_explicit_per_adapter_retry_config_for_eligibility() -> None:
    primary = _FrameworkFailAdapter(fail=True, status_code=418)
    primary_policy = LLMCallConfig(retry_on_status=(418,))
    secondary = _FrameworkFailAdapter(fail=False, status_code=429)
    adapter = FailoverLLMAdapter(
        [primary, secondary],
        failover_retry_config=primary_policy,
        adapter_failover_retry_configs=(primary_policy, LLMCallConfig()),
    )
    response = adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert response.content == "framework-ok"
    assert len(adapter.routing_attempts) == 1
