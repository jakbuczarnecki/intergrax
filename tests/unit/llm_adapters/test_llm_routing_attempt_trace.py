# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.registry.failover_adapter import (
    FailoverLLMAdapter,
    LLMRoutingAttemptRecord,
)
from intergrax.runtime.nexus.tracing.adapters.llm_routing_attempt import (
    LLMRoutingAttemptDiagV1,
    attach_failover_routing_trace_observer,
    routing_attempt_to_diag,
)


class _StubAdapter:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        fail: bool = False,
        status_code: int = 429,
    ) -> None:
        self.provider = provider
        self.model = model
        self.model_name_for_token_estimation = model
        self.call_config = MagicMock()
        self.call_config.retry_on_status = (429, 500, 502, 503, 504)
        self._fail = fail
        self._status_code = status_code
        self.context_window_tokens = 128_000

    def count_messages_tokens(self, messages: object) -> int:
        del messages
        return 1

    def generate_messages(self, messages: object, **kwargs: object) -> LLMAdapterResponse:
        del messages, kwargs
        if self._fail:
            exc = RuntimeError("rate limited")
            exc.status_code = self._status_code  # type: ignore[attr-defined]
            raise exc
        return LLMAdapterResponse(
            content=f"ok-{self.model}",
            usage=LLMTokenUsage(input_tokens=3, output_tokens=2),
            model=self.model,
            provider=self.provider.value,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_routing_attempt_to_diag_maps_profile_id() -> None:
    record = LLMRoutingAttemptRecord(
        profile_index=0,
        profile_id="openai:gpt-4o",
        provider="openai",
        model="gpt-4o",
        error="RuntimeError: rate limited",
    )
    diag = routing_attempt_to_diag(record)
    assert diag.profile_id == "openai:gpt-4o"
    assert diag.schema_id() == "intergrax.diag.engine.core_llm.routing_attempt"


@pytest.mark.unit
@pytest.mark.gate
def test_attach_failover_routing_trace_observer_emits_per_attempt() -> None:
    primary = _StubAdapter(provider=LLMProvider.OPENAI, model="gpt-4o", fail=True)
    secondary = _StubAdapter(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile")
    adapter = FailoverLLMAdapter(
        [primary, secondary],
        profile_ids=("openai:gpt-4o", "groq:llama-3.3-70b-versatile"),
    )
    emitted: list[LLMRoutingAttemptDiagV1] = []

    def _trace_event(**kwargs: object) -> None:
        payload = kwargs.get("payload")
        assert isinstance(payload, LLMRoutingAttemptDiagV1)
        emitted.append(payload)

    attach_failover_routing_trace_observer(adapter, _trace_event)
    adapter.generate_messages([ChatMessage(role="user", content="hi")])

    assert len(emitted) == 1
    assert emitted[0].profile_id == "openai:gpt-4o"
    assert emitted[0].provider == "openai"


@pytest.mark.unit
@pytest.mark.gate
def test_failover_adapter_clears_attempts_between_calls() -> None:
    primary = _StubAdapter(provider=LLMProvider.OPENAI, model="gpt-4o", fail=True)
    secondary = _StubAdapter(provider=LLMProvider.GROQ, model="backup")
    adapter = FailoverLLMAdapter([primary, secondary], profile_ids=("openai:gpt-4o", "groq:backup"))
    adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert len(adapter.routing_attempts) == 1
    primary._fail = False
    adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert len(adapter.routing_attempts) == 0
