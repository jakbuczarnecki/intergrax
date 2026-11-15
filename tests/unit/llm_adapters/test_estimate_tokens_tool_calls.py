# © Artur Czarnecki. All rights reserved.

"""The adapter token estimator counts assistant ``tool_calls``, ``name`` and ``tool_call_id``.

Estimator half of PDC-D2 finding F-D2-01 (the adapter estimator is owned by D3): a message with empty
content and a very large ``tool_calls[].function.arguments`` used to estimate as ~0 tokens. Messages
without those fields must estimate exactly as before.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse

pytestmark = pytest.mark.unit


class _Adapter(BaseLLMAdapter):
    provider = "fake"
    model = "fake-model"

    @property
    def context_window_tokens(self) -> int:
        return 8_192

    def generate_messages(self, messages: Sequence[ChatMessage], **_: Any) -> LLMAdapterResponse:
        return LLMAdapterResponse(content="x", provider=self.provider, model=self.model)


def _call(arguments: str, *, call_id: str = "call_1", name: str = "lookup") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture()
def adapter() -> _Adapter:
    return _Adapter()


def test_messages_without_tool_fields_estimate_exactly_as_before(adapter: _Adapter) -> None:
    messages = [
        ChatMessage(role="system", content="You are terse."),
        ChatMessage(role="user", content="What is the status of order 42?"),
        ChatMessage(role="assistant", content="Order 42 shipped yesterday."),
    ]
    expected = adapter.estimate_tokens_for_text("\n".join(m.content for m in messages))
    assert adapter.estimate_tokens_for_messages(messages) == expected
    assert adapter.estimate_tokens_for_messages(messages, model_hint="gpt-4o-mini") == (
        adapter.estimate_tokens_for_text("\n".join(m.content for m in messages), model_hint="gpt-4o-mini")
    )


def test_empty_messages_and_empty_content_still_estimate_zero(adapter: _Adapter) -> None:
    assert adapter.estimate_tokens_for_messages([]) == 0
    assert adapter.estimate_tokens_for_messages([ChatMessage(role="assistant", content="")]) == 0
    assert adapter.estimate_tokens_for_messages([ChatMessage(role="assistant", content="", tool_calls=[])]) == 0


def test_large_tool_call_arguments_are_budgeted(adapter: _Adapter) -> None:
    bare = ChatMessage(role="assistant", content="")
    heavy = ChatMessage(role="assistant", content="", tool_calls=[_call("a" * 200_000)])
    assert adapter.estimate_tokens_for_messages([bare, heavy]) > 10_000


def test_tool_call_identity_and_function_name_are_counted(adapter: _Adapter) -> None:
    short = ChatMessage(role="assistant", content="", tool_calls=[_call("{}", name="f", call_id="c")])
    long_name = ChatMessage(role="assistant", content="", tool_calls=[_call("{}", name="n" * 4_000, call_id="c")])
    long_id = ChatMessage(role="assistant", content="", tool_calls=[_call("{}", name="f", call_id="i" * 4_000)])
    base = adapter.estimate_tokens_for_messages([short])
    assert adapter.estimate_tokens_for_messages([long_name]) > base + 100
    assert adapter.estimate_tokens_for_messages([long_id]) > base + 100


def test_tool_message_name_and_tool_call_id_are_counted(adapter: _Adapter) -> None:
    plain = ChatMessage(role="tool", content="ok")
    tagged = ChatMessage(role="tool", content="ok", name="n" * 4_000, tool_call_id="i" * 4_000)
    assert adapter.estimate_tokens_for_messages([tagged]) > adapter.estimate_tokens_for_messages([plain]) + 100


def test_estimate_is_deterministic_and_independent_of_dict_key_order(adapter: _Adapter) -> None:
    one = ChatMessage(role="assistant", content="", tool_calls=[{"id": "c", "type": "function", "x": 1, "y": 2}])
    two = ChatMessage(role="assistant", content="", tool_calls=[{"y": 2, "x": 1, "type": "function", "id": "c"}])
    assert adapter.estimate_tokens_for_messages([one]) == adapter.estimate_tokens_for_messages([two])
    assert adapter.estimate_tokens_for_messages([one]) == adapter.estimate_tokens_for_messages([one])


def test_a_tool_call_payload_that_is_not_json_serializable_does_not_raise(adapter: _Adapter) -> None:
    odd = ChatMessage(role="assistant", content="", tool_calls=[{"id": "c", "arguments": object(), "n": {1, 2}}])
    assert adapter.estimate_tokens_for_messages([odd]) > 0


def test_count_messages_tokens_uses_the_same_estimate(adapter: _Adapter) -> None:
    heavy = ChatMessage(role="assistant", content="", tool_calls=[_call("a" * 50_000)])
    assert adapter.count_messages_tokens([heavy]) > 1_000


def test_unsortable_or_circular_tool_call_payloads_fall_back_to_their_text(adapter: _Adapter) -> None:
    mixed_keys = ChatMessage(role="assistant", content="", tool_calls=[{1: "a", "b": "c"}])
    loop: dict[str, Any] = {"id": "c"}
    loop["self"] = loop
    circular = ChatMessage(role="assistant", content="", tool_calls=[loop])
    assert adapter.estimate_tokens_for_messages([mixed_keys]) > 0
    assert adapter.estimate_tokens_for_messages([circular]) > 0
