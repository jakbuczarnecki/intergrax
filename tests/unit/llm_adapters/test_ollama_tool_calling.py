# © Artur Czarnecki. All rights reserved.

"""Deterministic tests for Ollama native tool calling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter

pytestmark = pytest.mark.unit

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup a value",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


@pytest.fixture()
def fake_chat() -> MagicMock:
    chat = MagicMock()
    chat.model = "qwen2.5:14b"
    return chat


@pytest.fixture()
def bound_chat(fake_chat: MagicMock) -> MagicMock:
    bound = MagicMock()
    fake_chat.bind_tools.return_value = bound
    return bound


@pytest.fixture()
def adapter(fake_chat: MagicMock) -> LangChainOllamaAdapter:
    return LangChainOllamaAdapter(chat=fake_chat, model="qwen2.5:14b")


def _tool_call_result(
  bound_chat: MagicMock,
  *,
  content: str = "",
  tool_calls: list[dict] | None = None,
  invalid_tool_calls: list | None = None,
) -> None:
    bound_chat.invoke.return_value = MagicMock(
        content=content,
        tool_calls=tool_calls or [],
        invalid_tool_calls=invalid_tool_calls or [],
    )


def test_supports_tools_and_structured_output(adapter: LangChainOllamaAdapter) -> None:
    assert adapter.supports_tools() is True
    assert adapter.supports_structured_output() is True


def test_native_binding_receives_schema_and_options(
    adapter: LangChainOllamaAdapter,
    fake_chat: MagicMock,
    bound_chat: MagicMock,
) -> None:
    _tool_call_result(bound_chat, content="done")
    messages = [ChatMessage(role="user", content="hi")]

    adapter.generate_with_tools(
        messages,
        TOOLS_SCHEMA,
        temperature=0.1,
        max_tokens=64,
        run_id="tool-bind",
    )

    fake_chat.bind_tools.assert_called_once_with(TOOLS_SCHEMA)
    bound_chat.invoke.assert_called_once()
    fake_chat.invoke.assert_not_called()
    _, kwargs = bound_chat.invoke.call_args
    assert kwargs["options"]["temperature"] == 0.1
    assert kwargs["options"]["num_predict"] == 64


def test_typed_tool_call_result(
    adapter: LangChainOllamaAdapter,
    bound_chat: MagicMock,
) -> None:
    _tool_call_result(
        bound_chat,
        tool_calls=[
            {
                "name": "lookup",
                "args": {"query": "wartosc"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    result = adapter.generate_with_tools(
        [ChatMessage(role="user", content="search")],
        TOOLS_SCHEMA,
    )

    assert isinstance(result, LLMAdapterResponse)
    assert isinstance(result.tool_calls, tuple)
    assert len(result.tool_calls) == 1
    assert isinstance(result.tool_calls[0], LLMToolCall)
    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].name == "lookup"
    assert json.loads(result.tool_calls[0].arguments_json) == {"query": "wartosc"}
    assert result.finish_reason == LLMFinishReason.TOOL_CALLS


def test_unicode_arguments_preserved(
    adapter: LangChainOllamaAdapter,
    bound_chat: MagicMock,
) -> None:
    _tool_call_result(
        bound_chat,
        tool_calls=[
            {
                "name": "lookup",
                "args": {"query": "żółć"},
                "id": "call-unicode",
                "type": "tool_call",
            }
        ],
    )
    result = adapter.generate_with_tools(
        [ChatMessage(role="user", content="search")],
        TOOLS_SCHEMA,
    )
    assert json.loads(result.tool_calls[0].arguments_json) == {"query": "żółć"}


def test_multiple_tool_calls_preserve_order(
    adapter: LangChainOllamaAdapter,
    bound_chat: MagicMock,
) -> None:
    _tool_call_result(
        bound_chat,
        tool_calls=[
            {"name": "first", "args": {"n": 1}, "id": "call-1", "type": "tool_call"},
            {"name": "second", "args": {"n": 2}, "id": "call-2", "type": "tool_call"},
        ],
    )
    result = adapter.generate_with_tools(
        [ChatMessage(role="user", content="multi")],
        TOOLS_SCHEMA,
    )
    assert [tc.id for tc in result.tool_calls] == ["call-1", "call-2"]
    assert [tc.name for tc in result.tool_calls] == ["first", "second"]


def test_no_tool_call_returns_completed(
    adapter: LangChainOllamaAdapter,
    bound_chat: MagicMock,
) -> None:
    _tool_call_result(bound_chat, content="plain answer")
    result = adapter.generate_with_tools(
        [ChatMessage(role="user", content="hello")],
        TOOLS_SCHEMA,
    )
    assert result.tool_calls == ()
    assert result.finish_reason == LLMFinishReason.COMPLETED
    assert result.content == "plain answer"


def test_invalid_tool_calls_raise_stable_error(
    adapter: LangChainOllamaAdapter,
    bound_chat: MagicMock,
) -> None:
    _tool_call_result(
        bound_chat,
        invalid_tool_calls=[{"name": "lookup", "args": "not-json", "id": "bad"}],
    )
    with pytest.raises(ValueError, match="Ollama returned invalid native tool calls"):
        adapter.generate_with_tools(
            [ChatMessage(role="user", content="search")],
            TOOLS_SCHEMA,
            run_id="invalid-tool-calls",
        )
    stats = adapter.usage.get_run_stats("invalid-tool-calls")
    assert stats.calls >= 1
    assert stats.errors >= 1


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        "none",
        {"type": "function", "function": {"name": "lookup"}},
        "lookup",
    ],
)
def test_unsupported_tool_choice_rejects_before_provider(
    adapter: LangChainOllamaAdapter,
    fake_chat: MagicMock,
    bound_chat: MagicMock,
    tool_choice: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Ollama native tool calling supports only tool_choice=None or 'auto'",
    ):
        adapter.generate_with_tools(
            [ChatMessage(role="user", content="search")],
            TOOLS_SCHEMA,
            tool_choice=tool_choice,  # type: ignore[arg-type]
        )
    fake_chat.bind_tools.assert_not_called()
    bound_chat.invoke.assert_not_called()


@pytest.mark.parametrize("tool_choice", [None, "auto"])
def test_accepted_tool_choice_reaches_native_binding(
    adapter: LangChainOllamaAdapter,
    fake_chat: MagicMock,
    bound_chat: MagicMock,
    tool_choice: str | None,
) -> None:
    _tool_call_result(bound_chat, content="ok")
    adapter.generate_with_tools(
        [ChatMessage(role="user", content="search")],
        TOOLS_SCHEMA,
        tool_choice=tool_choice,
    )
    fake_chat.bind_tools.assert_called_once_with(TOOLS_SCHEMA)
    bound_chat.invoke.assert_called_once()


def test_assistant_openai_style_tool_calls_convert_to_ai_message(
    adapter: LangChainOllamaAdapter,
) -> None:
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"query":"value"}',
                },
            }
        ],
    )
    lc_messages = adapter._to_lc_messages([message])
    assert len(lc_messages) == 1
    assert isinstance(lc_messages[0], AIMessage)
    assert lc_messages[0].tool_calls == [
        {
            "name": "lookup",
            "args": {"query": "value"},
            "id": "call-1",
            "type": "tool_call",
        }
    ]


def test_assistant_langchain_style_tool_calls_convert(
    adapter: LangChainOllamaAdapter,
) -> None:
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call-2",
                "name": "lookup",
                "args": {"query": "value"},
                "type": "tool_call",
            }
        ],
    )
    lc_messages = adapter._to_lc_messages([message])
    assert isinstance(lc_messages[0], AIMessage)
    assert lc_messages[0].tool_calls[0]["name"] == "lookup"
    assert lc_messages[0].tool_calls[0]["args"] == {"query": "value"}


def test_tool_result_converts_to_tool_message(adapter: LangChainOllamaAdapter) -> None:
    message = ChatMessage(role="tool", content="result", tool_call_id="call-1")
    lc_messages = adapter._to_lc_messages([message])
    assert len(lc_messages) == 1
    assert isinstance(lc_messages[0], ToolMessage)
    assert lc_messages[0].content == "result"
    assert lc_messages[0].tool_call_id == "call-1"


def test_missing_tool_call_id_raises(adapter: LangChainOllamaAdapter) -> None:
    message = ChatMessage(role="tool", content="result")
    with pytest.raises(ValueError, match="tool message requires tool_call_id"):
        adapter._to_lc_messages([message])


def test_malformed_internal_arguments_rejected(adapter: LangChainOllamaAdapter) -> None:
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{not-json",
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="tool call arguments must be valid JSON"):
        adapter._to_lc_messages([message])


def test_ordinary_message_conversion_unchanged(adapter: LangChainOllamaAdapter) -> None:
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="user"),
        ChatMessage(role="assistant", content="assistant"),
    ]
    lc_messages = adapter._to_lc_messages(messages)
    assert isinstance(lc_messages[0], SystemMessage)
    assert isinstance(lc_messages[1], HumanMessage)
    assert isinstance(lc_messages[2], AIMessage)
    assert lc_messages[2].tool_calls == []


def test_usage_tracking_registers_run_id(
    adapter: LangChainOllamaAdapter,
    bound_chat: MagicMock,
) -> None:
    _tool_call_result(
        bound_chat,
        tool_calls=[
            {
                "name": "lookup",
                "args": {"query": "x" * 200},
                "id": "call-usage",
                "type": "tool_call",
            }
        ],
    )
    with patch.object(
        adapter,
        "estimate_tokens_for_text",
        wraps=adapter.estimate_tokens_for_text,
    ) as estimate_text:
        adapter.generate_with_tools(
            [ChatMessage(role="user", content="search")],
            TOOLS_SCHEMA,
            run_id="tool-usage-run",
        )
        estimate_input = "".join(
            call.args[0] for call in estimate_text.call_args_list
        )
        assert "lookup" in estimate_input
        assert '"query"' in estimate_input

    stats = adapter.usage.get_run_stats("tool-usage-run")
    assert stats.calls == 1
    assert stats.errors == 0
