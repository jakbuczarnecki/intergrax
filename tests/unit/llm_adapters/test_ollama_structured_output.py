# © Artur Czarnecki. All rights reserved.

"""Deterministic tests for Ollama native JSON Schema structured output."""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter

pytestmark = pytest.mark.unit


class SampleStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    count: int


@pytest.fixture()
def fake_chat() -> MagicMock:
    chat = MagicMock()
    chat.model = "qwen2.5:14b"
    return chat


@pytest.fixture()
def adapter(fake_chat: MagicMock) -> LangChainOllamaAdapter:
    return LangChainOllamaAdapter(chat=fake_chat, model="qwen2.5:14b")


@pytest.fixture()
def messages() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="Return status ok and count 2.")]


@pytest.fixture()
def structured_runnable(fake_chat: MagicMock) -> MagicMock:
    runnable = MagicMock()
    fake_chat.with_structured_output.return_value = runnable
    return runnable


def test_native_json_schema_method_selected(
    adapter: LangChainOllamaAdapter,
    fake_chat: MagicMock,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    parsed = SampleStructuredOutput(status="ok", count=2)
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content='{"status":"ok","count":2}'),
        "parsed": parsed,
        "parsing_error": None,
    }

    adapter.generate_structured(messages, SampleStructuredOutput, temperature=0, max_tokens=128)

    fake_chat.with_structured_output.assert_called_once()
    args, kwargs = fake_chat.with_structured_output.call_args
    assert kwargs["method"] == "json_schema"
    assert kwargs["include_raw"] is True
    generation_schema = args[0]
    assert isinstance(generation_schema, dict)
    assert generation_schema["type"] == "object"
    assert "status" in generation_schema["properties"]


def test_original_messages_only_no_synthetic_schema(
    adapter: LangChainOllamaAdapter,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    parsed = SampleStructuredOutput(status="ok", count=2)
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content='{"status":"ok","count":2}'),
        "parsed": parsed,
        "parsing_error": None,
    }

    adapter.generate_structured(messages, SampleStructuredOutput)

    invoke_args = structured_runnable.invoke.call_args
    lc_msgs = invoke_args[0][0]
    assert len(lc_msgs) == 1
    assert isinstance(lc_msgs[0], HumanMessage)
    assert lc_msgs[0].content == messages[0].content
    for msg in lc_msgs:
        content = getattr(msg, "content", "")
        assert "JSON_SCHEMA" not in content
        assert "Return ONLY a single JSON object" not in content
        assert not (isinstance(msg, SystemMessage) and "JSON" in content)


def test_successful_pydantic_result(
    adapter: LangChainOllamaAdapter,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    raw_content = '{"status":"ok","count":2}'
    parsed = SampleStructuredOutput(status="ok", count=2)
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content=raw_content),
        "parsed": parsed,
        "parsing_error": None,
    }

    result = adapter.generate_structured(messages, SampleStructuredOutput, run_id="run-1")

    assert isinstance(result.parsed, SampleStructuredOutput)
    assert result.parsed.status == "ok"
    assert result.parsed.count == 2
    assert result.response.content == raw_content
    assert result.response.provider == LLMProvider.OLLAMA.value
    assert result.response.model == "qwen2.5:14b"


def test_dictionary_is_revalidated(
    adapter: LangChainOllamaAdapter,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content='{"status":"ok","count":2}'),
        "parsed": {"status": "ok", "count": 2},
        "parsing_error": None,
    }

    result = adapter.generate_structured(messages, SampleStructuredOutput)

    assert isinstance(result.parsed, SampleStructuredOutput)
    assert result.parsed.status == "ok"
    assert result.parsed.count == 2


def test_invalid_parsed_dictionary_fails_closed(
    adapter: LangChainOllamaAdapter,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content='{"status":"ok"}'),
        "parsed": {"status": "ok"},
        "parsing_error": None,
    }

    with pytest.raises(ValidationError):
        adapter.generate_structured(messages, SampleStructuredOutput)


def test_parsing_error_propagates(
    adapter: LangChainOllamaAdapter,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    parsing_error = ValueError("langchain parsing failed")
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content="not-json"),
        "parsed": None,
        "parsing_error": parsing_error,
    }

    with pytest.raises(ValueError, match="langchain parsing failed"):
        adapter.generate_structured(messages, SampleStructuredOutput)


def test_missing_parsed_result_fails(
    adapter: LangChainOllamaAdapter,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content=""),
        "parsed": None,
        "parsing_error": None,
    }

    with pytest.raises(ValueError, match="no parsed result"):
        adapter.generate_structured(messages, SampleStructuredOutput)


def test_generation_options_preserved(
    adapter: LangChainOllamaAdapter,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    parsed = SampleStructuredOutput(status="ok", count=2)
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content='{"status":"ok","count":2}'),
        "parsed": parsed,
        "parsing_error": None,
    }

    adapter.generate_structured(messages, SampleStructuredOutput, temperature=0.2, max_tokens=256)

    _, kwargs = structured_runnable.invoke.call_args
    assert kwargs["options"]["temperature"] == 0.2
    assert kwargs["options"]["num_predict"] == 256


def test_no_legacy_json_extraction(
    adapter: LangChainOllamaAdapter,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    parsed = SampleStructuredOutput(status="ok", count=2)
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content='noise {bad json} trailing'),
        "parsed": parsed,
        "parsing_error": None,
    }

    with patch.object(
        LangChainOllamaAdapter,
        "_extract_json_object",
        side_effect=AssertionError("legacy JSON extraction must not be used"),
    ):
        result = adapter.generate_structured(messages, SampleStructuredOutput)

    assert result.parsed == parsed
    assert result.response.content == 'noise {bad json} trailing'


def test_adapter_passes_projected_dictionary_schema(
    adapter: LangChainOllamaAdapter,
    fake_chat: MagicMock,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    parsed = SampleStructuredOutput(status="ok", count=2)
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content='{"status":"ok","count":2}'),
        "parsed": parsed,
        "parsing_error": None,
    }

    adapter.generate_structured(messages, SampleStructuredOutput)

    generation_schema = fake_chat.with_structured_output.call_args[0][0]
    assert isinstance(generation_schema, dict)
    assert generation_schema["type"] == "object"


def test_broader_generation_schema_payload_still_fails_model_validation(
    adapter: LangChainOllamaAdapter,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content='{"status":"ok"}'),
        "parsed": {"status": "ok"},
        "parsing_error": None,
    }

    with pytest.raises(ValidationError):
        adapter.generate_structured(messages, SampleStructuredOutput)


def test_failure_usage_lifecycle(
    adapter: LangChainOllamaAdapter,
    structured_runnable: MagicMock,
    messages: list[ChatMessage],
) -> None:
    structured_runnable.invoke.return_value = {
        "raw": AIMessage(content=""),
        "parsed": None,
        "parsing_error": None,
    }

    with pytest.raises(ValueError, match="no parsed result"):
        adapter.generate_structured(messages, SampleStructuredOutput, run_id="usage-fail")

    stats = adapter.usage.get_run_stats("usage-fail")
    assert stats.calls == 1
    assert stats.errors == 1
