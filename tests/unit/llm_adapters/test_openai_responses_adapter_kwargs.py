# © Artur Czarnecki. All rights reserved.

"""PP-4A-R1 — OpenAI Responses adapter constructor/request kwargs boundary."""

from __future__ import annotations

from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter
from intergrax.llm_adapters.registry.profile import LLMProfile

pytestmark = pytest.mark.unit

_TEST_API_KEY = "test-secret"
_TEST_MODEL = "gpt-5.1"
_TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]


def _mock_create_response(*, output_text: str = "ok") -> MagicMock:
    usage = MagicMock(input_tokens=3, output_tokens=2)
    response = MagicMock()
    response.usage = usage
    response.output_text = output_text
    response.output = []
    response.status = "completed"
    return response


def _capture_create_client(client: MagicMock) -> OpenAIChatResponsesAdapter:
    return OpenAIChatResponsesAdapter(
        client=client,
        model=_TEST_MODEL,
        api_key=_TEST_API_KEY,
        base_url="https://example.test/v1",
        organization="org-test",
        project="proj-test",
        calls_per_minute=100,
        store=True,
        temperature=0.2,
    )


def test_profile_passes_api_key_to_client_constructor() -> None:
    profile = LLMProfile(
        provider=LLMProvider.OPENAI,
        model=_TEST_MODEL,
        options={"api_key": _TEST_API_KEY, "base_url": "https://example.test/v1"},
    )
    with patch("intergrax.llm_adapters.providers.openai_responses_adapter.Client") as client_cls:
        client_instance = MagicMock()
        client_cls.return_value = client_instance
        adapter = profile.create_adapter()
        client_cls.assert_called_once_with(
            api_key=_TEST_API_KEY,
            base_url="https://example.test/v1",
        )
        assert adapter.model == _TEST_MODEL


def test_generate_with_tools_does_not_leak_constructor_kwargs() -> None:
    client = MagicMock()
    client.responses.create.return_value = _mock_create_response(output_text="tool-ok")

    adapter = _capture_create_client(client)
    adapter.generate_with_tools(
        [ChatMessage(role="user", content="hello")],
        _TOOLS,
        run_id="r1",
    )

    create_kwargs = client.responses.create.call_args.kwargs
    assert create_kwargs["model"] == _TEST_MODEL
    assert create_kwargs["tools"] == _TOOLS
    assert "input" in create_kwargs
    assert create_kwargs.get("store") is True
    assert create_kwargs.get("temperature") == 0.2
    for leaked in ("api_key", "base_url", "organization", "project", "client", "calls_per_minute"):
        assert leaked not in create_kwargs


def test_generate_messages_does_not_leak_constructor_kwargs() -> None:
    client = MagicMock()
    client.responses.create.return_value = _mock_create_response(output_text="plain")

    adapter = _capture_create_client(client)
    out = adapter.generate_messages([ChatMessage(role="user", content="hi")], run_id="r1")
    assert out.content == "plain"

    create_kwargs = client.responses.create.call_args.kwargs
    assert create_kwargs["model"] == _TEST_MODEL
    assert "input" in create_kwargs
    assert "api_key" not in create_kwargs
    assert "base_url" not in create_kwargs


def test_generate_structured_does_not_leak_constructor_kwargs() -> None:
    client = MagicMock()
    client.responses.create.return_value = _mock_create_response(output_text='{"value": 1}')

    adapter = _capture_create_client(client)

    class StructuredPayload(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int

    result = adapter.generate_structured(
        [ChatMessage(role="user", content="json")],
        StructuredPayload,
        run_id="r1",
    )
    assert result.parsed.value == 1

    create_kwargs = client.responses.create.call_args.kwargs
    assert create_kwargs["model"] == _TEST_MODEL
    assert "text" in create_kwargs
    assert "api_key" not in create_kwargs


def test_stream_paths_do_not_leak_constructor_kwargs() -> None:
    client = MagicMock()

    def _stream_events() -> Iterator[Any]:
        delta = MagicMock(type="response.output_text.delta", delta="chunk")
        yield delta

    stream_cm = MagicMock()
    stream_cm.__enter__.return_value = _stream_events()
    stream_cm.__exit__.return_value = False
    client.responses.stream.return_value = stream_cm

    adapter = _capture_create_client(client)
    events = list(adapter.stream_messages([ChatMessage(role="user", content="stream")], run_id="r1"))
    assert events

    stream_kwargs = client.responses.stream.call_args.kwargs
    assert stream_kwargs["model"] == _TEST_MODEL
    assert stream_kwargs.get("stream") is True
    assert "api_key" not in stream_kwargs
    assert "base_url" not in stream_kwargs

    stream_cm_tools = MagicMock()
    stream_inner = MagicMock()
    stream_inner.__iter__.return_value = iter([])
    stream_inner.get_final_response.return_value = _mock_create_response(output_text="done")
    stream_cm_tools.__enter__.return_value = stream_inner
    stream_cm_tools.__exit__.return_value = False
    client.responses.stream.return_value = stream_cm_tools

    list(
        adapter.stream_with_tools(
            [ChatMessage(role="user", content="tools-stream")],
            _TOOLS,
            run_id="r2",
        )
    )
    stream_tools_kwargs = client.responses.stream.call_args.kwargs
    assert stream_tools_kwargs["tools"] == _TOOLS
    assert "api_key" not in stream_tools_kwargs


def test_model_is_not_duplicated_through_request_defaults() -> None:
    client = MagicMock()
    client.responses.create.return_value = _mock_create_response()

    adapter = OpenAIChatResponsesAdapter(
        client=client,
        model="canonical-model",
        api_key=_TEST_API_KEY,
    )
    assert adapter.model == "canonical-model"
    assert "model" not in adapter.request_defaults

    adapter.generate_messages([ChatMessage(role="user", content="x")], run_id="r1")
    create_kwargs = client.responses.create.call_args.kwargs
    assert create_kwargs["model"] == "canonical-model"
