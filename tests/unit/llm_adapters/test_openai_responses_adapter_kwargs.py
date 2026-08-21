# © Artur Czarnecki. All rights reserved.

"""PP-4A-R1 — OpenAI Responses adapter constructor/request kwargs boundary."""

from __future__ import annotations

from typing import Any, Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.openai_responses_adapter import (
    OpenAIChatResponsesAdapter,
    _OpenAIToolNameMapping,
    _map_tools_to_responses_api,
    _prepare_responses_tools_and_mapping,
)
from intergrax.llm_adapters.registry.profile import LLMProfile

pytestmark = pytest.mark.unit

_TEST_API_KEY = "test-secret"
_TEST_MODEL = "gpt-5.1"
_TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
_CANONICAL_SQL_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Run a bounded SQL query",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    }
]
_PLATFORM_PROOF_SQL_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "platform_proof.sql.query",
            "description": "Run bounded platform proof SQL",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    }
]


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
    sent_tool = create_kwargs["tools"][0]
    assert sent_tool["type"] == "function"
    assert sent_tool["name"] == "lookup"
    assert sent_tool["parameters"] == {"type": "object"}
    assert "function" not in sent_tool
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
    sent_tool = stream_tools_kwargs["tools"][0]
    assert sent_tool["type"] == "function"
    assert sent_tool["name"] == "lookup"
    assert "function" not in sent_tool
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


def test_map_tools_to_responses_api_maps_canonical_function_tool() -> None:
    mapped = _map_tools_to_responses_api(_CANONICAL_SQL_TOOL)
    assert mapped == [
        {
            "type": "function",
            "name": "sql_query",
            "description": "Run a bounded SQL query",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        }
    ]


def test_generate_with_tools_maps_canonical_schema_to_responses_shape() -> None:
    client = MagicMock()
    client.responses.create.return_value = _mock_create_response(output_text="tool-ok")

    adapter = _capture_create_client(client)
    adapter.generate_with_tools(
        [ChatMessage(role="user", content="hello")],
        _CANONICAL_SQL_TOOL,
        run_id="r1",
    )

    sent_tool = client.responses.create.call_args.kwargs["tools"][0]
    assert sent_tool["type"] == "function"
    assert sent_tool["name"] == "sql_query"
    assert sent_tool["description"] == "Run a bounded SQL query"
    assert sent_tool["parameters"] == _CANONICAL_SQL_TOOL[0]["function"]["parameters"]
    assert "function" not in sent_tool


def test_stream_with_tools_uses_same_mapping() -> None:
    client = MagicMock()
    stream_cm = MagicMock()
    stream_inner = MagicMock()
    stream_inner.__iter__.return_value = iter([])
    stream_inner.get_final_response.return_value = _mock_create_response(output_text="done")
    stream_cm.__enter__.return_value = stream_inner
    stream_cm.__exit__.return_value = False
    client.responses.stream.return_value = stream_cm

    adapter = _capture_create_client(client)
    list(
        adapter.stream_with_tools(
            [ChatMessage(role="user", content="tools-stream")],
            _CANONICAL_SQL_TOOL,
            run_id="r2",
        )
    )

    sent_tool = client.responses.stream.call_args.kwargs["tools"][0]
    assert sent_tool["name"] == "sql_query"
    assert "function" not in sent_tool


def test_tools_schema_input_is_not_mutated() -> None:
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "sql_query",
                "description": "Run a bounded SQL query",
                "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}},
            },
        }
    ]
    original = [
        {
            "type": "function",
            "function": {
                "name": "sql_query",
                "description": "Run a bounded SQL query",
                "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}},
            },
        }
    ]

    client = MagicMock()
    client.responses.create.return_value = _mock_create_response(output_text="ok")
    adapter = _capture_create_client(client)
    adapter.generate_with_tools([ChatMessage(role="user", content="x")], tools_schema, run_id="r1")

    assert tools_schema == original
    assert "function" in tools_schema[0]


@pytest.mark.parametrize(
    "bad_tools,match",
    [
        ([{"type": "function", "function": {}}], "function.name must be a non-empty string"),
        ([{"type": "function", "function": "bad"}], "function must be a dict"),
        ([{"type": "function"}], "requires nested 'function'"),
    ],
)
def test_map_tools_to_responses_api_rejects_malformed_function_tools(
    bad_tools: list[dict[str, Any]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _map_tools_to_responses_api(bad_tools)


def test_map_tools_to_responses_api_allows_optional_description() -> None:
    mapped = _map_tools_to_responses_api(
        [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}]
    )
    assert mapped == [{"type": "function", "name": "noop", "parameters": {"type": "object"}}]
    assert "description" not in mapped[0]


def test_map_tools_to_responses_api_preserves_multiple_tools_order() -> None:
    tools = [
        {"type": "function", "function": {"name": "first", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "second", "parameters": {"type": "object"}}},
    ]
    mapped = _map_tools_to_responses_api(tools)
    assert [item["name"] for item in mapped] == ["first", "second"]


def test_extract_tool_calls_round_trip_to_canonical_llm_tool_call() -> None:
    client = MagicMock()
    function_call = SimpleNamespace(
        type="function_call",
        call_id="call_abc123",
        name="sql_query",
        arguments='{"sql":"SELECT 1"}',
    )
    response = MagicMock()
    response.usage = MagicMock(input_tokens=4, output_tokens=2)
    response.output_text = ""
    response.output = [function_call]
    response.status = "completed"
    client.responses.create.return_value = response

    adapter = _capture_create_client(client)
    result = adapter.generate_with_tools(
        [ChatMessage(role="user", content="run sql")],
        _CANONICAL_SQL_TOOL,
        run_id="r1",
    )

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.id == "call_abc123"
    assert tool_call.name == "sql_query"
    assert tool_call.arguments_json == '{"sql":"SELECT 1"}'


def test_tool_choice_string_values_pass_through_unchanged() -> None:
    client = MagicMock()
    client.responses.create.return_value = _mock_create_response(output_text="ok")
    adapter = _capture_create_client(client)

    for choice in ("auto", "required", "none"):
        adapter.generate_with_tools(
            [ChatMessage(role="user", content="x")],
            _TOOLS,
            tool_choice=choice,
            run_id=f"r-{choice}",
        )
        assert client.responses.create.call_args.kwargs["tool_choice"] == choice


def test_dotted_canonical_tool_name_maps_to_provider_safe_alias() -> None:
    provider_tools, mapping = _prepare_responses_tools_and_mapping(_PLATFORM_PROOF_SQL_TOOL)
    assert provider_tools[0]["name"] == "platform_proof_sql_query"
    assert mapping.to_provider("platform_proof.sql.query") == "platform_proof_sql_query"
    assert "." not in provider_tools[0]["name"]


def test_provider_alias_maps_back_to_canonical_tool_call_name() -> None:
    client = MagicMock()
    function_call = SimpleNamespace(
        type="function_call",
        call_id="call_pp_sql",
        name="platform_proof_sql_query",
        arguments='{"sql":"SELECT 1"}',
    )
    response = MagicMock()
    response.usage = MagicMock(input_tokens=4, output_tokens=2)
    response.output_text = ""
    response.output = [function_call]
    response.status = "completed"
    client.responses.create.return_value = response

    adapter = _capture_create_client(client)
    result = adapter.generate_with_tools(
        [ChatMessage(role="user", content="run sql")],
        _PLATFORM_PROOF_SQL_TOOL,
        run_id="r1",
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "platform_proof.sql.query"


def test_collision_produces_unique_provider_names_and_reverse_maps() -> None:
    tools = [
        {"type": "function", "function": {"name": "a.b", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "a_b", "parameters": {"type": "object"}}},
    ]
    provider_tools, mapping = _prepare_responses_tools_and_mapping(tools)
    provider_names = [tool["name"] for tool in provider_tools]
    assert len(set(provider_names)) == 2
    assert mapping.to_provider("a_b") == "a_b"
    assert mapping.to_provider("a.b") != "a_b"
    assert mapping.to_canonical(mapping.to_provider("a.b")) == "a.b"
    assert mapping.to_canonical("a_b") == "a_b"


def test_tool_name_mapping_is_deterministic_for_same_tool_set() -> None:
    tools = [
        {"type": "function", "function": {"name": "a.b", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "a_b", "parameters": {"type": "object"}}},
    ]
    first_tools, first_mapping = _prepare_responses_tools_and_mapping(tools)
    second_tools, second_mapping = _prepare_responses_tools_and_mapping(tools)
    assert [tool["name"] for tool in first_tools] == [tool["name"] for tool in second_tools]
    assert first_mapping.canonical_to_provider == second_mapping.canonical_to_provider


def test_generate_with_tools_sends_provider_safe_dotted_tool_name() -> None:
    client = MagicMock()
    client.responses.create.return_value = _mock_create_response(output_text="ok")
    adapter = _capture_create_client(client)

    adapter.generate_with_tools(
        [ChatMessage(role="user", content="hello")],
        _PLATFORM_PROOF_SQL_TOOL,
        run_id="r1",
    )

    sent_tool = client.responses.create.call_args.kwargs["tools"][0]
    assert sent_tool["name"] == "platform_proof_sql_query"
    assert _PLATFORM_PROOF_SQL_TOOL[0]["function"]["name"] == "platform_proof.sql.query"


def test_stream_with_tools_uses_same_provider_name_mapping() -> None:
    client = MagicMock()
    stream_cm = MagicMock()
    stream_inner = MagicMock()
    stream_inner.__iter__.return_value = iter([])
    stream_inner.get_final_response.return_value = _mock_create_response(output_text="done")
    stream_cm.__enter__.return_value = stream_inner
    stream_cm.__exit__.return_value = False
    client.responses.stream.return_value = stream_cm

    adapter = _capture_create_client(client)
    list(
        adapter.stream_with_tools(
            [ChatMessage(role="user", content="tools-stream")],
            _PLATFORM_PROOF_SQL_TOOL,
            run_id="r2",
        )
    )

    sent_tool = client.responses.stream.call_args.kwargs["tools"][0]
    assert sent_tool["name"] == "platform_proof_sql_query"


def test_named_tool_choice_maps_canonical_name_to_provider_alias() -> None:
    client = MagicMock()
    client.responses.create.return_value = _mock_create_response(output_text="ok")
    adapter = _capture_create_client(client)

    adapter.generate_with_tools(
        [ChatMessage(role="user", content="x")],
        _PLATFORM_PROOF_SQL_TOOL,
        tool_choice={"type": "function", "name": "platform_proof.sql.query"},
        run_id="r1",
    )

    assert client.responses.create.call_args.kwargs["tool_choice"] == {
        "type": "function",
        "name": "platform_proof_sql_query",
    }


def test_unknown_provider_tool_name_fails_closed() -> None:
    client = MagicMock()
    function_call = SimpleNamespace(
        type="function_call",
        call_id="call_unknown",
        name="unexpected_alias",
        arguments="{}",
    )
    response = MagicMock()
    response.usage = MagicMock(input_tokens=1, output_tokens=1)
    response.output_text = ""
    response.output = [function_call]
    response.status = "completed"
    client.responses.create.return_value = response

    adapter = _capture_create_client(client)
    with pytest.raises(ValueError, match="unknown provider tool name 'unexpected_alias'"):
        adapter.generate_with_tools(
            [ChatMessage(role="user", content="x")],
            _PLATFORM_PROOF_SQL_TOOL,
            run_id="r1",
        )


def test_openai_tool_name_mapping_direct_collision_behavior() -> None:
    mapping = _OpenAIToolNameMapping(["a_b", "a.b"])
    assert mapping.to_provider("a_b") == "a_b"
    assert mapping.to_provider("a.b") != "a_b"
    assert mapping.to_canonical(mapping.to_provider("a.b")) == "a.b"
