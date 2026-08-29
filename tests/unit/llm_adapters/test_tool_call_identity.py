# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.tool_call import (
    LLMToolCall,
    ToolCallIdentityError,
    finalize_accepted_tool_call_identities,
    merge_streaming_tool_calls,
    tool_calls_from_langchain_message,
    tool_calls_from_openai_dicts,
    validate_tool_call_identities,
)
from intergrax.llm_adapters.providers._langchain_compat import (
    tool_calls_from_langchain_message as parse_langchain_tool_calls,
)
from intergrax.runtime.nexus.tools.native_tool_plan_alignment import (
    validate_native_tool_plan_alignment,
)
from intergrax.runtime.nexus.tools.tool_loop import append_native_tool_messages
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.execution_models import ToolExecutionResult, ToolModelObservation
from pydantic import BaseModel


pytestmark = pytest.mark.unit


class _ProbeIn(BaseModel):
    label: str = "ok"


def test_valid_provider_id_preserved() -> None:
    call = LLMToolCall.from_openai_shape(
        call_id="provider-call-123",
        name="probe.a",
        arguments={"x": 1},
    )
    (normalized,) = finalize_accepted_tool_call_identities((call,))
    assert normalized.id == "provider-call-123"
    assert normalized.name == "probe.a"
    assert normalized.arguments_json == '{"x": 1}'


@pytest.mark.parametrize("blank_id", ["", "   "])
def test_blank_id_becomes_non_empty(blank_id: str) -> None:
    call = LLMToolCall.from_openai_shape(
        call_id=blank_id,
        name="probe.a",
        arguments={},
    )
    (normalized,) = finalize_accepted_tool_call_identities((call,))
    assert normalized.id.startswith("toolcall-")
    assert normalized.id.strip()


def test_duplicate_explicit_provider_ids_fail() -> None:
    calls = (
        LLMToolCall.from_openai_shape(
            call_id="provider-call-1",
            name="a",
            arguments={},
        ),
        LLMToolCall.from_openai_shape(
            call_id="provider-call-1",
            name="b",
            arguments={},
        ),
    )
    with pytest.raises(ToolCallIdentityError, match="duplicate tool call identity"):
        finalize_accepted_tool_call_identities(calls)


def test_openai_dicts_duplicate_explicit_ids_fail() -> None:
    with pytest.raises(ToolCallIdentityError, match="duplicate tool call identity"):
        tool_calls_from_openai_dicts(
            [
                {
                    "id": "provider-call-1",
                    "function": {"name": "a", "arguments": "{}"},
                },
                {
                    "id": "provider-call-1",
                    "function": {"name": "b", "arguments": "{}"},
                },
            ]
        )


def test_mixed_valid_blank_provider_ids_preserved_and_unique() -> None:
    calls = tool_calls_from_openai_dicts(
        [
            {"id": "provider-a", "function": {"name": "a", "arguments": "{}"}},
            {"id": "", "function": {"name": "b", "arguments": "{}"}},
            {"id": "provider-c", "function": {"name": "c", "arguments": "{}"}},
        ]
    )
    ids = [call.id for call in calls]
    assert ids[0] == "provider-a"
    assert ids[2] == "provider-c"
    assert ids[1].startswith("toolcall-")
    assert len(set(ids)) == 3


def test_mint_collision_with_seen_provider_id_retries() -> None:
    calls = (
        LLMToolCall.from_openai_shape(call_id="toolcall-collision", name="a", arguments={}),
        LLMToolCall.from_openai_shape(call_id="", name="b", arguments={}),
    )
    with patch(
        "intergrax.llm_adapters.contracts.tool_call.mint_tool_call_id",
        side_effect=["toolcall-collision", "toolcall-unique"],
    ):
        normalized = finalize_accepted_tool_call_identities(calls)
    assert normalized[0].id == "toolcall-collision"
    assert normalized[1].id == "toolcall-unique"


def test_two_blank_calls_receive_distinct_ids() -> None:
    calls = (
        LLMToolCall.from_openai_shape(call_id="", name="a", arguments={}),
        LLMToolCall.from_openai_shape(call_id="", name="b", arguments={}),
    )
    normalized = finalize_accepted_tool_call_identities(calls)
    assert len({call.id for call in normalized}) == 2
    assert all(call.id.startswith("toolcall-") for call in normalized)


def test_generated_ids_exclude_argument_content() -> None:
    secret = "super-secret-prompt-leak"
    (normalized,) = finalize_accepted_tool_call_identities(
        (
            LLMToolCall.from_openai_shape(
                call_id="",
                name="probe.a",
                arguments={"payload": secret},
            ),
        )
    )
    assert secret not in normalized.id


def test_openai_message_empty_id_normalized() -> None:
    (call,) = tool_calls_from_openai_dicts(
        [
            {
                "id": "",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": "x"}'},
            }
        ]
    )
    assert call.id.startswith("toolcall-")
    assert call.name == "lookup"
    assert call.arguments_json == '{"q": "x"}'


def test_langchain_empty_ids_normalized_distinctly() -> None:
    message = type(
        "AIMessage",
        (),
        {
            "tool_calls": [
                {"id": "", "name": "lookup", "args": {"q": "a"}},
                {"id": "   ", "name": "search", "args": {"q": "b"}},
            ]
        },
    )()
    calls = parse_langchain_tool_calls(message)
    assert len(calls) == 2
    assert calls[0].id.startswith("toolcall-")
    assert calls[1].id.startswith("toolcall-")
    assert calls[0].id != calls[1].id


def test_langchain_valid_provider_id_unchanged() -> None:
    message = type(
        "AIMessage",
        (),
        {
            "tool_calls": [
                {"id": "lc-call-1", "name": "lookup", "args": {"q": "a"}},
            ]
        },
    )()
    (call,) = tool_calls_from_langchain_message(message)
    assert call.id == "lc-call-1"


def test_openai_dicts_multiple_empty_ids_distinct() -> None:
    calls = tool_calls_from_openai_dicts(
        [
            {"id": "", "function": {"name": "a", "arguments": "{}"}},
            {"id": "", "function": {"name": "b", "arguments": "{}"}},
        ]
    )
    assert len(calls) == 2
    assert calls[0].id != calls[1].id


def test_streaming_merge_with_provider_id_preserves_id() -> None:
    merged = merge_streaming_tool_calls(
        (
            LLMToolCall.from_openai_shape(
                call_id="stream-1",
                name="lookup",
                arguments='{"q":',
            ),
            LLMToolCall.from_openai_shape(
                call_id="stream-1",
                name="lookup",
                arguments='"x"}',
            ),
        )
    )
    assert len(merged) == 1
    assert merged[0].id == "stream-1"
    assert merged[0].arguments_json == '{"q":"x"}'


def test_streaming_merge_without_id_mints_once_for_logical_call() -> None:
    merged = merge_streaming_tool_calls(
        (
            LLMToolCall.from_openai_shape(call_id="", name="lookup", arguments='{"q":'),
            LLMToolCall.from_openai_shape(call_id="", name="lookup", arguments='"x"}'),
        )
    )
    assert len(merged) == 1
    assert merged[0].id.startswith("toolcall-")
    assert merged[0].arguments_json == '{"q":"x"}'


def test_streaming_merge_multiple_empty_ids_distinct() -> None:
    merged = merge_streaming_tool_calls(
        (
            LLMToolCall.from_openai_shape(call_id="", name="a", arguments="{}"),
            LLMToolCall.from_openai_shape(call_id="", name="b", arguments="{}"),
        )
    )
    assert len(merged) == 2
    assert merged[0].id != merged[1].id


def test_validate_tool_call_identities_rejects_duplicate() -> None:
    calls = (
        LLMToolCall(id="provider-call-1", name="a", arguments_json="{}"),
        LLMToolCall(id="provider-call-1", name="b", arguments_json="{}"),
    )
    with pytest.raises(ToolCallIdentityError, match="duplicate tool call identity"):
        validate_tool_call_identities(calls)


def test_validate_tool_call_identities_rejects_blank() -> None:
    with pytest.raises(ToolCallIdentityError, match="empty identity"):
        validate_tool_call_identities(
            (LLMToolCall(id="   ", name="probe.a", arguments_json="{}"),)
        )


def test_validate_native_tool_plan_alignment_rejects_blank_id() -> None:
    with pytest.raises(ToolCallIdentityError, match="empty identity"):
        validate_native_tool_plan_alignment(
            (LLMToolCall(id="", name="probe.a", arguments_json="{}"),),
            ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="probe.a",
                        input=_ProbeIn(),
                    )
                ]
            ),
        )


def test_normalized_id_flows_to_tool_message() -> None:
    provider_id = "provider-stable-id"
    call = LLMToolCall.from_openai_shape(
        call_id=provider_id,
        name="probe.a",
        arguments={"label": "ok"},
    )
    (normalized,) = finalize_accepted_tool_call_identities((call,))
    messages: list[ChatMessage] = []
    result = ToolExecutionResult.ok(_ProbeIn(label="ok"))
    append_native_tool_messages(
        messages,
        assistant_content="",
        tool_calls=(normalized,),
        outcomes=[
            type(
                "Outcome",
                (),
                {
                    "model_observation": ToolModelObservation.from_execution_result(result),
                },
            )()
        ],
    )
    tool_message = next(message for message in messages if message.role == "tool")
    assert tool_message.tool_call_id == provider_id
