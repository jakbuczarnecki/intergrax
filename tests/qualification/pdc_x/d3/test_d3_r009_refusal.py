# © Artur Czarnecki. All rights reserved.

"""PDC-D3-R009 — a refusal is normalized the same way for every provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters._shared.openai_completion_mapping import (
    adapter_response_from_openai_chat_completion,
    adapter_response_from_openai_responses,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import (
    REFUSAL_UNSPECIFIED,
    LLMFinishReason,
    normalize_refusal,
    parse_finish_reason,
    refusal_text,
)
from openai.types.chat import ChatCompletion

pytestmark = pytest.mark.unit

_REFUSAL_TEXT = "I can't help with that request."


def assert_normalized_refusal(response: LLMAdapterResponse, *, text: str | None = None) -> None:
    """The one shape every provider's refusal must take."""
    assert response.finish_reason is LLMFinishReason.REFUSAL
    assert isinstance(response.refusal, str) and response.refusal.strip()
    if text is not None:
        assert response.refusal == text


def assert_not_refusal(response: LLMAdapterResponse) -> None:
    assert response.finish_reason is not LLMFinishReason.REFUSAL
    assert response.refusal is None


# --------------------------------------------------------------------------- vendor vocabularies


@pytest.mark.parametrize(
    ("vendor", "raw", "expected"),
    [
        ("anthropic", "refusal", LLMFinishReason.REFUSAL),
        ("generic", "refused", LLMFinishReason.REFUSAL),
        ("openai", "content_filter", LLMFinishReason.CONTENT_FILTER),
        ("gemini", "SAFETY", LLMFinishReason.CONTENT_FILTER),
        ("gemini", "PROHIBITED_CONTENT", LLMFinishReason.CONTENT_FILTER),
        ("gemini", "BLOCKLIST", LLMFinishReason.CONTENT_FILTER),
        ("gemini", "SPII", LLMFinishReason.CONTENT_FILTER),
        ("gemini", "RECITATION", LLMFinishReason.CONTENT_FILTER),
        ("bedrock", "guardrail_intervened", LLMFinishReason.CONTENT_FILTER),
        ("bedrock", "content_filtered", LLMFinishReason.CONTENT_FILTER),
        ("cohere", "ERROR_TOXIC", LLMFinishReason.CONTENT_FILTER),
        # Existing mappings stay untouched.
        ("anthropic", "end_turn", LLMFinishReason.COMPLETED),
        ("anthropic", "tool_use", LLMFinishReason.TOOL_CALLS),
        ("openai", "length", LLMFinishReason.LENGTH),
        ("any", "something_new", LLMFinishReason.UNKNOWN),
    ],
)
def test_vendor_stop_reasons_map_to_one_vocabulary(vendor: str, raw: str, expected: LLMFinishReason) -> None:
    assert parse_finish_reason(raw) is expected


# --------------------------------------------------------------------------- real provider mappings


def _openai_completion(*, content: str | None, refusal: str | None, finish: str = "stop") -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": content, "refusal": refusal},
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )


def test_openai_chat_refusal_payload_is_normalized() -> None:
    res = adapter_response_from_openai_chat_completion(
        _openai_completion(content=None, refusal=_REFUSAL_TEXT), model="gpt-4o-mini", provider="openai"
    )
    assert_normalized_refusal(res, text=_REFUSAL_TEXT)
    assert res.content == ""


def test_openai_chat_ordinary_answer_carries_no_refusal() -> None:
    res = adapter_response_from_openai_chat_completion(
        _openai_completion(content="hi", refusal=None), model="gpt-4o-mini", provider="openai"
    )
    assert_not_refusal(res)
    assert res.finish_reason is LLMFinishReason.COMPLETED


def test_openai_chat_content_filter_is_not_reported_as_a_refusal() -> None:
    res = adapter_response_from_openai_chat_completion(
        _openai_completion(content="", refusal=None, finish="content_filter"),
        model="gpt-4o-mini",
        provider="openai",
    )
    assert res.finish_reason is LLMFinishReason.CONTENT_FILTER
    assert res.refusal is None


def test_openai_responses_refusal_part_is_normalized() -> None:
    response = SimpleNamespace(
        id="resp-1",
        status="completed",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        output=[
            SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal=_REFUSAL_TEXT)]),
        ],
    )
    res = adapter_response_from_openai_responses(response, model="gpt-4o", provider="openai", content="")
    assert_normalized_refusal(res, text=_REFUSAL_TEXT)


def test_openai_responses_without_refusal_part_is_unchanged() -> None:
    response = SimpleNamespace(
        id="resp-2",
        status="completed",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        output=[SimpleNamespace(content=[SimpleNamespace(type="output_text", text="ok")])],
    )
    res = adapter_response_from_openai_responses(response, model="gpt-4o", provider="openai", content="ok")
    assert_not_refusal(res)


def test_claude_refusal_stop_reason_is_normalized() -> None:
    pytest.importorskip("anthropic", reason="optional extra llm-anthropic; the source gate covers the Claude adapter")
    from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter

    adapter = ClaudeChatAdapter(client=MagicMock(), model="claude-3-5-sonnet-latest")
    resp = SimpleNamespace(id="msg-1", stop_reason="refusal", content=[])
    res = adapter._build_response_from_message(resp, content="")
    assert_normalized_refusal(res, text=REFUSAL_UNSPECIFIED)


def test_claude_end_turn_is_not_a_refusal() -> None:
    pytest.importorskip("anthropic", reason="optional extra llm-anthropic; the source gate covers the Claude adapter")
    from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter

    adapter = ClaudeChatAdapter(client=MagicMock(), model="claude-3-5-sonnet-latest")
    resp = SimpleNamespace(id="msg-2", stop_reason="end_turn", content=[])
    assert_not_refusal(adapter._build_response_from_message(resp, content="fine"))


# --------------------------------------------------------------------------- the shared builder


@pytest.mark.parametrize(
    "provider", ["openai", "claude", "gemini", "mistral", "aws_bedrock", "cohere_native", "ollama"]
)
def test_builder_yields_the_same_refusal_shape_for_every_provider(provider: str) -> None:
    with_text = build_adapter_response(
        content="", finish_reason=LLMFinishReason.COMPLETED, provider=provider, refusal=_REFUSAL_TEXT
    )
    assert_normalized_refusal(with_text, text=_REFUSAL_TEXT)

    finish_only = build_adapter_response(content="", finish_reason=LLMFinishReason.REFUSAL, provider=provider)
    assert_normalized_refusal(finish_only, text=REFUSAL_UNSPECIFIED)


@pytest.mark.parametrize("finish", [f for f in LLMFinishReason])
@pytest.mark.parametrize("payload", [None, "", "   ", _REFUSAL_TEXT, MagicMock(), 42])
def test_refusal_invariant_holds_for_every_combination(finish: LLMFinishReason, payload: Any) -> None:
    """finish_reason is REFUSAL if and only if refusal is a non-empty string (except an ERROR finish)."""
    res = build_adapter_response(content="x", finish_reason=finish, refusal=payload)
    if res.finish_reason is LLMFinishReason.REFUSAL:
        assert isinstance(res.refusal, str) and res.refusal
    else:
        assert res.refusal is None
    if refusal_text(payload) is not None and finish is not LLMFinishReason.ERROR:
        assert res.finish_reason is LLMFinishReason.REFUSAL


def test_error_finish_is_never_rewritten_into_a_refusal() -> None:
    assert normalize_refusal(LLMFinishReason.ERROR, _REFUSAL_TEXT) == (LLMFinishReason.ERROR, None)


def test_non_string_refusal_payload_from_an_sdk_stub_is_dropped() -> None:
    assert refusal_text(MagicMock()) is None
    assert refusal_text(["not", "text"]) is None
