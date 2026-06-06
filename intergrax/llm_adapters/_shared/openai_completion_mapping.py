# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion_usage import CompletionUsage

from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason, parse_finish_reason
from intergrax.llm_adapters.contracts.provider_extensions import LLMProviderExtensions, OpenAIProviderExtensions
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall, tool_calls_from_openai_dicts, tool_calls_from_openai_message


def usage_from_openai_chat_completion(usage: CompletionUsage | None) -> LLMTokenUsage:
    if usage is None:
        return LLMTokenUsage()
    cached = 0
    if usage.prompt_tokens_details is not None:
        cached = int(usage.prompt_tokens_details.cached_tokens or 0)
    reasoning = 0
    if usage.completion_tokens_details is not None:
        reasoning = int(usage.completion_tokens_details.reasoning_tokens or 0)
    return LLMTokenUsage.from_counts(
        input_tokens=int(usage.prompt_tokens or 0),
        output_tokens=int(usage.completion_tokens or 0),
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
    )


def usage_from_openai_responses(usage: Any) -> LLMTokenUsage:
    if usage is None:
        return LLMTokenUsage()
    try:
        input_tokens = int(usage.input_tokens or 0)
        output_tokens = int(usage.output_tokens or 0)
    except AttributeError:
        return LLMTokenUsage()
    return LLMTokenUsage.from_counts(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def adapter_response_from_openai_responses(
    response: Any,
    *,
    model: str,
    provider: str,
    content: str = "",
    tool_calls: tuple[Any, ...] = (),
) -> LLMAdapterResponse:
    usage = usage_from_openai_responses(response.usage)
    response_id = str(response.id or "") or None
    finish = parse_finish_reason(response.status)
    typed_calls: tuple[LLMToolCall, ...]
    if tool_calls and isinstance(tool_calls[0], LLMToolCall):
        typed_calls = tuple(tool_calls)  # type: ignore[return-value]
    else:
        typed_calls = tool_calls_from_openai_dicts(tool_calls)
    if typed_calls and finish == LLMFinishReason.COMPLETED:
        finish = LLMFinishReason.TOOL_CALLS
    extensions = LLMProviderExtensions(usage_source="sdk")
    return build_adapter_response(
        content=content or "",
        finish_reason=finish,
        usage=usage,
        model=model,
        provider=provider,
        response_id=response_id,
        tool_calls=typed_calls,
        provider_extensions=extensions,
    )


def adapter_response_from_openai_chat_completion(
    res: ChatCompletion,
    *,
    model: str,
    provider: str,
) -> LLMAdapterResponse:
    usage = usage_from_openai_chat_completion(res.usage)
    response_id = str(res.id or "") or None
    fingerprint = res.system_fingerprint
    extensions = LLMProviderExtensions(
        usage_source="sdk",
        openai=OpenAIProviderExtensions(system_fingerprint=str(fingerprint) if fingerprint else None),
    )
    choices = res.choices or []
    if not choices:
        return build_adapter_response(
            content="",
            finish_reason=LLMFinishReason.COMPLETED,
            usage=usage,
            model=model,
            provider=provider,
            response_id=response_id,
            provider_extensions=extensions,
        )
    choice = choices[0]
    msg = choice.message
    finish = parse_finish_reason(choice.finish_reason)
    tool_calls = tool_calls_from_openai_message(msg)
    if tool_calls and finish == LLMFinishReason.COMPLETED:
        finish = LLMFinishReason.TOOL_CALLS
    return build_adapter_response(
        content=msg.content or "",
        finish_reason=finish,
        usage=usage,
        model=model,
        provider=provider,
        response_id=response_id,
        tool_calls=tool_calls,
        provider_extensions=extensions,
    )
