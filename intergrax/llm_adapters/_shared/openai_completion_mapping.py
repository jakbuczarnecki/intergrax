# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import replace
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion_usage import CompletionUsage

from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters._shared.openai_tool_call_interop import (
    tool_calls_from_openai_dicts,
    tool_calls_from_openai_message,
)
from intergrax.llm_adapters._shared.stream_finish_signal import finish_reason_from_raw
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import (
    LLMFinishReason,
    parse_finish_reason,
    refusal_text,
)
from intergrax.llm_adapters.contracts.provider_extensions import (
    LLMProviderExtensions,
    OpenAIProviderExtensions,
    VllmProviderExtensions,
)
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage, LLMTokenUsageValidationError
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.utils import attribute_access


def reported_served_model(response: object) -> str | None:
    """Model name the SDK response reports as having served the call (``response.model``), if any."""
    value = attribute_access.optional(response, "model", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def usage_from_openai_chat_completion(usage: CompletionUsage | None) -> LLMTokenUsage:
    mapped, _ = _usage_from_openai_chat_completion(usage, provider_is_vllm=False)
    return mapped


def vllm_usage_from_openai_chat_completion(
    usage: CompletionUsage | None,
) -> tuple[LLMTokenUsage, VllmProviderExtensions]:
    mapped, extensions = _usage_from_openai_chat_completion(usage, provider_is_vllm=True)
    assert extensions is not None
    return mapped, extensions


def _usage_from_openai_chat_completion(
    usage: CompletionUsage | None,
    *,
    provider_is_vllm: bool,
) -> tuple[LLMTokenUsage, VllmProviderExtensions | None]:
    if usage is None:
        extensions = VllmProviderExtensions(prompt_tokens_details_reported=False) if provider_is_vllm else None
        return LLMTokenUsage(), extensions

    details_reported = usage.prompt_tokens_details is not None
    cached = 0
    if details_reported:
        cached = int(usage.prompt_tokens_details.cached_tokens or 0)
    reasoning = 0
    if usage.completion_tokens_details is not None:
        reasoning = int(usage.completion_tokens_details.reasoning_tokens or 0)
    try:
        mapped = LLMTokenUsage.from_counts(
            input_tokens=int(usage.prompt_tokens or 0),
            output_tokens=int(usage.completion_tokens or 0),
            cached_input_tokens=cached,
            reasoning_tokens=reasoning,
            validate_cached_bounds=provider_is_vllm,
        )
    except LLMTokenUsageValidationError:
        raise
    extensions = None
    if provider_is_vllm:
        extensions = VllmProviderExtensions(
            prompt_tokens_details_reported=details_reported,
        )
    return mapped, extensions


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


def _refusal_from_openai_responses_output(response: Any) -> str | None:
    """First ``refusal`` content part of a Responses API result, if any."""
    output = getattr(response, "output", None)
    if not isinstance(output, (list, tuple)):
        return None
    for item in output:
        parts = getattr(item, "content", None)
        if not isinstance(parts, (list, tuple)):
            continue
        for part in parts:
            if getattr(part, "type", None) == "refusal":
                text = refusal_text(getattr(part, "refusal", None))
                if text is not None:
                    return text
    return None


def responses_stop_signal(response: Any) -> object:
    """Stop signal of a Responses API result: the incomplete reason when ``status == "incomplete"``."""
    status = attribute_access.optional(response, "status", None)
    if status == "incomplete":
        details = attribute_access.optional(response, "incomplete_details", None)
        reason = attribute_access.optional(details, "reason", None)
        if isinstance(reason, str) and reason.strip():
            return reason
    return status


def responses_refusal(response: Any) -> str | None:
    """Refusal text of a Responses API result, if any."""
    return _refusal_from_openai_responses_output(response)


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
    finish = finish_reason_from_raw(responses_stop_signal(response))
    typed_calls: tuple[LLMToolCall, ...]
    if tool_calls and isinstance(tool_calls[0], LLMToolCall):
        typed_calls = tuple(tool_calls)  # type: ignore[return-value]
    else:
        typed_calls = tool_calls_from_openai_dicts(tool_calls)
    if typed_calls and finish == LLMFinishReason.COMPLETED:
        finish = LLMFinishReason.TOOL_CALLS
    extensions = LLMProviderExtensions(usage_source="sdk")
    return replace(
        build_adapter_response(
            content=content or "",
            finish_reason=finish,
            usage=usage,
            model=model,
            provider=provider,
            response_id=response_id,
            refusal=_refusal_from_openai_responses_output(response),
            tool_calls=typed_calls,
            provider_extensions=extensions,
        ),
        served_model=reported_served_model(response),
    )


def adapter_response_from_openai_chat_completion(
    res: ChatCompletion,
    *,
    model: str,
    provider: str,
) -> LLMAdapterResponse:
    response_id = str(res.id or "") or None
    if provider == "vllm":
        usage, vllm_extensions = vllm_usage_from_openai_chat_completion(res.usage)
        extensions = LLMProviderExtensions(usage_source="sdk", vllm=vllm_extensions)
    else:
        usage = usage_from_openai_chat_completion(res.usage)
        fingerprint = res.system_fingerprint
        extensions = LLMProviderExtensions(
            usage_source="sdk",
            openai=OpenAIProviderExtensions(system_fingerprint=str(fingerprint) if fingerprint else None),
        )
    served_model = reported_served_model(res)
    choices = res.choices or []
    if not choices:
        return replace(
            build_adapter_response(
                content="",
                finish_reason=LLMFinishReason.COMPLETED,
                usage=usage,
                model=model,
                provider=provider,
                response_id=response_id,
                provider_extensions=extensions,
            ),
            served_model=served_model,
        )
    choice = choices[0]
    msg = choice.message
    finish = parse_finish_reason(choice.finish_reason)
    tool_calls = tool_calls_from_openai_message(msg)
    if tool_calls and finish == LLMFinishReason.COMPLETED:
        finish = LLMFinishReason.TOOL_CALLS
    return replace(
        build_adapter_response(
            content=msg.content or "",
            finish_reason=finish,
            usage=usage,
            model=model,
            provider=provider,
            response_id=response_id,
            refusal=refusal_text(getattr(msg, "refusal", None)),
            tool_calls=tool_calls,
            provider_extensions=extensions,
        ),
        served_model=served_model,
    )
