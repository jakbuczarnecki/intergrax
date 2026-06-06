# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.provider_extensions import LLMProviderExtensions
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent, LLMStreamEventKind
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall


def build_adapter_response(
    *,
    content: str = "",
    finish_reason: LLMFinishReason = LLMFinishReason.COMPLETED,
    usage: LLMTokenUsage | None = None,
    model: str | None = None,
    provider: str | None = None,
    response_id: str | None = None,
    refusal: str | None = None,
    tool_calls: tuple[LLMToolCall, ...] = (),
    provider_extensions: LLMProviderExtensions | None = None,
) -> LLMAdapterResponse:
    return LLMAdapterResponse(
        content=content or "",
        finish_reason=finish_reason,
        usage=usage,
        model=model,
        provider=provider,
        response_id=response_id,
        refusal=refusal,
        tool_calls=tool_calls,
        provider_extensions=provider_extensions,
    )


def partial_stream_event(*, delta_content: str) -> LLMStreamEvent:
    return LLMStreamEvent(
        kind=LLMStreamEventKind.PARTIAL,
        delta_content=delta_content,
        response=None,
    )


def final_stream_event(*, response: LLMAdapterResponse) -> LLMStreamEvent:
    return LLMStreamEvent(
        kind=LLMStreamEventKind.FINAL,
        delta_content="",
        response=response,
    )
