# © Artur Czarnecki. All rights reserved.

"""Typed last-call summary for adaptive harness signals (M-LLM-R.7.5)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse


@dataclass(frozen=True, slots=True)
class LLMCallSummary:
    finish_reason: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    has_refusal: bool
    has_tool_calls: bool


def llm_call_summary_from_response(response: LLMAdapterResponse) -> LLMCallSummary:
    usage = response.usage
    return LLMCallSummary(
        finish_reason=response.finish_reason.value,
        model=response.model or "",
        provider=response.provider or "",
        input_tokens=int(usage.input_tokens) if usage else 0,
        output_tokens=int(usage.output_tokens) if usage else 0,
        has_refusal=bool(response.refusal),
        has_tool_calls=response.has_tool_calls,
    )
