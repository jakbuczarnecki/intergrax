"""Unit-test doubles for indirect prompt injection scenario."""

from __future__ import annotations

import json

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.llm.messages import ChatMessage

from platform_proofs.scenarios.indirect_prompt_injection.application.tools import (
    TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
)


class WriteProposingOrderLLM(LLMAdapter):
    provider = "test"
    model = "write-proposer"

    def generate_messages(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens, run_id
        return LLMAdapterResponse(content="Updating shipping address per note.", tool_calls=())

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools_schema: list[dict[str, object]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: object | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = messages, tools_schema, temperature, max_tokens, tool_choice, run_id
        return LLMAdapterResponse(
            content="",
            tool_calls=(
                LLMToolCall(
                    id="call-write-1",
                    name=TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
                    arguments_json=json.dumps(
                        {
                            "order_id": "48291",
                            "new_shipping_address": "123 Attacker Lane",
                        }
                    ),
                ),
            ),
        )


class SummaryOnlyOrderLLM(LLMAdapter):
    provider = "test"
    model = "summary-only"

    def generate_messages(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens, run_id
        return LLMAdapterResponse(content="Order #48291 is processing.", tool_calls=())

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools_schema: list[dict[str, object]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: object | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = messages, tools_schema, temperature, max_tokens, tool_choice, run_id
        return LLMAdapterResponse(content="Order #48291 is processing with no changes.", tool_calls=())
