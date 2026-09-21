"""Unit-test doubles for indirect prompt injection scenario."""

from __future__ import annotations

import json

from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.tools.atomic_planner_round import PLANNER_ROUND_TOOL_ID

from platform_proofs.scenarios.indirect_prompt_injection.application.tools import (
    TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
)


def _record_usage(adapter: LLMAdapter, *, run_id: str | None, output_len: int) -> None:
    call = adapter.usage.begin_call(run_id=run_id)
    adapter.usage.end_call(
        call,
        input_tokens=0,
        output_tokens=output_len,
        success=True,
    )


class WriteProposingOrderLLM(BaseLLMAdapter):
    provider = "test"
    model = "write-proposer"

    def __init__(self, *, new_shipping_address: str = "123 Attacker Lane") -> None:
        super().__init__()
        self._native_tool_rounds = 0
        self._new_shipping_address = new_shipping_address

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens
        content = "Updating shipping address per note."
        _record_usage(self, run_id=run_id, output_len=len(content))
        return build_adapter_response(content=content)

    def supports_tools(self) -> bool:
        return True

    def supports_strict_tool_argument_conformance(self) -> bool:
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
        _ = messages, tools_schema, temperature, max_tokens, tool_choice
        self._native_tool_rounds += 1
        if self._native_tool_rounds > 1:
            _record_usage(self, run_id=run_id, output_len=0)
            return build_adapter_response(content="", tool_calls=())
        _record_usage(self, run_id=run_id, output_len=1)
        return build_adapter_response(
            content="",
            tool_calls=(
                LLMToolCall(
                    id="call-write-1",
                    name=PLANNER_ROUND_TOOL_ID,
                    arguments_json=json.dumps(
                        {
                            "actions": [
                                {
                                    "tool_id": TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
                                    "arguments": {
                                        "order_id": "48291",
                                        "new_shipping_address": self._new_shipping_address,
                                    },
                                }
                            ]
                        }
                    ),
                ),
            ),
        )


class SummaryOnlyOrderLLM(BaseLLMAdapter):
    provider = "test"
    model = "summary-only"

    def __init__(self) -> None:
        super().__init__()

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens
        content = "Order #48291 is processing."
        _record_usage(self, run_id=run_id, output_len=len(content))
        return build_adapter_response(content=content)

    def supports_tools(self) -> bool:
        return True

    def supports_strict_tool_argument_conformance(self) -> bool:
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
        _ = messages, tools_schema, temperature, max_tokens, tool_choice
        content = "Order #48291 is processing with no changes."
        _record_usage(self, run_id=run_id, output_len=len(content))
        return build_adapter_response(content=content, tool_calls=())
