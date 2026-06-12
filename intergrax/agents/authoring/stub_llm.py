# © Artur Czarnecki. All rights reserved.

"""Deterministic stub LLM adapters for fleet migration scaffolds and lab agents."""

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.memory.conversational_memory import ChatMessage


class PrefixStubLLMAdapter(LLMAdapter):
    """Returns ``{prefix}: {last_user_message}`` for harness and staging agents."""

    def __init__(self, *, prefix: str, provider: str | None = None, model: str = "stub") -> None:
        self.provider = provider or prefix
        self.model = model
        self._prefix = prefix

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        _ = temperature, max_tokens, run_id
        for msg in reversed(messages):
            content = msg.content or ""
            if content:
                return build_adapter_response(content=f"{self._prefix}: {content[:200]}")
        return build_adapter_response(content=f"{self._prefix}: (empty)")
