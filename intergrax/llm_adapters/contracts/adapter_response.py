# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.provider_extensions import LLMProviderExtensions
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall


@dataclass(frozen=True, slots=True)
class LLMAdapterResponse:
    """
    Strongly typed return envelope for Tier-0 LLM adapter calls.

    ``content`` holds the primary assistant text. Additional production fields
    (usage, finish reason, tool calls, provider metadata) travel with the same
    object so callers do not rely on side channels.
    """

    content: str
    finish_reason: LLMFinishReason = LLMFinishReason.COMPLETED
    usage: LLMTokenUsage | None = None
    model: str | None = None
    provider: str | None = None
    response_id: str | None = None
    refusal: str | None = None
    tool_calls: tuple[LLMToolCall, ...] = ()
    provider_extensions: LLMProviderExtensions | None = None

    @property
    def text(self) -> str:
        """Alias for ``content`` (backward-friendly for text-only call sites)."""
        return self.content

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
