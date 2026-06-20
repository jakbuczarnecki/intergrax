# © Artur Czarnecki. All rights reserved.

"""Optional vendor tokenizer plugin contract (M-LLM-X.14.7)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from intergrax.llm.messages import ChatMessage


@runtime_checkable
class TokenizerPlugin(Protocol):
    """
    Optional vendor-native tokenizer for accurate budgeting on non-OpenAI models.

    Hosts may register a plugin per provider slug; default adapters continue to use
    tiktoken heuristics where vendor tokenizers are unavailable.
    """

    def count_text_tokens(self, text: str) -> int: ...

    def count_messages_tokens(self, messages: Sequence[ChatMessage]) -> int: ...
