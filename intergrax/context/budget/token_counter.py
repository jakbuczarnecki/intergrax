# © Artur Czarnecki. All rights reserved.

"""Replaceable token counting for context budgeting (CE-02)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from intergrax.llm.messages import ChatMessage


@runtime_checkable
class ContextTokenCounter(Protocol):
    """Provider-neutral token estimator."""

    @property
    def strategy_id(self) -> str: ...

    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: Sequence[ChatMessage]) -> int: ...


class CharEstimateContextTokenCounter:
    """Deterministic chars/4 estimator — default, no tokenizer dependency."""

    @property
    def strategy_id(self) -> str:
        return "char_estimate_token_counter.v1"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 4) if text else 0

    def count_messages(self, messages: Sequence[ChatMessage]) -> int:
        return sum(self.count_text(message.content or "") for message in messages)
