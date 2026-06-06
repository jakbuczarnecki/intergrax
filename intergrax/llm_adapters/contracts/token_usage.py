# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMTokenUsage:
    """Token accounting for a single LLM adapter call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return int(self.input_tokens) + int(self.output_tokens)

    @classmethod
    def from_counts(
        cls,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> LLMTokenUsage:
        return cls(
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            cached_input_tokens=int(cached_input_tokens or 0),
            reasoning_tokens=int(reasoning_tokens or 0),
        )
