# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass


class LLMTokenUsageValidationError(ValueError):
    """Raised when provider-reported token counts are internally inconsistent."""


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

    @property
    def uncached_input_tokens(self) -> int | None:
        """Provider-reported uncached prompt tokens, or ``None`` when counts are invalid."""
        if self.cached_input_tokens > self.input_tokens:
            return None
        return int(self.input_tokens) - int(self.cached_input_tokens)

    @classmethod
    def from_counts(
        cls,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
        validate_cached_bounds: bool = False,
    ) -> LLMTokenUsage:
        usage = cls(
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            cached_input_tokens=int(cached_input_tokens or 0),
            reasoning_tokens=int(reasoning_tokens or 0),
        )
        if validate_cached_bounds and usage.uncached_input_tokens is None:
            raise LLMTokenUsageValidationError(
                "cached_input_tokens exceeds input_tokens"
            )
        return usage
