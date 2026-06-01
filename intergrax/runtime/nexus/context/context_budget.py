# © Artur Czarnecki. All rights reserved.

"""Context budget policy and trimming (architecture §28.1, Phase R-Context)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from intergrax.contracts.context_assembly import ContextSummaryTier


@dataclass(frozen=True, slots=True)
class ContextBudgetPolicy:
    """Central limits for assembled agent messages."""

    max_chars: int = 16_000
    max_tokens_estimate: int = 4_000
    summary_tier: ContextSummaryTier = ContextSummaryTier.FULL

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        if self.max_tokens_estimate < 1:
            raise ValueError("max_tokens_estimate must be >= 1")


@dataclass(frozen=True, slots=True)
class ContextTrimResult:
    message: str
    trimmed: bool
    original_chars: int
    final_chars: int


def trim_message_to_budget(message: str, policy: ContextBudgetPolicy) -> ContextTrimResult:
    original = len(message)
    if original <= policy.max_chars:
        return ContextTrimResult(
            message=message,
            trimmed=False,
            original_chars=original,
            final_chars=original,
        )
    trimmed_text = message[: policy.max_chars]
    return ContextTrimResult(
        message=trimmed_text,
        trimmed=True,
        original_chars=original,
        final_chars=len(trimmed_text),
    )


def estimate_tokens(char_count: int) -> int:
    """Rough chars→tokens estimate for telemetry (no tokenizer dependency)."""
    return max(1, char_count // 4)
