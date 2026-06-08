# © Artur Czarnecki. All rights reserved.

"""Context budget policy and trimming (architecture §28.1, Phase R-Context)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from typing import Optional

from intergrax.contracts.context_assembly import ContextSummaryTier
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


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
    token_limit_chars = policy.max_tokens_estimate * 4
    effective_limit = min(policy.max_chars, token_limit_chars)
    if original <= effective_limit:
        return ContextTrimResult(
            message=message,
            trimmed=False,
            original_chars=original,
            final_chars=original,
        )
    trimmed_text = message[:effective_limit]
    return ContextTrimResult(
        message=trimmed_text,
        trimmed=True,
        original_chars=original,
        final_chars=len(trimmed_text),
    )


def resolve_input_budget_tokens(
    adapter: LLMAdapter,
    *,
    max_output_tokens: Optional[int] = None,
    margin_tokens: int = 256,
) -> int:
    """Derive global input token budget from adapter context window."""
    context_window = int(adapter.context_window_tokens)
    if max_output_tokens is not None:
        reserved = min(max_output_tokens, context_window // 2)
    else:
        reserved = context_window // 4
    reserved = max(reserved, margin_tokens)
    return max(512, context_window - reserved - margin_tokens)


def estimate_tokens(char_count: int) -> int:
    """Rough chars→tokens estimate for telemetry (no tokenizer dependency)."""
    return max(1, char_count // 4)


def trim_message_to_budget_tokenizer_aware(
    message: str,
    policy: ContextBudgetPolicy,
    *,
    count_tokens: Callable[[str], int] | None = None,
) -> ContextTrimResult:
    """
    Tokenizer-aware trim (Phase MEM-DEPTH-1.3).

    Uses ``count_tokens`` when provided; falls back to char estimate.
    """
    counter = count_tokens or (lambda text: estimate_tokens(len(text)))
    original = message
    token_count = counter(original)
    if token_count <= policy.max_tokens_estimate and len(original) <= policy.max_chars:
        return ContextTrimResult(
            message=original,
            trimmed=False,
            original_chars=len(original),
            final_chars=len(original),
        )

    # Binary search char boundary against token budget.
    low = 0
    high = min(len(original), policy.max_chars)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        snippet = original[:mid]
        if counter(snippet) <= policy.max_tokens_estimate:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    trimmed_text = original[:best] if best > 0 else original[: min(len(original), policy.max_chars)]
    return ContextTrimResult(
        message=trimmed_text,
        trimmed=True,
        original_chars=len(original),
        final_chars=len(trimmed_text),
    )
