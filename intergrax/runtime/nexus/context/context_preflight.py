# © Artur Czarnecki. All rights reserved.

"""Pre-flight context window invariant (Phase MEM-DEPTH-1.5)."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.context.context_budget import estimate_tokens
from intergrax.runtime.nexus.context.context_budget import resolve_input_budget_tokens
from intergrax.runtime.nexus.context.context_compiler_models import ContextPreflightResult


def _default_count_tokens(text: str) -> int:
    return estimate_tokens(len(text))


def count_message_tokens(
    messages: Sequence[ChatMessage],
    *,
    count_tokens: Callable[[str], int],
) -> int:
    total = 0
    for message in messages:
        total += count_tokens(message.content or "")
    return total


def verify_context_preflight(
    messages: List[ChatMessage],
    adapter: LLMAdapter,
    *,
    max_output_tokens: Optional[int] = None,
    margin_tokens: int = 256,
    count_tokens: Callable[[str], int] | None = None,
) -> ContextPreflightResult:
    """
    Verify ``assembled_tokens + max_output <= context_window - margin``.

    Raises ValueError when invariant is violated after compile pass.
    """
    if count_tokens is None:
        assembled = int(adapter.count_messages_tokens(messages))
    else:
        assembled = count_message_tokens(messages, count_tokens=count_tokens)
    context_window = int(adapter.context_window_tokens)

    if max_output_tokens is not None:
        reserved_output = min(max_output_tokens, context_window // 2)
    else:
        reserved_output = context_window // 4

    allowed_input = resolve_input_budget_tokens(
        adapter,
        max_output_tokens=max_output_tokens,
        margin_tokens=margin_tokens,
    )

    ok = assembled <= allowed_input
    message = (
        "Context pre-flight OK"
        if ok
        else (
            f"Context overflow: assembled={assembled} allowed_input={allowed_input} "
            f"context_window={context_window} reserved_output={reserved_output}"
        )
    )

    result = ContextPreflightResult(
        ok=ok,
        assembled_tokens=assembled,
        max_output_tokens=reserved_output,
        context_window=context_window,
        margin_tokens=margin_tokens,
        message=message,
    )
    if not ok:
        raise ValueError(result.message)
    return result
