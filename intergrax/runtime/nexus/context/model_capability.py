# © Artur Czarnecki. All rights reserved.

"""Bridge LLM adapter limits into provider-neutral CE budget snapshots (CE-02)."""

from __future__ import annotations

from typing import Optional

from intergrax.context.budget.contracts import ModelContextCapabilitySnapshot
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.context.context_budget import resolve_input_budget_tokens


def snapshot_model_capability(
    adapter: LLMAdapter,
    *,
    max_output_tokens: Optional[int] = None,
    margin_tokens: int = 256,
) -> ModelContextCapabilitySnapshot:
    """Align capability snapshot with ``resolve_input_budget_tokens`` (CE-02)."""
    context_window = max(1, int(adapter.context_window_tokens))
    allowed_input = resolve_input_budget_tokens(
        adapter,
        max_output_tokens=max_output_tokens,
        margin_tokens=margin_tokens,
    )
    platform_margin = min(max(0, margin_tokens), max(1, context_window // 4))
    reserved_output = max(1, context_window - platform_margin - allowed_input)
    while reserved_output + platform_margin >= context_window:
        if platform_margin > 0:
            platform_margin -= 1
        else:
            reserved_output = max(1, reserved_output - 1)
    return ModelContextCapabilitySnapshot(
        model_context_window=context_window,
        reserved_output_tokens=reserved_output,
        platform_margin_tokens=platform_margin,
    )


def allowed_input_tokens_from_adapter(
    adapter: LLMAdapter,
    *,
    max_output_tokens: Optional[int] = None,
    margin_tokens: int = 256,
) -> int:
    """Legacy helper — prefer ``snapshot_model_capability`` + budget resolver."""
    return resolve_input_budget_tokens(
        adapter,
        max_output_tokens=max_output_tokens,
        margin_tokens=margin_tokens,
    )
