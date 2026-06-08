# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.nexus.context.context_budget import (
    ContextBudgetPolicy,
    estimate_tokens,
    trim_message_to_budget_tokenizer_aware,
)
from intergrax.tools.providers.context_tool.contracts import (
    ContextEstimateTokensInput,
    ContextEstimateTokensOutput,
    ContextSummarizeInput,
    ContextSummarizeOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

CONTEXT_SUMMARIZE_TOOL_ID = "context.summarize"
CONTEXT_ESTIMATE_TOKENS_TOOL_ID = "context.estimate_tokens"


def context_summarize(ctx: ToolWiringContext, params: ContextSummarizeInput) -> ContextSummarizeOutput:
    _ = ctx
    original = params.text
    original_tokens = estimate_tokens(len(original))
    policy = ContextBudgetPolicy(max_tokens_estimate=params.max_tokens, max_chars=len(original))
    result = trim_message_to_budget_tokenizer_aware(original, policy)
    final_tokens = estimate_tokens(len(result.message))
    return ContextSummarizeOutput(
        summary=result.message,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        trimmed=bool(result.trimmed),
    )


def context_estimate_tokens(ctx: ToolWiringContext, params: ContextEstimateTokensInput) -> ContextEstimateTokensOutput:
    _ = ctx
    text = params.text or ""
    char_count = len(text)
    return ContextEstimateTokensOutput(char_count=char_count, token_estimate=estimate_tokens(char_count))
