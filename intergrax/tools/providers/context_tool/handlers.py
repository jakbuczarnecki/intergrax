# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.context_tool.contracts import (
    ContextEstimateTokensInput,
    ContextEstimateTokensOutput,
    ContextSummarizeInput,
    ContextSummarizeOutput,
)
from intergrax.tools.providers.context_tool.service import context_estimate_tokens, context_summarize


class ContextSummarizeHandler(ServiceToolHandler[ContextSummarizeInput, ContextSummarizeOutput]):
    _service = context_summarize


class ContextEstimateTokensHandler(ServiceToolHandler[ContextEstimateTokensInput, ContextEstimateTokensOutput]):
    _service = context_estimate_tokens
