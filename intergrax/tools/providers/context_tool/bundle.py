# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.context_tool.contracts import (
    ContextEstimateTokensInput,
    ContextEstimateTokensOutput,
    ContextSummarizeInput,
    ContextSummarizeOutput,
)
from intergrax.tools.providers.context_tool.handlers import (
    ContextEstimateTokensHandler,
    ContextSummarizeHandler,
)
from intergrax.tools.providers.context_tool.service import (
    CONTEXT_ESTIMATE_TOKENS_TOOL_ID,
    CONTEXT_SUMMARIZE_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

CONTEXT_BUNDLE_ID = "context"
CONTEXT_TOOL_IDS: tuple[str, ...] = (CONTEXT_SUMMARIZE_TOOL_ID, CONTEXT_ESTIMATE_TOKENS_TOOL_ID)


def register_context_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=CONTEXT_SUMMARIZE_TOOL_ID,
            name=CONTEXT_SUMMARIZE_TOOL_ID,
            description="Compress text to a target token budget using harness context trim policy.",
            description_short="Summarize/trim text.",
            input_schema=ContextSummarizeInput,
            output_schema=ContextSummarizeOutput,
            error_mapping={},
            side_effects=False,
            category="context",
            risk_level=ToolRiskLevel.LOW,
            tags=("context", "compression", "dx"),
        ),
        ContextSummarizeHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CONTEXT_ESTIMATE_TOKENS_TOOL_ID,
            name=CONTEXT_ESTIMATE_TOKENS_TOOL_ID,
            description="Estimate token count for text using harness char heuristic.",
            description_short="Estimate tokens.",
            input_schema=ContextEstimateTokensInput,
            output_schema=ContextEstimateTokensOutput,
            error_mapping={},
            side_effects=False,
            category="context",
            risk_level=ToolRiskLevel.LOW,
            tags=("context", "budget", "dx"),
        ),
        ContextEstimateTokensHandler(ctx),
    )
