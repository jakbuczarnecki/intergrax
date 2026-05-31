# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.braintrust.contracts import BraintrustLogEvalInput, BraintrustLogEvalOutput
from intergrax.tools.providers.braintrust.handlers import BraintrustLogEvalHandler
from intergrax.tools.providers.braintrust.service import BRAINTRUST_LOG_EVAL_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

BRAINTRUST_BUNDLE_ID = "braintrust"
BRAINTRUST_TOOL_IDS: tuple[str, ...] = (BRAINTRUST_LOG_EVAL_TOOL_ID,)


def register_braintrust_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=BRAINTRUST_LOG_EVAL_TOOL_ID,
            name=BRAINTRUST_LOG_EVAL_TOOL_ID,
            description="Log an agent eval score to Braintrust project logs.",
            description_short="Log Braintrust eval.",
            input_schema=BraintrustLogEvalInput,
            output_schema=BraintrustLogEvalOutput,
            error_mapping={},
            side_effects=True,
            category="observability",
            risk_level=ToolRiskLevel.LOW,
            tags=("braintrust", "observability", "eval"),
        ),
        BraintrustLogEvalHandler(ctx),
    )
