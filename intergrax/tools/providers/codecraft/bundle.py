# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.codecraft.contracts import CodeCraftRunToolInput, CodeCraftRunToolOutput
from intergrax.tools.providers.codecraft.handlers import CodeCraftRunHandler
from intergrax.tools.providers.codecraft.service import CODECRAFT_RUN_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

CODECRAFT_BUNDLE_ID = "codecraft"
CODECRAFT_TOOL_IDS: tuple[str, ...] = (CODECRAFT_RUN_TOOL_ID,)


def register_codecraft_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=CODECRAFT_RUN_TOOL_ID,
            name=CODECRAFT_RUN_TOOL_ID,
            description=(
                "Single-shot ephemeral code craft: static L0 gate, policy check, "
                "and sandbox execution when profile mode allows."
            ),
            description_short="Run governed ephemeral code craft.",
            input_schema=CodeCraftRunToolInput,
            output_schema=CodeCraftRunToolOutput,
            error_mapping={},
            side_effects=True,
            category="codecraft",
            risk_level=ToolRiskLevel.HIGH,
            tags=("codecraft", "sandbox", "execution"),
        ),
        CodeCraftRunHandler(ctx),
    )
