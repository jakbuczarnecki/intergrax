# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.codecraft.contracts import (
    CodeCraftDisposeToolInput,
    CodeCraftDisposeToolOutput,
    CodeCraftGetStateToolInput,
    CodeCraftGetStateToolOutput,
    CodeCraftIterateToolInput,
    CodeCraftIterateToolOutput,
    CodeCraftListEphemeralToolsInput,
    CodeCraftListEphemeralToolsOutput,
    CodeCraftPromoteToolInput,
    CodeCraftPromoteToolOutput,
    CodeCraftRunToolInput,
    CodeCraftRunToolOutput,
    CodeCraftStartToolInput,
    CodeCraftStartToolOutput,
)
from intergrax.tools.providers.codecraft.handlers import (
    CodeCraftDisposeHandler,
    CodeCraftGetStateHandler,
    CodeCraftIterateHandler,
    CodeCraftListEphemeralToolsHandler,
    CodeCraftPromoteHandler,
    CodeCraftRunHandler,
    CodeCraftStartHandler,
)
from intergrax.tools.providers.codecraft.service import (
    CODECRAFT_DISPOSE_TOOL_ID,
    CODECRAFT_GET_STATE_TOOL_ID,
    CODECRAFT_ITERATE_TOOL_ID,
    CODECRAFT_LIST_EPHEMERAL_TOOLS_TOOL_ID,
    CODECRAFT_PROMOTE_TOOL_ID,
    CODECRAFT_RUN_TOOL_ID,
    CODECRAFT_START_TOOL_ID,
    CODECRAFT_TOOL_IDS,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

CODECRAFT_BUNDLE_ID = "codecraft"


def register_codecraft_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=CODECRAFT_RUN_TOOL_ID,
            name=CODECRAFT_RUN_TOOL_ID,
            description="Single-shot ephemeral code craft: L0 gate, policy, sandbox exec.",
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
    registry.register(
        ToolContract(
            tool_id=CODECRAFT_START_TOOL_ID,
            name=CODECRAFT_START_TOOL_ID,
            description="Open a code craft session with goal and optional initial code.",
            description_short="Start code craft session.",
            input_schema=CodeCraftStartToolInput,
            output_schema=CodeCraftStartToolOutput,
            error_mapping={},
            side_effects=True,
            category="codecraft",
            risk_level=ToolRiskLevel.HIGH,
            tags=("codecraft", "session"),
        ),
        CodeCraftStartHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CODECRAFT_ITERATE_TOOL_ID,
            name=CODECRAFT_ITERATE_TOOL_ID,
            description="Run one craft iteration: gate, optional HITL, exec, tests, CVL verdict.",
            description_short="Iterate code craft loop.",
            input_schema=CodeCraftIterateToolInput,
            output_schema=CodeCraftIterateToolOutput,
            error_mapping={},
            side_effects=True,
            category="codecraft",
            risk_level=ToolRiskLevel.HIGH,
            tags=("codecraft", "iteration"),
        ),
        CodeCraftIterateHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CODECRAFT_GET_STATE_TOOL_ID,
            name=CODECRAFT_GET_STATE_TOOL_ID,
            description="Return craft session state, iterations, and ephemeral tool ids.",
            description_short="Get code craft state.",
            input_schema=CodeCraftGetStateToolInput,
            output_schema=CodeCraftGetStateToolOutput,
            error_mapping={},
            side_effects=False,
            category="codecraft",
            risk_level=ToolRiskLevel.LOW,
            tags=("codecraft", "introspection"),
        ),
        CodeCraftGetStateHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CODECRAFT_DISPOSE_TOOL_ID,
            name=CODECRAFT_DISPOSE_TOOL_ID,
            description="Dispose craft session and ephemeral registry entries.",
            description_short="Dispose code craft session.",
            input_schema=CodeCraftDisposeToolInput,
            output_schema=CodeCraftDisposeToolOutput,
            error_mapping={},
            side_effects=True,
            category="codecraft",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("codecraft", "cleanup"),
        ),
        CodeCraftDisposeHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CODECRAFT_PROMOTE_TOOL_ID,
            name=CODECRAFT_PROMOTE_TOOL_ID,
            description="Promote craft output with typed schema validation (supervised hosts).",
            description_short="Promote craft result.",
            input_schema=CodeCraftPromoteToolInput,
            output_schema=CodeCraftPromoteToolOutput,
            error_mapping={},
            side_effects=True,
            category="codecraft",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("codecraft", "promotion"),
        ),
        CodeCraftPromoteHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CODECRAFT_LIST_EPHEMERAL_TOOLS_TOOL_ID,
            name=CODECRAFT_LIST_EPHEMERAL_TOOLS_TOOL_ID,
            description="List ephemeral tools registered for an active craft session.",
            description_short="List ephemeral craft tools.",
            input_schema=CodeCraftListEphemeralToolsInput,
            output_schema=CodeCraftListEphemeralToolsOutput,
            error_mapping={},
            side_effects=False,
            category="codecraft",
            risk_level=ToolRiskLevel.LOW,
            tags=("codecraft", "ephemeral"),
        ),
        CodeCraftListEphemeralToolsHandler(ctx),
    )


__all__ = ["CODECRAFT_BUNDLE_ID", "CODECRAFT_TOOL_IDS", "register_codecraft_tools"]
