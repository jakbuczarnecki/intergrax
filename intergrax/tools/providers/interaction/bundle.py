# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.interaction.contracts import (
    InteractionGetLastInputInput,
    InteractionGetLastInputOutput,
    InteractionGetSessionHistoryInput,
    InteractionGetSessionHistoryOutput,
    InteractionListSessionsInput,
    InteractionListSessionsOutput,
)
from intergrax.tools.providers.interaction.handlers import (
    InteractionGetLastInputHandler,
    InteractionGetSessionHistoryHandler,
    InteractionListSessionsHandler,
)
from intergrax.tools.providers.interaction.service import (
    INTERACTION_GET_LAST_INPUT_TOOL_ID,
    INTERACTION_GET_SESSION_HISTORY_TOOL_ID,
    INTERACTION_LIST_SESSIONS_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

INTERACTION_BUNDLE_ID = "interaction"
INTERACTION_TOOL_IDS: tuple[str, ...] = (
    INTERACTION_LIST_SESSIONS_TOOL_ID,
    INTERACTION_GET_LAST_INPUT_TOOL_ID,
    INTERACTION_GET_SESSION_HISTORY_TOOL_ID,
)


def register_interaction_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=INTERACTION_LIST_SESSIONS_TOOL_ID,
            name=INTERACTION_LIST_SESSIONS_TOOL_ID,
            description="List recent chat sessions for a tenant/user via session storage binding.",
            description_short="List interaction sessions.",
            input_schema=InteractionListSessionsInput,
            output_schema=InteractionListSessionsOutput,
            error_mapping={},
            side_effects=False,
            category="interaction",
            risk_level=ToolRiskLevel.LOW,
            tags=("interaction", "session", "read_only"),
        ),
        InteractionListSessionsHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=INTERACTION_GET_LAST_INPUT_TOOL_ID,
            name=INTERACTION_GET_LAST_INPUT_TOOL_ID,
            description="Fetch the last user input message for a session (read-only).",
            description_short="Get last session input.",
            input_schema=InteractionGetLastInputInput,
            output_schema=InteractionGetLastInputOutput,
            error_mapping={},
            side_effects=False,
            category="interaction",
            risk_level=ToolRiskLevel.LOW,
            tags=("interaction", "session", "read_only"),
        ),
        InteractionGetLastInputHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=INTERACTION_GET_SESSION_HISTORY_TOOL_ID,
            name=INTERACTION_GET_SESSION_HISTORY_TOOL_ID,
            description="Fetch recent chat messages for a session (read-only).",
            description_short="Get session history.",
            input_schema=InteractionGetSessionHistoryInput,
            output_schema=InteractionGetSessionHistoryOutput,
            error_mapping={},
            side_effects=False,
            category="interaction",
            risk_level=ToolRiskLevel.LOW,
            tags=("interaction", "session", "read_only"),
        ),
        InteractionGetSessionHistoryHandler(ctx),
    )
