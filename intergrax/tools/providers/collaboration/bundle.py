# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.collaboration.contracts import CollaborationSendMailInput, CollaborationSendMailOutput
from intergrax.tools.providers.collaboration.handlers import CollaborationSendMailHandler
from intergrax.tools.providers.collaboration.service import COLLABORATION_SEND_MAIL_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

COLLABORATION_BUNDLE_ID = "collaboration"
COLLABORATION_TOOL_IDS: tuple[str, ...] = (COLLABORATION_SEND_MAIL_TOOL_ID,)


def register_collaboration_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=COLLABORATION_SEND_MAIL_TOOL_ID,
            name=COLLABORATION_SEND_MAIL_TOOL_ID,
            description="Send mail via the configured collaboration suite (M365, Google Workspace, …).",
            description_short="Send mail.",
            input_schema=CollaborationSendMailInput,
            output_schema=CollaborationSendMailOutput,
            error_mapping={},
            side_effects=True,
            category="collaboration",
            risk_level=ToolRiskLevel.HIGH,
            tags=("collaboration", "mail"),
        ),
        CollaborationSendMailHandler(ctx),
    )
