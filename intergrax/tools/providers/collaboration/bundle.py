# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.collaboration.contracts import (
    CollaborationGetMessageInput,
    CollaborationGetMessageOutput,
    CollaborationGetUserInput,
    CollaborationGetUserOutput,
    CollaborationListCalendarInput,
    CollaborationListCalendarOutput,
    CollaborationListMessagesInput,
    CollaborationListMessagesOutput,
    CollaborationSendMailInput,
    CollaborationSendMailOutput,
)
from intergrax.tools.providers.collaboration.handlers import (
    CollaborationGetMessageHandler,
    CollaborationGetUserHandler,
    CollaborationListCalendarHandler,
    CollaborationListMessagesHandler,
    CollaborationSendMailHandler,
)
from intergrax.tools.providers.collaboration.service import (
    COLLABORATION_GET_MESSAGE_TOOL_ID,
    COLLABORATION_GET_USER_TOOL_ID,
    COLLABORATION_LIST_CALENDAR_TOOL_ID,
    COLLABORATION_LIST_MESSAGES_TOOL_ID,
    COLLABORATION_SEND_MAIL_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

COLLABORATION_BUNDLE_ID = "collaboration"
COLLABORATION_TOOL_IDS: tuple[str, ...] = (
    COLLABORATION_SEND_MAIL_TOOL_ID,
    COLLABORATION_LIST_MESSAGES_TOOL_ID,
    COLLABORATION_GET_MESSAGE_TOOL_ID,
    COLLABORATION_LIST_CALENDAR_TOOL_ID,
    COLLABORATION_GET_USER_TOOL_ID,
)


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
    registry.register(
        ToolContract(
            tool_id=COLLABORATION_LIST_MESSAGES_TOOL_ID,
            name=COLLABORATION_LIST_MESSAGES_TOOL_ID,
            description="List mail messages from a folder in the configured collaboration suite.",
            description_short="List mail messages.",
            input_schema=CollaborationListMessagesInput,
            output_schema=CollaborationListMessagesOutput,
            error_mapping={},
            side_effects=False,
            category="collaboration",
            risk_level=ToolRiskLevel.LOW,
            tags=("collaboration", "mail"),
        ),
        CollaborationListMessagesHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=COLLABORATION_GET_MESSAGE_TOOL_ID,
            name=COLLABORATION_GET_MESSAGE_TOOL_ID,
            description="Fetch a single mail message from the configured collaboration suite.",
            description_short="Get mail message.",
            input_schema=CollaborationGetMessageInput,
            output_schema=CollaborationGetMessageOutput,
            error_mapping={},
            side_effects=False,
            category="collaboration",
            risk_level=ToolRiskLevel.LOW,
            tags=("collaboration", "mail"),
        ),
        CollaborationGetMessageHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=COLLABORATION_LIST_CALENDAR_TOOL_ID,
            name=COLLABORATION_LIST_CALENDAR_TOOL_ID,
            description="List calendar events in an ISO8601 time window via the collaboration suite.",
            description_short="List calendar events.",
            input_schema=CollaborationListCalendarInput,
            output_schema=CollaborationListCalendarOutput,
            error_mapping={},
            side_effects=False,
            category="collaboration",
            risk_level=ToolRiskLevel.LOW,
            tags=("collaboration", "calendar"),
        ),
        CollaborationListCalendarHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=COLLABORATION_GET_USER_TOOL_ID,
            name=COLLABORATION_GET_USER_TOOL_ID,
            description="Resolve a directory user record from the configured collaboration suite.",
            description_short="Get directory user.",
            input_schema=CollaborationGetUserInput,
            output_schema=CollaborationGetUserOutput,
            error_mapping={},
            side_effects=False,
            category="collaboration",
            risk_level=ToolRiskLevel.LOW,
            tags=("collaboration", "directory"),
        ),
        CollaborationGetUserHandler(ctx),
    )
