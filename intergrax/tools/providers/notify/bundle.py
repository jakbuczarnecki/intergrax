# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.notify.contracts import (
    NotifySendBatchInput,
    NotifySendBatchOutput,
    NotifySendInput,
    NotifySendOutput,
)
from intergrax.tools.providers.notify.handlers import NotifySendBatchHandler, NotifySendHandler
from intergrax.tools.providers.notify.service import NOTIFY_SEND_BATCH_TOOL_ID, NOTIFY_SEND_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

NOTIFY_BUNDLE_ID = "notify"
NOTIFY_TOOL_IDS: tuple[str, ...] = (NOTIFY_SEND_TOOL_ID, NOTIFY_SEND_BATCH_TOOL_ID)


def register_notify_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=NOTIFY_SEND_TOOL_ID,
            name=NOTIFY_SEND_TOOL_ID,
            description="Send an outbound notification (Slack, Teams, log, webhook — depends on Tier-3 wiring).",
            description_short="Send notification message.",
            input_schema=NotifySendInput,
            output_schema=NotifySendOutput,
            error_mapping={},
            side_effects=True,
            category="notification",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("notify", "notification"),
        ),
        NotifySendHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=NOTIFY_SEND_BATCH_TOOL_ID,
            name=NOTIFY_SEND_BATCH_TOOL_ID,
            description="Send multiple outbound notifications in one catalog call.",
            description_short="Send notification batch.",
            input_schema=NotifySendBatchInput,
            output_schema=NotifySendBatchOutput,
            error_mapping={},
            side_effects=True,
            category="notification",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("notify", "notification", "batch"),
        ),
        NotifySendBatchHandler(ctx),
    )
