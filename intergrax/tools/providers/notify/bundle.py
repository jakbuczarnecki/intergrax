# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.notify.contracts import (
    NotifyCancelScheduledInput,
    NotifyCancelScheduledOutput,
    NotifyListScheduledInput,
    NotifyListScheduledOutput,
    NotifyScheduleInput,
    NotifyScheduleOutput,
    NotifySendBatchInput,
    NotifySendBatchOutput,
    NotifySendInput,
    NotifySendOutput,
)
from intergrax.tools.providers.notify.handlers import (
    NotifyCancelScheduledHandler,
    NotifyListScheduledHandler,
    NotifyScheduleHandler,
    NotifySendBatchHandler,
    NotifySendHandler,
)
from intergrax.tools.providers.notify.service import (
    NOTIFY_CANCEL_SCHEDULED_TOOL_ID,
    NOTIFY_LIST_SCHEDULED_TOOL_ID,
    NOTIFY_SCHEDULE_TOOL_ID,
    NOTIFY_SEND_BATCH_TOOL_ID,
    NOTIFY_SEND_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

NOTIFY_BUNDLE_ID = "notify"
NOTIFY_TOOL_IDS: tuple[str, ...] = (
    NOTIFY_SEND_TOOL_ID,
    NOTIFY_SEND_BATCH_TOOL_ID,
    NOTIFY_SCHEDULE_TOOL_ID,
    NOTIFY_LIST_SCHEDULED_TOOL_ID,
    NOTIFY_CANCEL_SCHEDULED_TOOL_ID,
)


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
    registry.register(
        ToolContract(
            tool_id=NOTIFY_SCHEDULE_TOOL_ID,
            name=NOTIFY_SCHEDULE_TOOL_ID,
            description="Schedule an outbound notification for future delivery via the configured schedule store.",
            description_short="Schedule notification.",
            input_schema=NotifyScheduleInput,
            output_schema=NotifyScheduleOutput,
            error_mapping={},
            side_effects=True,
            category="notification",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("notify", "notification", "schedule"),
        ),
        NotifyScheduleHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=NOTIFY_LIST_SCHEDULED_TOOL_ID,
            name=NOTIFY_LIST_SCHEDULED_TOOL_ID,
            description="List deferred notifications recorded by notify.schedule.",
            description_short="List scheduled notifications.",
            input_schema=NotifyListScheduledInput,
            output_schema=NotifyListScheduledOutput,
            error_mapping={},
            side_effects=False,
            category="notification",
            risk_level=ToolRiskLevel.LOW,
            tags=("notify", "notification", "schedule", "read_only"),
        ),
        NotifyListScheduledHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=NOTIFY_CANCEL_SCHEDULED_TOOL_ID,
            name=NOTIFY_CANCEL_SCHEDULED_TOOL_ID,
            description="Cancel a pending deferred notification schedule entry.",
            description_short="Cancel scheduled notification.",
            input_schema=NotifyCancelScheduledInput,
            output_schema=NotifyCancelScheduledOutput,
            error_mapping={},
            side_effects=True,
            category="notification",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("notify", "notification", "schedule"),
        ),
        NotifyCancelScheduledHandler(ctx),
    )
