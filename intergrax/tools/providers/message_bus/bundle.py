# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.message_bus.contracts import (
    MessageBusEnqueueInput,
    MessageBusEnqueueOutput,
    MessageBusGetResultInput,
    MessageBusGetResultOutput,
    MessageBusGetStatusInput,
    MessageBusGetStatusOutput,
)
from intergrax.tools.providers.message_bus.handlers import (
    MessageBusEnqueueHandler,
    MessageBusGetResultHandler,
    MessageBusGetStatusHandler,
)
from intergrax.tools.providers.message_bus.service import (
    MESSAGE_BUS_ENQUEUE_TOOL_ID,
    MESSAGE_BUS_GET_RESULT_TOOL_ID,
    MESSAGE_BUS_GET_STATUS_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

MESSAGE_BUS_BUNDLE_ID = "message_bus"
MESSAGE_BUS_TOOL_IDS: tuple[str, ...] = (
    MESSAGE_BUS_ENQUEUE_TOOL_ID,
    MESSAGE_BUS_GET_STATUS_TOOL_ID,
    MESSAGE_BUS_GET_RESULT_TOOL_ID,
)


def register_message_bus_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=MESSAGE_BUS_ENQUEUE_TOOL_ID,
            name=MESSAGE_BUS_ENQUEUE_TOOL_ID,
            description="Enqueue an asynchronous task on the configured message bus / task queue.",
            description_short="Enqueue async task.",
            input_schema=MessageBusEnqueueInput,
            output_schema=MessageBusEnqueueOutput,
            error_mapping={},
            side_effects=True,
            category="message_bus",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("message_bus", "async"),
        ),
        MessageBusEnqueueHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=MESSAGE_BUS_GET_STATUS_TOOL_ID,
            name=MESSAGE_BUS_GET_STATUS_TOOL_ID,
            description="Poll task status from the message bus by task handle.",
            description_short="Get task status.",
            input_schema=MessageBusGetStatusInput,
            output_schema=MessageBusGetStatusOutput,
            error_mapping={},
            side_effects=False,
            category="message_bus",
            risk_level=ToolRiskLevel.LOW,
            tags=("message_bus", "async"),
        ),
        MessageBusGetStatusHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=MESSAGE_BUS_GET_RESULT_TOOL_ID,
            name=MESSAGE_BUS_GET_RESULT_TOOL_ID,
            description="Fetch final task result from the message bus when completed.",
            description_short="Get task result.",
            input_schema=MessageBusGetResultInput,
            output_schema=MessageBusGetResultOutput,
            error_mapping={},
            side_effects=False,
            category="message_bus",
            risk_level=ToolRiskLevel.LOW,
            tags=("message_bus", "async"),
        ),
        MessageBusGetResultHandler(ctx),
    )
