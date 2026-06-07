# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.billing.contracts import (
    BillingListUsageInput,
    BillingListUsageOutput,
    BillingRecordUsageInput,
    BillingRecordUsageOutput,
)
from intergrax.tools.providers.billing.handlers import BillingListUsageHandler, BillingRecordUsageHandler
from intergrax.tools.providers.billing.service import BILLING_LIST_USAGE_TOOL_ID, BILLING_RECORD_USAGE_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

BILLING_BUNDLE_ID = "billing"
BILLING_TOOL_IDS: tuple[str, ...] = (BILLING_RECORD_USAGE_TOOL_ID, BILLING_LIST_USAGE_TOOL_ID)


def register_billing_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=BILLING_RECORD_USAGE_TOOL_ID,
            name=BILLING_RECORD_USAGE_TOOL_ID,
            description="Submit a usage metering event to the configured billing meter backend.",
            description_short="Record meter usage.",
            input_schema=BillingRecordUsageInput,
            output_schema=BillingRecordUsageOutput,
            error_mapping={},
            side_effects=True,
            category="billing",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("billing", "metering", "cost"),
        ),
        BillingRecordUsageHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=BILLING_LIST_USAGE_TOOL_ID,
            name=BILLING_LIST_USAGE_TOOL_ID,
            description="List recent usage metering events for a customer id.",
            description_short="List meter usage.",
            input_schema=BillingListUsageInput,
            output_schema=BillingListUsageOutput,
            error_mapping={},
            side_effects=False,
            category="billing",
            risk_level=ToolRiskLevel.LOW,
            tags=("billing", "metering", "cost"),
        ),
        BillingListUsageHandler(ctx),
    )
