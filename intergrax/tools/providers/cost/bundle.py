# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.cost.contracts import (
    CostCheckQuotaInput,
    CostCheckQuotaOutput,
    CostForecastSpendInput,
    CostForecastSpendOutput,
    CostGetRunBudgetInput,
    CostGetRunBudgetOutput,
)
from intergrax.tools.providers.cost.handlers import (
    CostCheckQuotaHandler,
    CostForecastSpendHandler,
    CostGetRunBudgetHandler,
)
from intergrax.tools.providers.cost.service import (
    COST_CHECK_QUOTA_TOOL_ID,
    COST_FORECAST_SPEND_TOOL_ID,
    COST_GET_RUN_BUDGET_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

COST_BUNDLE_ID = "cost"
COST_TOOL_IDS: tuple[str, ...] = (
    COST_GET_RUN_BUDGET_TOOL_ID,
    COST_CHECK_QUOTA_TOOL_ID,
    COST_FORECAST_SPEND_TOOL_ID,
)


def register_cost_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=COST_GET_RUN_BUDGET_TOOL_ID,
            name=COST_GET_RUN_BUDGET_TOOL_ID,
            description="Read per-run budget limits and optional tenant envelope status (V-COST).",
            description_short="Get run budget.",
            input_schema=CostGetRunBudgetInput,
            output_schema=CostGetRunBudgetOutput,
            error_mapping={},
            side_effects=False,
            category="cost",
            risk_level=ToolRiskLevel.LOW,
            tags=("cost", "budget", "governance"),
        ),
        CostGetRunBudgetHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=COST_CHECK_QUOTA_TOOL_ID,
            name=COST_CHECK_QUOTA_TOOL_ID,
            description="Evaluate a quota usage request against configured resource quotas (V-COST).",
            description_short="Check resource quota.",
            input_schema=CostCheckQuotaInput,
            output_schema=CostCheckQuotaOutput,
            error_mapping={},
            side_effects=False,
            category="cost",
            risk_level=ToolRiskLevel.LOW,
            tags=("cost", "quota", "governance"),
        ),
        CostCheckQuotaHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=COST_FORECAST_SPEND_TOOL_ID,
            name=COST_FORECAST_SPEND_TOOL_ID,
            description="Forecast spend drift and anomalies from configured budget envelopes (V-COST.3).",
            description_short="Forecast spend drift.",
            input_schema=CostForecastSpendInput,
            output_schema=CostForecastSpendOutput,
            error_mapping={},
            side_effects=False,
            category="cost",
            risk_level=ToolRiskLevel.LOW,
            tags=("cost", "forecast", "governance"),
        ),
        CostForecastSpendHandler(ctx),
    )
