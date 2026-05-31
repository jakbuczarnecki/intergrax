# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.observability.contracts import (
    LogsSearchInput,
    LogsSearchOutput,
    MetricsQueryInstantInput,
    MetricsQueryInstantOutput,
    TracesQueryInput,
    TracesQueryOutput,
)
from intergrax.tools.providers.observability.handlers import (
    LogsSearchHandler,
    MetricsQueryInstantHandler,
    TracesQueryHandler,
)
from intergrax.tools.providers.observability.service import (
    LOGS_SEARCH_TOOL_ID,
    METRICS_QUERY_INSTANT_TOOL_ID,
    TRACES_QUERY_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

OBSERVABILITY_BUNDLE_ID = "observability"
OBSERVABILITY_TOOL_IDS: tuple[str, ...] = (
    METRICS_QUERY_INSTANT_TOOL_ID,
    LOGS_SEARCH_TOOL_ID,
    TRACES_QUERY_TOOL_ID,
)


def register_observability_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=METRICS_QUERY_INSTANT_TOOL_ID,
            name=METRICS_QUERY_INSTANT_TOOL_ID,
            description="Run an instant PromQL metrics query against the configured observability backend.",
            description_short="Query metrics (PromQL).",
            input_schema=MetricsQueryInstantInput,
            output_schema=MetricsQueryInstantOutput,
            error_mapping={},
            side_effects=False,
            category="observability",
            risk_level=ToolRiskLevel.LOW,
            tags=("metrics", "prometheus", "observability"),
        ),
        MetricsQueryInstantHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=LOGS_SEARCH_TOOL_ID,
            name=LOGS_SEARCH_TOOL_ID,
            description="Search log entries using the configured observability backend (Elasticsearch).",
            description_short="Search logs.",
            input_schema=LogsSearchInput,
            output_schema=LogsSearchOutput,
            error_mapping={},
            side_effects=False,
            injects_context=True,
            category="observability",
            risk_level=ToolRiskLevel.LOW,
            tags=("logs", "elasticsearch", "observability"),
        ),
        LogsSearchHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=TRACES_QUERY_TOOL_ID,
            name=TRACES_QUERY_TOOL_ID,
            description="Query recent traces/spans from the configured observability backend (Langfuse, etc.).",
            description_short="Query traces.",
            input_schema=TracesQueryInput,
            output_schema=TracesQueryOutput,
            error_mapping={},
            side_effects=False,
            category="observability",
            risk_level=ToolRiskLevel.LOW,
            tags=("traces", "observability", "langfuse"),
        ),
        TracesQueryHandler(ctx),
    )
