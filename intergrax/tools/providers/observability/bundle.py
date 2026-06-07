# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.observability.contracts import (
    ErrorsCaptureInput,
    ErrorsCaptureOutput,
    LogsSearchInput,
    LogsSearchOutput,
    LogsTailInput,
    LogsTailOutput,
    MetricsQueryInstantInput,
    MetricsQueryInstantOutput,
    MetricsQueryRangeInput,
    MetricsQueryRangeOutput,
    TracesQueryInput,
    TracesQueryOutput,
)
from intergrax.tools.providers.observability.handlers import (
    ErrorsCaptureHandler,
    LogsSearchHandler,
    LogsTailHandler,
    MetricsQueryInstantHandler,
    MetricsQueryRangeHandler,
    TracesQueryHandler,
)
from intergrax.tools.providers.observability.service import (
    ERRORS_CAPTURE_TOOL_ID,
    LOGS_SEARCH_TOOL_ID,
    LOGS_TAIL_TOOL_ID,
    METRICS_QUERY_INSTANT_TOOL_ID,
    METRICS_QUERY_RANGE_TOOL_ID,
    TRACES_QUERY_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

OBSERVABILITY_BUNDLE_ID = "observability"
OBSERVABILITY_TOOL_IDS: tuple[str, ...] = (
    METRICS_QUERY_INSTANT_TOOL_ID,
    METRICS_QUERY_RANGE_TOOL_ID,
    LOGS_SEARCH_TOOL_ID,
    LOGS_TAIL_TOOL_ID,
    TRACES_QUERY_TOOL_ID,
    ERRORS_CAPTURE_TOOL_ID,
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
            tool_id=METRICS_QUERY_RANGE_TOOL_ID,
            name=METRICS_QUERY_RANGE_TOOL_ID,
            description="Run a range PromQL metrics query against the configured observability backend.",
            description_short="Query metrics range.",
            input_schema=MetricsQueryRangeInput,
            output_schema=MetricsQueryRangeOutput,
            error_mapping={},
            side_effects=False,
            category="observability",
            risk_level=ToolRiskLevel.LOW,
            tags=("metrics", "prometheus", "observability"),
        ),
        MetricsQueryRangeHandler(ctx),
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
            tool_id=LOGS_TAIL_TOOL_ID,
            name=LOGS_TAIL_TOOL_ID,
            description="Tail recent log entries using the configured observability backend (Elasticsearch).",
            description_short="Tail recent logs.",
            input_schema=LogsTailInput,
            output_schema=LogsTailOutput,
            error_mapping={},
            side_effects=False,
            injects_context=True,
            category="observability",
            risk_level=ToolRiskLevel.LOW,
            tags=("logs", "elasticsearch", "observability"),
        ),
        LogsTailHandler(ctx),
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
    registry.register(
        ToolContract(
            tool_id=ERRORS_CAPTURE_TOOL_ID,
            name=ERRORS_CAPTURE_TOOL_ID,
            description="Capture an error or diagnostic event via the configured observability backend (Sentry, etc.).",
            description_short="Report error event.",
            input_schema=ErrorsCaptureInput,
            output_schema=ErrorsCaptureOutput,
            error_mapping={},
            side_effects=True,
            category="observability",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("errors", "sentry", "observability"),
        ),
        ErrorsCaptureHandler(ctx),
    )
