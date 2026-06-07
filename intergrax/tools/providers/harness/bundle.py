# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.harness.contracts import (
    HarnessCompareRunsInput,
    HarnessCompareRunsOutput,
    HarnessExportRunBundleInput,
    HarnessExportRunBundleOutput,
    HarnessGetRunCostInput,
    HarnessGetRunCostOutput,
    HarnessGetRunEventsInput,
    HarnessGetRunEventsOutput,
    HarnessGetRunInput,
    HarnessGetRunOutput,
    HarnessListRunsInput,
    HarnessListRunsOutput,
)
from intergrax.tools.providers.harness.handlers import (
    HarnessCompareRunsHandler,
    HarnessExportRunBundleHandler,
    HarnessGetRunCostHandler,
    HarnessGetRunEventsHandler,
    HarnessGetRunHandler,
    HarnessListRunsHandler,
)
from intergrax.tools.providers.harness.service import (
    HARNESS_COMPARE_RUNS_TOOL_ID,
    HARNESS_EXPORT_RUN_BUNDLE_TOOL_ID,
    HARNESS_GET_RUN_COST_TOOL_ID,
    HARNESS_GET_RUN_EVENTS_TOOL_ID,
    HARNESS_GET_RUN_TOOL_ID,
    HARNESS_LIST_RUNS_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

HARNESS_BUNDLE_ID = "harness"
HARNESS_TOOL_IDS: tuple[str, ...] = (
    HARNESS_GET_RUN_TOOL_ID,
    HARNESS_LIST_RUNS_TOOL_ID,
    HARNESS_GET_RUN_COST_TOOL_ID,
    HARNESS_GET_RUN_EVENTS_TOOL_ID,
    HARNESS_COMPARE_RUNS_TOOL_ID,
    HARNESS_EXPORT_RUN_BUNDLE_TOOL_ID,
)


def register_harness_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=HARNESS_GET_RUN_TOOL_ID,
            name=HARNESS_GET_RUN_TOOL_ID,
            description="Read a persisted harness run trace scoped by tenant (prod audit / RCA).",
            description_short="Get persisted run trace.",
            input_schema=HarnessGetRunInput,
            output_schema=HarnessGetRunOutput,
            error_mapping={},
            side_effects=False,
            category="harness",
            risk_level=ToolRiskLevel.LOW,
            tags=("harness", "trace", "observability"),
        ),
        HarnessGetRunHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HARNESS_LIST_RUNS_TOOL_ID,
            name=HARNESS_LIST_RUNS_TOOL_ID,
            description="List recent persisted harness runs for a tenant.",
            description_short="List persisted runs.",
            input_schema=HarnessListRunsInput,
            output_schema=HarnessListRunsOutput,
            error_mapping={},
            side_effects=False,
            category="harness",
            risk_level=ToolRiskLevel.LOW,
            tags=("harness", "trace", "observability"),
        ),
        HarnessListRunsHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HARNESS_GET_RUN_COST_TOOL_ID,
            name=HARNESS_GET_RUN_COST_TOOL_ID,
            description="Return LLM usage and duration stats for a persisted harness run (V-COST).",
            description_short="Get run cost stats.",
            input_schema=HarnessGetRunCostInput,
            output_schema=HarnessGetRunCostOutput,
            error_mapping={},
            side_effects=False,
            category="harness",
            risk_level=ToolRiskLevel.LOW,
            tags=("harness", "cost", "trace"),
        ),
        HarnessGetRunCostHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HARNESS_GET_RUN_EVENTS_TOOL_ID,
            name=HARNESS_GET_RUN_EVENTS_TOOL_ID,
            description="Return filtered trace events for a persisted harness run.",
            description_short="Get run trace events.",
            input_schema=HarnessGetRunEventsInput,
            output_schema=HarnessGetRunEventsOutput,
            error_mapping={},
            side_effects=False,
            category="harness",
            risk_level=ToolRiskLevel.LOW,
            tags=("harness", "trace", "events"),
        ),
        HarnessGetRunEventsHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HARNESS_COMPARE_RUNS_TOOL_ID,
            name=HARNESS_COMPARE_RUNS_TOOL_ID,
            description="Compare persisted harness runs (duration, events, LLM usage).",
            description_short="Compare persisted runs.",
            input_schema=HarnessCompareRunsInput,
            output_schema=HarnessCompareRunsOutput,
            error_mapping={},
            side_effects=False,
            category="harness",
            risk_level=ToolRiskLevel.LOW,
            tags=("harness", "trace", "compare"),
        ),
        HarnessCompareRunsHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HARNESS_EXPORT_RUN_BUNDLE_TOOL_ID,
            name=HARNESS_EXPORT_RUN_BUNDLE_TOOL_ID,
            description="Export a persisted run trace bundle as JSON for RCA or sharing.",
            description_short="Export run trace bundle.",
            input_schema=HarnessExportRunBundleInput,
            output_schema=HarnessExportRunBundleOutput,
            error_mapping={},
            side_effects=False,
            category="harness",
            risk_level=ToolRiskLevel.LOW,
            tags=("harness", "trace", "export"),
        ),
        HarnessExportRunBundleHandler(ctx),
    )
