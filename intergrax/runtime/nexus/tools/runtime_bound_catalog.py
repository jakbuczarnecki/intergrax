# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime-bound tool id catalog (classification only — physical dispatch via Tool Engine)."""

from __future__ import annotations

from intergrax.tools.providers.cost.tool_ids import (
    COST_CHECK_QUOTA_TOOL_ID,
    COST_FORECAST_SPEND_TOOL_ID,
    COST_GET_RUN_BUDGET_TOOL_ID,
)
from intergrax.tools.providers.harness.tool_ids import (
    HARNESS_COMPARE_RUNS_TOOL_ID,
    HARNESS_EXPORT_RUN_BUNDLE_TOOL_ID,
    HARNESS_GET_RUN_COST_TOOL_ID,
    HARNESS_GET_RUN_EVENTS_TOOL_ID,
    HARNESS_GET_RUN_TOOL_ID,
    HARNESS_LIST_RUNS_TOOL_ID,
)
from intergrax.tools.providers.memory.tool_ids import (
    MEMORY_LIST_KEYS_TOOL_ID,
    MEMORY_READ_TOOL_ID,
    MEMORY_WRITE_TOOL_ID,
)
from intergrax.tools.providers.workspace.tool_ids import (
    WORKSPACE_DELETE_FILE_TOOL_ID,
    WORKSPACE_LIST_FILES_TOOL_ID,
    WORKSPACE_READ_FILE_TOOL_ID,
    WORKSPACE_SEARCH_TOOL_ID,
    WORKSPACE_SNAPSHOT_TOOL_ID,
    WORKSPACE_WRITE_FILE_TOOL_ID,
)

RUNTIME_BOUND_TOOL_IDS: frozenset[str] = frozenset(
    {
        WORKSPACE_WRITE_FILE_TOOL_ID,
        WORKSPACE_READ_FILE_TOOL_ID,
        WORKSPACE_LIST_FILES_TOOL_ID,
        WORKSPACE_SNAPSHOT_TOOL_ID,
        WORKSPACE_DELETE_FILE_TOOL_ID,
        WORKSPACE_SEARCH_TOOL_ID,
        MEMORY_READ_TOOL_ID,
        MEMORY_WRITE_TOOL_ID,
        MEMORY_LIST_KEYS_TOOL_ID,
        HARNESS_GET_RUN_TOOL_ID,
        HARNESS_LIST_RUNS_TOOL_ID,
        HARNESS_GET_RUN_COST_TOOL_ID,
        HARNESS_GET_RUN_EVENTS_TOOL_ID,
        HARNESS_COMPARE_RUNS_TOOL_ID,
        HARNESS_EXPORT_RUN_BUNDLE_TOOL_ID,
        COST_GET_RUN_BUDGET_TOOL_ID,
        COST_CHECK_QUOTA_TOOL_ID,
        COST_FORECAST_SPEND_TOOL_ID,
    }
)


def is_runtime_bound_tool(tool_name: str) -> bool:
    return tool_name in RUNTIME_BOUND_TOOL_IDS
