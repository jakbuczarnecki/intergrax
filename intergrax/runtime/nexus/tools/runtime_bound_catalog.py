# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Dispatch runtime-bound catalog tools from UAEP ``RuntimeExecutionContext`` (§42.12)."""

from __future__ import annotations

import time
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
from intergrax.tools.providers.cost.contracts import (
    CostCheckQuotaInput,
    CostForecastSpendInput,
    CostGetRunBudgetInput,
)
from intergrax.tools.providers.cost.service import (
    COST_CHECK_QUOTA_TOOL_ID,
    COST_FORECAST_SPEND_TOOL_ID,
    COST_GET_RUN_BUDGET_TOOL_ID,
    cost_check_quota,
    cost_forecast_spend,
    cost_get_run_budget,
)
from intergrax.tools.providers.harness.contracts import (
    HarnessCompareRunsInput,
    HarnessExportRunBundleInput,
    HarnessGetRunCostInput,
    HarnessGetRunEventsInput,
    HarnessGetRunInput,
    HarnessListRunsInput,
)
from intergrax.tools.providers.harness.service import (
    HARNESS_COMPARE_RUNS_TOOL_ID,
    HARNESS_EXPORT_RUN_BUNDLE_TOOL_ID,
    HARNESS_GET_RUN_COST_TOOL_ID,
    HARNESS_GET_RUN_EVENTS_TOOL_ID,
    HARNESS_GET_RUN_TOOL_ID,
    HARNESS_LIST_RUNS_TOOL_ID,
    harness_compare_runs,
    harness_export_run_bundle,
    harness_get_run,
    harness_get_run_cost,
    harness_get_run_events,
    harness_list_runs,
)
from intergrax.tools.providers.memory.contracts import MemoryListKeysInput, MemoryReadInput, MemoryWriteInput
from intergrax.tools.providers.memory.service import (
    MEMORY_LIST_KEYS_TOOL_ID,
    MEMORY_READ_TOOL_ID,
    MEMORY_WRITE_TOOL_ID,
    memory_list_keys,
    memory_read,
    memory_write,
)
from intergrax.tools.providers.workspace.contracts import (
    WorkspaceDeleteFileInput,
    WorkspaceListFilesInput,
    WorkspaceReadFileInput,
    WorkspaceSearchInput,
    WorkspaceSnapshotInput,
    WorkspaceWriteFileInput,
)
from intergrax.tools.providers.workspace.service import (
    WORKSPACE_DELETE_FILE_TOOL_ID,
    WORKSPACE_LIST_FILES_TOOL_ID,
    WORKSPACE_READ_FILE_TOOL_ID,
    WORKSPACE_SEARCH_TOOL_ID,
    WORKSPACE_SNAPSHOT_TOOL_ID,
    WORKSPACE_WRITE_FILE_TOOL_ID,
    workspace_delete_file,
    workspace_list_files,
    workspace_read_file,
    workspace_search,
    workspace_snapshot,
    workspace_write_file,
)
from intergrax.tools.registry.wiring import ToolWiringContext

ServiceFn = Callable[[ToolWiringContext, BaseModel], BaseModel]

_RUNTIME_BOUND_TOOLS: dict[str, tuple[type[BaseModel], ServiceFn]] = {
    WORKSPACE_WRITE_FILE_TOOL_ID: (WorkspaceWriteFileInput, workspace_write_file),
    WORKSPACE_READ_FILE_TOOL_ID: (WorkspaceReadFileInput, workspace_read_file),
    WORKSPACE_LIST_FILES_TOOL_ID: (WorkspaceListFilesInput, workspace_list_files),
    WORKSPACE_SNAPSHOT_TOOL_ID: (WorkspaceSnapshotInput, workspace_snapshot),
    WORKSPACE_DELETE_FILE_TOOL_ID: (WorkspaceDeleteFileInput, workspace_delete_file),
    WORKSPACE_SEARCH_TOOL_ID: (WorkspaceSearchInput, workspace_search),
    MEMORY_READ_TOOL_ID: (MemoryReadInput, memory_read),
    MEMORY_WRITE_TOOL_ID: (MemoryWriteInput, memory_write),
    MEMORY_LIST_KEYS_TOOL_ID: (MemoryListKeysInput, memory_list_keys),
    HARNESS_GET_RUN_TOOL_ID: (HarnessGetRunInput, harness_get_run),
    HARNESS_LIST_RUNS_TOOL_ID: (HarnessListRunsInput, harness_list_runs),
    HARNESS_GET_RUN_COST_TOOL_ID: (HarnessGetRunCostInput, harness_get_run_cost),
    HARNESS_GET_RUN_EVENTS_TOOL_ID: (HarnessGetRunEventsInput, harness_get_run_events),
    HARNESS_COMPARE_RUNS_TOOL_ID: (HarnessCompareRunsInput, harness_compare_runs),
    HARNESS_EXPORT_RUN_BUNDLE_TOOL_ID: (HarnessExportRunBundleInput, harness_export_run_bundle),
    COST_GET_RUN_BUDGET_TOOL_ID: (CostGetRunBudgetInput, cost_get_run_budget),
    COST_CHECK_QUOTA_TOOL_ID: (CostCheckQuotaInput, cost_check_quota),
    COST_FORECAST_SPEND_TOOL_ID: (CostForecastSpendInput, cost_forecast_spend),
}

RUNTIME_BOUND_TOOL_IDS: frozenset[str] = frozenset(_RUNTIME_BOUND_TOOLS.keys())


def is_runtime_bound_tool(tool_name: str) -> bool:
    return tool_name in RUNTIME_BOUND_TOOL_IDS


def build_runtime_bound_context(exec_ctx: RuntimeExecutionContext) -> ToolWiringContext:
    workspace = exec_ctx.metadata.get("shadow_workspace")
    shadow: ShadowWorkspace | None = workspace if isinstance(workspace, ShadowWorkspace) else None
    trace_reader: RunTraceReader | None = None
    for key in ("trace_reader", "trace_store"):
        candidate = exec_ctx.metadata.get(key)
        if isinstance(candidate, RunTraceReader):
            trace_reader = candidate
            break
    run_budget = exec_ctx.metadata.get("run_budget")
    budget: RunBudget | None = run_budget if isinstance(run_budget, RunBudget) else None
    cost_envelopes = exec_ctx.metadata.get("cost_envelopes", ())
    cost_quotas = exec_ctx.metadata.get("cost_quotas", ())
    extras: dict[str, object] = {}
    request = exec_ctx.request
    if request is not None and request.metadata:
        extras["task_metadata"] = dict(request.metadata)
    return ToolWiringContext(
        shadow_workspace=shadow,
        memory_view=exec_ctx.memory_view,
        trace_reader=trace_reader,
        run_budget=budget,
        cost_envelopes=tuple(cost_envelopes) if isinstance(cost_envelopes, (list, tuple)) else (),
        cost_quotas=tuple(cost_quotas) if isinstance(cost_quotas, (list, tuple)) else (),
        extras=extras,
    )


def invoke_runtime_bound_tool(exec_ctx: RuntimeExecutionContext, request: ToolRequest) -> ToolResponse:
    started = time.perf_counter()
    entry = _RUNTIME_BOUND_TOOLS.get(request.tool_name)
    if entry is None:
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.FAILED,
            error=f"runtime_bound_tool_unknown:{request.tool_name}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    input_schema, service = entry
    try:
        params = input_schema.model_validate(dict(request.input or {}))
    except ValidationError as exc:
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.FAILED,
            error=f"validation_error:{exc}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    ctx = build_runtime_bound_context(exec_ctx)
    try:
        output = service(ctx, params)
    except Exception as exc:  # noqa: BLE001 — gateway boundary
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.FAILED,
            error=f"{type(exc).__name__}:{exc}",
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    payload: dict[str, Any]
    if isinstance(output, BaseModel):
        payload = output.model_dump()
    else:
        payload = {"result": output}

    return ToolResponse(
        request_id=request.request_id,
        status=ToolResponseStatus.SUCCESS,
        output=payload,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
