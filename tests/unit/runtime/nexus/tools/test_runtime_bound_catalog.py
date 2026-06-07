# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.architecture.cost_budget import BudgetEnvelope, BudgetScope
from intergrax.runtime.nexus.tools.runtime_bound_catalog import (
    RUNTIME_BOUND_TOOL_IDS,
    invoke_runtime_bound_tool,
)
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace

pytestmark = pytest.mark.unit


@pytest.fixture
def exec_ctx(tmp_path: Path) -> RuntimeExecutionContext:
    workspace = ShadowWorkspace.create(tmp_path, tenant_id="t1", task_id="task-1")
    return RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="agent-1",
        metadata={"shadow_workspace": workspace},
    )


def test_runtime_bound_workspace_write_read(exec_ctx: RuntimeExecutionContext) -> None:
    write = invoke_runtime_bound_tool(
        exec_ctx,
        ToolRequest(
            request_id="req-1",
            tool_name="workspace.write_file",
            agent_id="agent-1",
            input={"path": "out.txt", "content": "bound path"},
        ),
    )
    assert write.status == ToolResponseStatus.SUCCESS
    read = invoke_runtime_bound_tool(
        exec_ctx,
        ToolRequest(
            request_id="req-2",
            tool_name="workspace.read_file",
            agent_id="agent-1",
            input={"path": "out.txt"},
        ),
    )
    assert read.status == ToolResponseStatus.SUCCESS
    assert read.output is not None
    assert read.output["content"] == "bound path"


def test_runtime_bound_cost_forecast_spend(exec_ctx: RuntimeExecutionContext) -> None:
    exec_ctx.metadata["cost_envelopes"] = (
        BudgetEnvelope(scope=BudgetScope.TENANT, scope_id="t-1", limit_amount=100.0, spent_amount=50.0),
    )
    response = invoke_runtime_bound_tool(
        exec_ctx,
        ToolRequest(
            request_id="req-cost-1",
            tool_name="cost.forecast_spend",
            agent_id="agent-1",
            input={"growth_multiplier": 1.1},
        ),
    )
    assert response.status == ToolResponseStatus.SUCCESS
    assert response.output is not None
    assert len(response.output["forecasts"]) == 1


def test_runtime_bound_catalog_includes_cost_forecast() -> None:
    assert "cost.forecast_spend" in RUNTIME_BOUND_TOOL_IDS
