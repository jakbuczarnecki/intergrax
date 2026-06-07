# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.workspace.bundle import register_workspace_tools
from intergrax.tools.providers.workspace.contracts import WorkspaceReadFileInput, WorkspaceWriteFileInput
from intergrax.tools.providers.workspace.service import workspace_read_file, workspace_write_file
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


@pytest.fixture
def shadow_workspace(tmp_path: Path) -> ShadowWorkspace:
    return ShadowWorkspace.create(tmp_path, tenant_id="tenant-1", task_id="task-1")


def test_workspace_write_and_read(shadow_workspace: ShadowWorkspace) -> None:
    ctx = ToolWiringContext(shadow_workspace=shadow_workspace)
    workspace_write_file(ctx, WorkspaceWriteFileInput(path="notes/summary.txt", content="hello workspace"))
    out = workspace_read_file(ctx, WorkspaceReadFileInput(path="notes/summary.txt"))
    assert out.content == "hello workspace"
    assert out.workspace_id == shadow_workspace.workspace_id


def test_workspace_delete_and_search(shadow_workspace: ShadowWorkspace) -> None:
    from intergrax.tools.providers.workspace.contracts import WorkspaceDeleteFileInput, WorkspaceSearchInput
    from intergrax.tools.providers.workspace.service import workspace_delete_file, workspace_search, workspace_write_file

    ctx = ToolWiringContext(shadow_workspace=shadow_workspace)
    workspace_write_file(ctx, WorkspaceWriteFileInput(path="src/main.py", content="def run():\n    return 42\n"))
    search_out = workspace_search(ctx, WorkspaceSearchInput(query="return 42"))
    assert search_out.match_count == 1
    assert search_out.matches[0].path == "src/main.py"

    delete_out = workspace_delete_file(ctx, WorkspaceDeleteFileInput(path="src/main.py"))
    assert delete_out.deleted is True
    search_after = workspace_search(ctx, WorkspaceSearchInput(query="return 42"))
    assert search_after.match_count == 0


def test_workspace_not_configured() -> None:
    with pytest.raises(RuntimeError, match="shadow_workspace_not_configured"):
        workspace_write_file(ToolWiringContext(), WorkspaceWriteFileInput(path="a.txt", content="x"))


def test_workspace_via_runtime_invoker(shadow_workspace: ShadowWorkspace) -> None:
    ctx = ToolWiringContext(shadow_workspace=shadow_workspace)
    registry = ToolRegistry()
    register_workspace_tools(registry, ctx)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = build_runtime_state_for_tests(run_id="ws_run")
    request = ToolExecutionRequest(
        run_id="ws_run",
        step_id="step/1",
        tool_id="workspace.write_file",
        input=WorkspaceWriteFileInput(path="artifact.txt", content="via invoker"),
    )
    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is True
