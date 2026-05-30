# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the research_application FastAPI host."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


def build_research_mcp_server(
    *,
    nexus_loop: NexusLoop,
    route_prefix: str,
    tool_registry=None,
) -> FastMCP:
    """MCP tools for research host — includes research pipeline tool."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    kwargs: dict = {
        "name": "Intergrax Research MCP",
        "nexus_loop": nexus_loop,
        "default_capability": "research.pipeline",
        "default_tenant_id": "research",
        "default_user_id": "research-user",
    }
    if isinstance(tool_registry, ToolRegistry):
        kwargs["tool_registry"] = tool_registry
    mcp = build_nexus_mcp_server(**kwargs)
    runner = UnifiedTaskRunner(nexus_loop)

    @mcp.tool
    async def run_research_pipeline(
        message: str,
        tenant_id: str = "research",
        user_id: str = "research-user",
    ) -> dict[str, Any]:
        """Run research → summarize pipeline (same as POST .../run)."""
        run_id = new_run_id()
        task = Task(
            task_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            message=message,
            context=TaskContext(
                capability="research.pipeline",
                intent="research_summarize",
            ),
        )
        result = await runner.run_task(task)
        return {
            "task_id": result.task_id,
            "run_id": result.run_id,
            "state": result.state.value,
            "answer": result.answer,
            "agent_id": result.agent_id,
            "metadata": dict(result.metadata),
        }

    return mcp
