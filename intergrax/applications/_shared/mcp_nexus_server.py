# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""FastMCP tools backed by a shared :class:`~intergrax.runtime.nexus.nexus_loop.NexusLoop`."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_catalog_tools import mount_catalog_tools_on_mcp
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.tools.registry.runtime import ToolRegistry


def build_nexus_mcp_server(
    *,
    name: str,
    nexus_loop: NexusLoop,
    default_capability: str,
    default_tenant_id: str = "mcp",
    default_user_id: str = "mcp-user",
    tool_registry: ToolRegistry | None = None,
) -> FastMCP:
    """
    Register ``list_agents`` and ``run_agent`` on a FastMCP instance.

    Tools use the same registry and :class:`~intergrax.runtime.task.unified_task_runner.UnifiedTaskRunner`
    as the Tier-3 FastAPI host.
    """
    mcp = FastMCP(name)
    runner = UnifiedTaskRunner(nexus_loop)

    @mcp.tool
    async def list_agents() -> list[dict[str, Any]]:
        """List registered agents and capabilities (same roster as the HTTP agents endpoint)."""
        agents: list[dict[str, Any]] = []
        for agent_id in nexus_loop.registry.list_agent_ids():
            contract = nexus_loop.registry.get(agent_id).get_contract()
            agents.append(
                {
                    "agent_id": contract.id,
                    "name": contract.name,
                    "capabilities": list(contract.capabilities),
                }
            )
        return agents

    @mcp.tool
    async def run_agent(
        message: str,
        capability: str = default_capability,
        tenant_id: str = default_tenant_id,
        user_id: str = default_user_id,
        session_id: str | None = None,
        intent: str | None = None,
    ) -> dict[str, Any]:
        """Run one agent task via UnifiedTaskRunner (same execution path as HTTP POST /run)."""
        run_id = new_run_id()
        context = TaskContext(capability=capability, intent=intent) if intent else TaskContext(capability=capability)
        task = Task(
            task_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            message=message,
            context=context,
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

    if tool_registry is not None:
        mount_catalog_tools_on_mcp(mcp, tool_registry)

    return mcp
