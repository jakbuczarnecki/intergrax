# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""FastMCP tools backed by canonical :class:`~intergrax.runtime.execution.host_task.HostTaskExecution`."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_catalog_tools import mount_catalog_tools_on_mcp
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import mint_intake_execution_identity
from intergrax.tools.registry.runtime import ToolRegistry


async def execute_mcp_agent_task(
    host_execution: HostTaskExecutionPort,
    *,
    message: str,
    capability: str,
    tenant_id: str,
    user_id: str,
    session_id: str | None = None,
    intent: str | None = None,
) -> dict[str, object]:
    """Run one MCP agent task through canonical host task execution."""
    task_id, _run_id = mint_intake_execution_identity()
    context = TaskContext(capability=capability, intent=intent) if intent else TaskContext(capability=capability)
    task = Task(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        message=message,
        context=context,
    )
    result = await host_execution.execute(task)
    return {
        "task_id": result.task_id,
        "run_id": result.run_id,
        "state": result.state.value,
        "answer": result.answer,
        "agent_id": result.agent_id,
        "metadata": dict(result.metadata),
    }


def build_nexus_mcp_server(
    *,
    name: str,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    default_capability: str,
    default_tenant_id: str = "mcp",
    default_user_id: str = "mcp-user",
    tool_registry: ToolRegistry | None = None,
) -> FastMCP:
    """
    Register ``list_agents`` and ``run_agent`` on a FastMCP instance.

    Tools use the same registry and canonical host task execution as the Tier-3 FastAPI host.
    """
    mcp = FastMCP(name)

    @mcp.tool
    async def list_agents() -> list[dict[str, Any]]:
        """List registered agents and capabilities (same roster as the HTTP agents endpoint)."""
        agents: list[dict[str, Any]] = []
        for agent_id in registry.list_agent_ids():
            contract = registry.get(agent_id).get_contract()
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
    ) -> dict[str, object]:
        """Run one agent task via canonical host task execution (same path as HTTP POST /run)."""
        return await execute_mcp_agent_task(
            host_execution,
            message=message,
            capability=capability,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            intent=intent,
        )

    if tool_registry is not None:
        mount_catalog_tools_on_mcp(mcp, tool_registry)

    return mcp
