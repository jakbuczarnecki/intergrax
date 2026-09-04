# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical Tier-3 host templates for scaffold generators (NPSC-2)."""

from __future__ import annotations

from textwrap import dedent

from intergrax.scaffold.agent_catalog import ScaffoldAgentSpec
from intergrax.scaffold.application_names import ScaffoldApplicationNames


def render_canonical_lab_serving_router_py(names: ScaffoldApplicationNames) -> str:
    short = names.short
    pascal = names.pascal
    route_prefix = names.route_prefix
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Optional

        from fastapi import APIRouter, FastAPI, HTTPException, status
        from pydantic import BaseModel, Field

        from intergrax.runtime.execution.host_task import HostTaskExecutionPort
        from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
        from intergrax.runtime.task.task import Task, TaskContext
        from intergrax.runtime.task.task_run_bridge import new_run_id


        class {pascal}RunRequestV1(BaseModel):
            tenant_id: str = "lab"
            user_id: str = "lab-user"
            session_id: Optional[str] = None
            message: str = Field(min_length=1)
            capability: str = Field(min_length=1)
            metadata: dict[str, object] = Field(default_factory=dict)


        class {pascal}RunResponseV1(BaseModel):
            task_id: str
            run_id: Optional[str] = None
            state: str
            answer: str = ""
            agent_id: Optional[str] = None
            metadata: dict[str, object] = Field(default_factory=dict)


        @dataclass
        class {pascal}RunService:
            host_execution: HostTaskExecutionPort

            @classmethod
            def from_host_execution(
                cls,
                host_execution: HostTaskExecutionPort,
            ) -> {pascal}RunService:
                return cls(host_execution=host_execution)

            async def run_task(self, body: {pascal}RunRequestV1) -> {pascal}RunResponseV1:
                run_id = new_run_id()
                task = Task(
                    task_id=run_id,
                    tenant_id=body.tenant_id,
                    user_id=body.user_id,
                    session_id=body.session_id,
                    message=body.message,
                    context=TaskContext(capability=body.capability),
                    metadata=dict(body.metadata),
                )
                result = await self.host_execution.execute(task)
                return {pascal}RunResponseV1(
                    task_id=result.task_id,
                    run_id=result.run_id,
                    state=result.state.value,
                    answer=result.answer,
                    agent_id=result.agent_id,
                    metadata=dict(result.metadata),
                )


        def mount_{short}_routes(
            app: FastAPI,
            *,
            host_execution: HostTaskExecutionPort,
            registry: AgentRegistryRead,
            prefix: str = "{route_prefix}",
        ) -> {pascal}RunService:
            service = {pascal}RunService.from_host_execution(host_execution)
            router = APIRouter(prefix=prefix, tags=["{short}"])

            @router.post("/run", response_model={pascal}RunResponseV1)
            async def run_agent(body: {pascal}RunRequestV1) -> {pascal}RunResponseV1:
                try:
                    return await service.run_task(body)
                except Exception as exc:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"run_error: {{exc.__class__.__name__}}",
                    ) from exc

            @router.get("/agents")
            async def list_agents() -> dict[str, list[dict[str, object]]]:
                agents: list[dict[str, object]] = []
                for agent_id in registry.list_agent_ids():
                    contract = registry.get(agent_id).get_contract()
                    agents.append(
                        {{
                            "agent_id": contract.id,
                            "name": contract.name,
                            "capabilities": list(contract.capabilities),
                        }}
                    )
                return {{"agents": agents}}

            app.include_router(router)
            return service
        '''
    )


def render_canonical_product_serving_router_py(
    names: ScaffoldApplicationNames,
    specs: list[ScaffoldAgentSpec],
) -> str:
    short = names.short
    pascal = names.pascal
    pkg = names.pkg
    route_prefix = names.route_prefix
    default_cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from dataclasses import dataclass

        from fastapi import APIRouter, FastAPI, HTTPException, status

        from intergrax.runtime.execution.host_task import HostTaskExecutionPort
        from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
        from intergrax.runtime.task.task import Task, TaskContext
        from intergrax.runtime.task.task_run_bridge import new_run_id
        from {pkg}.serving.schemas import {pascal}RunRequestV1, {pascal}RunResponseV1


        @dataclass
        class {pascal}RunService:
            host_execution: HostTaskExecutionPort
            default_agent_id: str

            @classmethod
            def from_host_execution(
                cls,
                host_execution: HostTaskExecutionPort,
                *,
                default_agent_id: str,
            ) -> {pascal}RunService:
                return cls(
                    host_execution=host_execution,
                    default_agent_id=default_agent_id,
                )

            async def run_task(self, body: {pascal}RunRequestV1) -> {pascal}RunResponseV1:
                run_id = new_run_id()
                task = Task(
                    task_id=run_id,
                    tenant_id=body.tenant_id,
                    user_id=body.user_id,
                    session_id=body.session_id,
                    agent_id=self.default_agent_id,
                    message=body.message,
                    context=TaskContext(capability=body.capability or "{default_cap}"),
                )
                result = await self.host_execution.execute(task)
                return {pascal}RunResponseV1(
                    task_id=result.task_id,
                    run_id=result.run_id,
                    state=result.state.value,
                    answer=result.answer,
                    agent_id=result.agent_id,
                    metadata=dict(result.metadata),
                )


        def mount_{short}_routes(
            app: FastAPI,
            *,
            host_execution: HostTaskExecutionPort,
            registry: AgentRegistryRead,
            prefix: str = "{route_prefix}",
            default_agent_id: str = "echo",
        ) -> {pascal}RunService:
            service = {pascal}RunService.from_host_execution(
                host_execution,
                default_agent_id=default_agent_id,
            )
            router = APIRouter(prefix=prefix, tags=["{short}"])

            @router.post("/run", response_model={pascal}RunResponseV1)
            async def run_agent(body: {pascal}RunRequestV1) -> {pascal}RunResponseV1:
                try:
                    return await service.run_task(body)
                except Exception as exc:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"run_error: {{exc.__class__.__name__}}",
                    ) from exc

            @router.get("/agents")
            async def list_agents() -> dict[str, list[dict[str, object]]]:
                agents: list[dict[str, object]] = []
                for agent_id in registry.list_agent_ids():
                    contract = registry.get(agent_id).get_contract()
                    agents.append(
                        {{
                            "agent_id": contract.id,
                            "name": contract.name,
                            "capabilities": list(contract.capabilities),
                        }}
                    )
                return {{"agents": agents}}

            app.include_router(router)
            return service
        '''
    )


def render_canonical_mcp_server_py(
    names: ScaffoldApplicationNames,
    specs: list[ScaffoldAgentSpec],
) -> str:
    pkg = names.pkg
    short = names.short
    display = names.display
    cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """FastMCP server coupled to the {pkg} FastAPI host."""

        from __future__ import annotations

        from fastmcp import FastMCP

        from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
        from intergrax.runtime.execution.host_task import HostTaskExecutionPort
        from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead


        def build_{short}_mcp_server(
            *,
            host_execution: HostTaskExecutionPort,
            registry: AgentRegistryRead,
            route_prefix: str,
            tool_registry: object | None = None,
        ) -> FastMCP:
            """MCP tools mirror the HTTP API (canonical host task execution)."""
            _ = route_prefix
            from intergrax.tools.registry.runtime import ToolRegistry

            kwargs: dict[str, object] = {{
                "name": "{display} MCP",
                "host_execution": host_execution,
                "registry": registry,
                "default_capability": "{cap}",
            }}
            if isinstance(tool_registry, ToolRegistry):
                kwargs["tool_registry"] = tool_registry
            return build_nexus_mcp_server(**kwargs)
        '''
    )
