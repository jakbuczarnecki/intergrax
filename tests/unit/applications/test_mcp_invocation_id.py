# © Artur Czarnecki. All rights reserved.

"""MCP-RC MRC-4 — dedicated MCP invocation correlation id (R-P7-01 / REQ-MCP-OBS-01, OD-MCP-RC-01 D-5)."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ID = re.compile(r"^mcpi-[0-9a-f]{32}$")


def _server(*, tenant_marker: str = "secret-tenant-marker"):
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(
        side_effect=lambda task, **_: TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="ok",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )
    )
    mcp = build_nexus_mcp_server(
        name="rc-trace",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="cap",
        default_tenant_id=tenant_marker,
        identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT,
    )
    return mcp, port


@pytest.mark.asyncio
async def test_run_agent_payload_and_task_carry_the_same_well_formed_invocation_id() -> None:
    mcp, port = _server()
    async with Client(mcp) as client:
        result = await client.call_tool("run_agent", {"message": "x"})
    invocation_id = result.data["mcp_invocation_id"]
    assert _ID.match(invocation_id)
    task = port.execute.await_args.args[0]
    assert task.metadata["mcp_invocation_id"] == invocation_id


@pytest.mark.asyncio
async def test_invocation_id_is_unique_per_call_including_concurrent_calls() -> None:
    mcp, _port = _server()
    async with Client(mcp) as client:
        results = await asyncio.gather(*(client.call_tool("run_agent", {"message": str(i)}) for i in range(40)))
    ids = [r.data["mcp_invocation_id"] for r in results]
    assert len(set(ids)) == 40


@pytest.mark.asyncio
async def test_invocation_id_is_a_correlation_key_not_an_execution_authority_nor_tenant_derived() -> None:
    mcp, port = _server(tenant_marker="secret-tenant-marker")
    async with Client(mcp) as client:
        result = await client.call_tool("run_agent", {"message": "x"})
    invocation_id = result.data["mcp_invocation_id"]
    assert "secret-tenant-marker" not in invocation_id
    assert "mcp-user" not in invocation_id
    assert invocation_id != result.data["task_id"]
    assert invocation_id != result.data["run_id"]
    task = port.execute.await_args.args[0]
    assert task.tenant_id == "secret-tenant-marker"  # authority still comes from identity, not from the id
    assert str(task.task_id) != invocation_id
