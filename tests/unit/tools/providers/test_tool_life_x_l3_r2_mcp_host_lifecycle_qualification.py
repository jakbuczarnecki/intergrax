# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-R2 — canonical MCP host lifecycle + full subprocess adversarial matrix."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client

from intergrax.applications._shared.mcp_nexus_server import (
    build_nexus_mcp_server,
    execute_mcp_agent_task,
    task_from_mcp_agent_intake,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.runtime.task.task import TaskResult
from intergrax.tools.mcp_result_bounds import bound_mcp_structured_result
from tests.fixtures.tool_life_x_mcp_subprocess_qualification import (
    mcp_call_tool,
    mcp_qualification_subprocess,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT_A = "tenant_l3r2_a"
_TENANT_B = "tenant_l3r2_b"


def _completed_result(answer: str = "host-ok") -> TaskResult:
    return TaskResult(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
        answer=answer,
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


@pytest.mark.asyncio
async def test_l3r2_host_run_agent_delegates_to_host_task_execution_port() -> None:
    """Contract test — MCP ``run_agent`` maps intake and calls HostTaskExecutionPort (not ToolRuntime E2E)."""
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_result("delegated"))
    registry = AgentRegistry()
    mcp = build_nexus_mcp_server(
        name="l3r2-host-contract",
        host_execution=port,
        registry=registry,
        default_capability="demo.cap",
    )
    async with Client(mcp) as client:
        result = await client.call_tool(
            "run_agent",
            {
                "message": "hello",
                "capability": "demo.cap",
                "tenant_id": _TENANT_A,
                "user_id": "user-a",
            },
        )
    port.execute.assert_awaited_once()
    task = port.execute.await_args.args[0]
    assert task.tenant_id == _TENANT_A
    assert task.user_id == "user-a"
    assert task.message == "hello"
    assert result.data["answer"] == "delegated"


@pytest.mark.asyncio
async def test_l3r2_host_tenant_isolation_between_mcp_calls() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_result())
    registry = AgentRegistry()
    mcp = build_nexus_mcp_server(
        name="l3r2-tenant",
        host_execution=port,
        registry=registry,
        default_capability="demo.cap",
    )
    async with Client(mcp) as client:
        await client.call_tool(
            "run_agent",
            {"message": "a", "tenant_id": _TENANT_A, "user_id": "u1"},
        )
        await client.call_tool(
            "run_agent",
            {"message": "b", "tenant_id": _TENANT_B, "user_id": "u2"},
        )
    assert port.execute.await_count == 2
    first_task = port.execute.await_args_list[0].args[0]
    second_task = port.execute.await_args_list[1].args[0]
    assert first_task.tenant_id == _TENANT_A
    assert second_task.tenant_id == _TENANT_B
    assert first_task.message != second_task.message


def test_l3r2_mcp_intake_rejects_empty_tenant_id() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        task_from_mcp_agent_intake(
            message="x",
            capability="demo.cap",
            tenant_id="",
            user_id="u",
        )


@pytest.mark.asyncio
async def test_l3r2_execute_mcp_agent_task_single_host_invocation() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_result())
    await execute_mcp_agent_task(
        port,
        message="once",
        capability="demo.cap",
        tenant_id=_TENANT_A,
        user_id="u",
    )
    await execute_mcp_agent_task(
        port,
        message="once",
        capability="demo.cap",
        tenant_id=_TENANT_A,
        user_id="u",
    )
    assert port.execute.await_count == 2


def test_mcp_scenario_05_partial_response_transport_failure() -> None:
    with mcp_qualification_subprocess("crash_immediate") as (base, proc):
        assert base is not None
        with pytest.raises(Exception):
            asyncio.run(mcp_call_tool(base, "echo", {}, timeout_s=5.0))
        time.sleep(0.3)
        assert proc.poll() is not None


def test_mcp_scenario_06_mid_operation_disconnect() -> None:
    with mcp_qualification_subprocess("disconnect_mid") as (base, proc):
        assert base is not None
        with pytest.raises(Exception):
            asyncio.run(mcp_call_tool(base, "echo", {}, timeout_s=5.0))
        time.sleep(0.5)
        assert proc.poll() is not None


def test_mcp_scenario_07_client_cancellation() -> None:
    async def _cancelled_call() -> None:
        with mcp_qualification_subprocess("hang") as (base, proc):
            assert base is not None
            task = asyncio.create_task(mcp_call_tool(base, "echo", {}, timeout_s=30.0))
            await asyncio.sleep(0.25)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            proc.terminate()

    asyncio.run(_cancelled_call())


def test_mcp_scenario_09_late_result_after_client_timeout() -> None:
    with mcp_qualification_subprocess("late") as (base, proc):
        assert base is not None
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(mcp_call_tool(base, "echo", {}, timeout_s=0.8))
        assert proc.poll() is None
        proc.terminate()
        proc.wait(timeout=5.0)


def test_mcp_scenario_10_reconnect_fresh_session_not_late_replay() -> None:
    with mcp_qualification_subprocess("late") as (base, _late_proc):
        assert base is not None
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(mcp_call_tool(base, "echo", {}, timeout_s=0.5))
    with mcp_qualification_subprocess("ok") as (base2, _ok_proc):
        assert base2 is not None
        result = asyncio.run(mcp_call_tool(base2, "echo", {"message": "fresh"}, timeout_s=5.0))
        assert "fresh" in str(result)


def test_mcp_scenario_13_subprocess_cleanup_after_hang() -> None:
    with mcp_qualification_subprocess("hang") as (base, proc):
        assert base is not None
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(mcp_call_tool(base, "echo", {}, timeout_s=0.5))
    time.sleep(0.2)
    assert proc.poll() is not None


def test_mcp_scenario_14_no_indefinite_wait_on_failed_startup() -> None:
    with pytest.raises(TimeoutError, match="startup_timeout"):
        with mcp_qualification_subprocess("never_ready"):
            pass


def test_mcp_scenario_15_concurrent_invocations_isolated() -> None:
    async def _parallel() -> tuple[object, object]:
        with mcp_qualification_subprocess("ok") as (base, proc):
            assert base is not None
            a, b = await asyncio.gather(
                mcp_call_tool(base, "echo", {"message": "one"}, timeout_s=5.0),
                mcp_call_tool(base, "echo", {"message": "two"}, timeout_s=5.0),
            )
            assert proc.poll() is None
            return a, b

    left, right = asyncio.run(_parallel())
    assert "one" in str(left)
    assert "two" in str(right)


def test_mcp_oversized_delivered_before_host_bounding() -> None:
    """Transport may return large payload; logical bounding is host-layer (execute_mcp_agent_task)."""
    huge = {"answer": "x" * 100_000}
    bounded, truncated = bound_mcp_structured_result(huge, max_bytes=256)
    assert truncated is True
    assert bounded.get("error") == "mcp_payload_byte_limit_exceeded"
