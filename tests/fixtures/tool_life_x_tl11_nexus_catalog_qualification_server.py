# © Artur Czarnecki. All rights reserved.

"""Subprocess Nexus MCP (catalog discovery) for TLX-TL11 physical qualification."""

from __future__ import annotations

import os
import socket
import sys
import time

if os.environ.get("TOOL_LIFE_X_TL11_MODE") == "unavailable":
    raise SystemExit(2)
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.exporters.mcp import to_mcp_tools
from intergrax.tools.mcp_result_bounds import bound_mcp_structured_result, emit_bounded_mcp_list
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler

_MODE = os.environ.get("TOOL_LIFE_X_TL11_MODE", "ok")
_REQUIRE_AUTH = os.environ.get("TOOL_LIFE_X_TL11_REQUIRE_AUTH", "") == "1"


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "ok"


def _catalog_handler() -> ToolHandler[_In, _Out]:
    async def _run(_input: _In, _ctx: object) -> _Out:
        raise AssertionError("catalog discovery qualification must not invoke business handlers")

    return _run


def _contract(tool_id: str) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=f"TL11 qual tool {tool_id}",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="tl11-qual",
    )


class _StubHostExecution:
    async def execute(
        self,
        task: Task,
        *,
        run_id: object = None,
        attempt_id: object = None,
        execution_id: object = None,
        resume_checkpoint: object = None,
        restore_existing_execution: bool = False,
    ) -> TaskResult:
        return TaskResult(
            task_id=task.task_id or mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="tl11-stub",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )


def _mount_qual_catalog_tools(mcp: object, registry: ToolRegistry) -> None:
    """Same contracts as production mount; fault injection via ``TOOL_LIFE_X_TL11_MODE``."""

    @mcp.tool  # type: ignore[attr-defined]
    def list_catalog_tools() -> list[dict[str, Any]]:
        if _MODE == "hang_list":
            time.sleep(10_000)
        if _MODE == "slow_list":
            time.sleep(2.5)
        if _MODE == "crash_list":
            time.sleep(0.2)
            os._exit(7)
        payload = to_mcp_tools(registry)
        return emit_bounded_mcp_list(payload)

    @mcp.tool  # type: ignore[attr-defined]
    def describe_catalog_tool(tool_id: str) -> dict[str, Any]:
        if _MODE == "hang_describe":
            time.sleep(10_000)
        if _MODE == "crash_describe":
            time.sleep(0.15)
            os._exit(8)
        if not registry.has(tool_id):
            payload = {"error": f"unknown_tool:{tool_id}"}
            bounded, truncated = bound_mcp_structured_result(payload)
            if truncated:
                return {"error": "mcp_payload_byte_limit_exceeded", "tool_id": tool_id}
            return bounded if isinstance(bounded, dict) else {"error": "mcp_payload_byte_limit_exceeded"}
        registered = registry.get(tool_id)
        items = to_mcp_tools([registered.contract])
        payload = items[0] if items else {"error": f"unknown_tool:{tool_id}"}
        bounded, truncated = bound_mcp_structured_result(payload)
        if truncated:
            return {"error": "mcp_payload_byte_limit_exceeded", "tool_id": tool_id}
        return bounded if isinstance(bounded, dict) else {"error": "mcp_payload_byte_limit_exceeded"}


def main() -> None:
    if _MODE == "unavailable":
        sys.exit(2)
    if _MODE == "never_ready":
        time.sleep(60_000)
        return

    registry = ToolRegistry()
    registry.register(_contract("tl11.qual.one"), _catalog_handler())

    mcp = build_nexus_mcp_server(
        name="tool-life-x-tl11-catalog",
        host_execution=_StubHostExecution(),
        registry=AgentRegistry(),
        default_capability="tl11.qual.cap",
        tool_registry=None,
    )
    _mount_qual_catalog_tools(mcp, registry)

    if _REQUIRE_AUTH:
        api = FastAPI()
        api.state.harness_auth = HarnessAuthState(
            require_api_key=True,
            resolved_api_key="tl11-qual-secret",
            api_key_principal_tenant_id="tl11-bound-tenant",
        )
        http_app = couple_fastapi_with_mcp(api, mcp, mount_path="/mcp")
    else:
        http_app = mcp.http_app(path="/")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    print(port, flush=True)
    uvicorn.run(http_app, host="127.0.0.1", port=port, log_level="error")


if __name__ == "__main__":
    main()
