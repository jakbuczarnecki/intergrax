"""Server-side cancellation of a custom tool under cap 1: CancelledError must pass sanitized_tool_errors and free the slot."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(os.environ["AUDIT_ROOT"])
sys.path[:0] = [str(ROOT), str(ROOT / "applications")]
from fastmcp.exceptions import ToolError
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.qualification.mcp_server_x.reference_app.build import build_mcp_server_x_reference_qualification_server as ref

started = asyncio.Event()


async def slow(message: str = "") -> dict[str, object]:
    started.set()
    await asyncio.sleep(30)
    return {}


async def main() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    mcp = ref(host_execution=port, registry=AgentRegistry(), custom_echo=slow,
              identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT, max_concurrent_tool_calls=1)
    tool = await mcp.get_tool("qual_echo")
    t = asyncio.create_task(tool.run({"message": "a"}))
    await asyncio.wait_for(started.wait(), 5)
    try:
        await tool.run({"message": "b"})
        print("FAIL second call admitted over cap")
    except Exception as e:
        print("PASS over-cap refused:", str(e)[:100])
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        print("PASS server-side CancelledError propagated unchanged (not converted to MCP_INTERNAL_ERROR)")
    except BaseException as e:  # noqa: BLE001
        print("FAIL cancellation mapped to", type(e).__name__, str(e)[:100])
    started.clear()
    t2 = asyncio.create_task(tool.run({"message": "c"}))
    try:
        await asyncio.wait_for(started.wait(), 3)
        print("PASS slot released after cancellation (third call admitted)")
    except asyncio.TimeoutError:
        print("FAIL slot leaked after cancellation")
    t2.cancel()


asyncio.run(main())
