import asyncio, os, sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
ROOT = Path(os.environ["AUDIT_ROOT"]); sys.path[:0] = [str(ROOT), str(ROOT / "applications")]
from fastmcp import Client
from fastmcp.exceptions import ToolError
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
async def main():
    for exc in (RuntimeError("SECRET-HOST-FAILURE-hunter2 /srv/app/db.py:42"), ValueError("tenant=victim token=abc")):
        port = AsyncMock(spec=HostTaskExecutionPort); port.execute = AsyncMock(side_effect=exc)
        mcp = build_nexus_mcp_server(name="leak", host_execution=port, registry=AgentRegistry(), default_capability="c", identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT)
        async with Client(mcp) as c:
            try: await c.call_tool("run_agent", {"message": "x"})
            except ToolError as e: print("run_agent host failure ->", str(e)[:200])
asyncio.run(main())
