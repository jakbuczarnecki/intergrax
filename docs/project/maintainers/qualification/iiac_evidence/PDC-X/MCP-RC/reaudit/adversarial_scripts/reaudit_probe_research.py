"""Research pipeline custom tool (double identity resolution: gate + execute_mcp_agent_task_with_resolved_identity) over real HTTP."""
from __future__ import annotations
import asyncio, os, sys
from pathlib import Path
from unittest.mock import AsyncMock
ROOT = Path(os.environ["AUDIT_ROOT"]); sys.path[:0] = [str(ROOT), str(ROOT / "applications")]
from fastapi import FastAPI
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from applications.research_application.mcp.server import build_research_mcp_server
from tests.unit.applications.test_mcp_trust_a_principal_binding import _completed, _streamable_mcp_call_tool, _uvicorn_mcp_base_url

async def main():
    port = AsyncMock(spec=HostTaskExecutionPort); port.execute = AsyncMock(return_value=_completed())
    mcp = build_research_mcp_server(host_execution=port, registry=AgentRegistry(), route_prefix="/research")
    api = FastAPI(); api.state.harness_auth = HarnessAuthState(require_api_key=True, resolved_api_key="K", api_key_principal_tenant_id="T1")
    for label, app, hdr, args in (
        ("no auth configured", FastAPI(), None, {"message": "m"}),
        ("key", api, {"X-Api-Key": "K"}, {"message": "m"}),
        ("key + forged tenant", api, {"X-Api-Key": "K"}, {"message": "m", "tenant_id": "victim"}),
    ):
        port.execute.reset_mock()
        mcp = build_research_mcp_server(host_execution=port, registry=AgentRegistry(), route_prefix="/research")
        with _uvicorn_mcp_base_url(couple_fastapi_with_mcp(app, mcp, mount_path="/mcp")) as base:
            r = await _streamable_mcp_call_tool(base, "run_research_pipeline", args, headers=hdr)
            print("RESULT", label, "isError=", r.isError, "executions=", port.execute.await_count, "|", r.content[0].text[:110].replace("\n", " "))
asyncio.run(main())
