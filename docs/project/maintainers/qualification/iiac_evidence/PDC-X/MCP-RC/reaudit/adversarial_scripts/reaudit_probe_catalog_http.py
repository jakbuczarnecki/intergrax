"""Catalog discovery tools over real streamable HTTP in the default mode (closes the H1 script artefact of the first audit)."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["AUDIT_ROOT"])
sys.path[:0] = [str(ROOT), str(ROOT / "applications")]
from fastapi import FastAPI
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.tools.registry.runtime import ToolRegistry
from tests.unit.applications.test_mcp_rc_audit_corrections import _contract, _host, _noop
from tests.unit.applications.test_mcp_trust_a_principal_binding import _streamable_mcp_call_tool, _uvicorn_mcp_base_url


async def main() -> None:
    reg = ToolRegistry()
    reg.register(_contract(), _noop)
    mcp = build_nexus_mcp_server(name="cat", host_execution=_host(), registry=AgentRegistry(), default_capability="c", tool_registry=reg)
    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(require_api_key=True, resolved_api_key="K", api_key_principal_tenant_id="T1")
    for label, app, hdr in (("no auth configured, no key", FastAPI(), None), ("auth configured, key", api, {"X-Api-Key": "K"})):
        mcp = build_nexus_mcp_server(name="cat", host_execution=_host(), registry=AgentRegistry(), default_capability="c", tool_registry=reg)
        with _uvicorn_mcp_base_url(couple_fastapi_with_mcp(app, mcp, mount_path="/mcp")) as base:
            for tool, args in (("list_catalog_tools", {}), ("describe_catalog_tool", {"tool_id": "af.tool"})):
                r = await _streamable_mcp_call_tool(base, tool, args, headers=hdr)
                print("RESULT", label, tool, "isError=", r.isError, "|", r.content[0].text[:90].replace("\n", " "))


asyncio.run(main())
