import asyncio, os, sys
from pathlib import Path
ROOT = Path(os.environ["AUDIT_ROOT"]); sys.path[:0] = [str(ROOT), str(ROOT / "applications")]
src = open(ROOT / ".tmp/session/mcp-rc-audit/adv/adv_identity.py", encoding="utf-8").read().split("async def main():")[0]
ns = {}; exec(compile(src, "h", "exec"), ns)
from fastapi import FastAPI
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_application_server import compose_application_mcp_server
from intergrax.runtime.registry.agent_registry import AgentRegistry
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
async def main():
    mcp = compose_application_mcp_server(display_name="p4", host_execution=ns["host_port"](), registry=AgentRegistry(), default_capability="c")
    app = couple_fastapi_with_mcp(FastAPI(), mcp, mount_path="/mcp")
    app.state.harness_auth = HarnessAuthState(require_api_key=True, resolved_api_key="K", api_key_principal_tenant_id="T1")
    with ns["Srv"](app) as s:
        async with streamablehttp_client(f"http://127.0.0.1:{s.port}/mcp/") as (r, w, _):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                t = await sess.list_tools()
                print("UNAUTH tools/list ->", [x.name for x in t.tools])
asyncio.run(main())
