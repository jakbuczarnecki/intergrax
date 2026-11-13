"""A3 probes: catalog discovery gating over HTTP; raw HTTP for empty bearer."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["AUDIT_ROOT"])
sys.path[:0] = [str(ROOT), str(ROOT / "applications"), str(ROOT / "agents")]
sys.argv = sys.argv[:1]
import httpx
from fastapi import FastAPI
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_application_server import compose_application_mcp_server
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.tools.registry.runtime import ToolRegistry

# reuse helpers from A1 without running its main
import importlib.util

spec = importlib.util.spec_from_file_location("a1src", ROOT / ".tmp/session/mcp-rc-audit/adv/adv_identity.py")
src = open(ROOT / ".tmp/session/mcp-rc-audit/adv/adv_identity.py", encoding="utf-8").read().split("async def main():")[0]
ns: dict = {}
exec(compile(src, "a1helpers", "exec"), ns)
Srv, host_port, call, text = ns["Srv"], ns["host_port"], ns["call"], ns["text"]
OUT = []


def rec(c, o, d=""):
    OUT.append((c, o, str(d)[:300]))
    print(f"{o:6} {c} :: {str(d)[:240]}", flush=True)


async def main():
    mcp = compose_application_mcp_server(
        display_name="p3", host_execution=host_port(), registry=AgentRegistry(), default_capability="c",
        tool_registry=ToolRegistry(),
    )
    with Srv(couple_fastapi_with_mcp(FastAPI(), mcp, mount_path="/mcp")) as s:
        u = f"http://127.0.0.1:{s.port}/mcp/"
        for tool, args in [("list_catalog_tools", {}), ("describe_catalog_tool", {"tool_id": "x"})]:
            r = await call(u, tool, args)
            rec(f"P3 default/no-auth {tool}", "PASS" if r.isError and "MCP_PRINCIPAL_REQUIRED" in text(r) else "FAIL", text(r))
    # raw bearer-empty probe against an api-key host
    mcp = compose_application_mcp_server(display_name="p3b", host_execution=host_port(), registry=AgentRegistry(), default_capability="c")
    app = couple_fastapi_with_mcp(FastAPI(), mcp, mount_path="/mcp")
    app.state.harness_auth = HarnessAuthState(require_api_key=True, resolved_api_key="K", api_key_principal_tenant_id="T1")
    with Srv(app) as s:
        u = f"http://127.0.0.1:{s.port}/mcp/"
        body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "a", "version": "1"}}}
        async with httpx.AsyncClient() as c:
            for hdr in [{}, {"Authorization": "Bearer "}, {"Authorization": "Bearer"}, {"X-Api-Key": "K"}]:
                r = await c.post(u, json=body, headers={"Accept": "application/json, text/event-stream", **hdr})
                rec(f"P3 raw initialize {hdr}", "INFO", f"status={r.status_code} body={r.text[:160]!r}")
    Path(ROOT / ".tmp/session/mcp-rc-audit/adv/adv_probe3_results.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8")


asyncio.run(main())
