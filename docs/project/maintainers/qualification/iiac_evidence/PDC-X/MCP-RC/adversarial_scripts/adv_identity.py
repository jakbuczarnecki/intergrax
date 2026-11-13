"""Auditor throw-away adversarial script A1: identity over real streamable HTTP (MCP-RC audit)."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(os.environ["AUDIT_ROOT"])
sys.path[:0] = [str(ROOT), str(ROOT / "applications"), str(ROOT / "agents")]

from unittest.mock import AsyncMock

import uvicorn
from fastapi import FastAPI
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_application_server import compose_application_mcp_server
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.contracts.mcp_server_definition import McpHostCustomBinding
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.unit.applications.test_mcp_trust_a_principal_binding import _completed, _streamable_mcp_call_tool

RESULTS: list[dict] = []


def rec(case, outcome, detail=""):
    RESULTS.append({"case": case, "outcome": outcome, "detail": str(detail)[:300]})
    print(f"{outcome:8} {case} :: {str(detail)[:200]}", flush=True)


class Srv:
    def __init__(self, app, host="127.0.0.1"):
        s = socket.socket()
        s.bind((host, 0))
        self.port = s.getsockname()[1]
        s.close()
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=self.port, log_level="error"))
        self.t = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.t.start()
        d = time.monotonic() + 10
        while not self.server.started and time.monotonic() < d:
            time.sleep(0.02)
        return self

    def __exit__(self, *a):
        self.server.should_exit = True
        self.t.join(5)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    finally:
        s.close()


def host_port():
    p = AsyncMock(spec=HostTaskExecutionPort)
    p.execute = AsyncMock(return_value=_completed())
    return p


def text(r):
    return r.content[0].text if r.content else ""


class _Err:
    isError = True

    def __init__(self, msg):
        self.content = [type("C", (), {"text": msg})()]


async def call(url, tool, args, headers=None):
    try:
        return await _streamable_mcp_call_tool(url, tool, args, headers=headers, timeout_s=20)
    except BaseException as e:  # transport level refusal
        return _Err(f"TRANSPORT:{type(e).__name__}:{e}")


async def main():
    # H1 default mode, no auth: every tool refuses
    port = host_port()
    mcp = build_nexus_mcp_server(name="a1", host_execution=port, registry=AgentRegistry(), default_capability="c")
    with Srv(couple_fastapi_with_mcp(FastAPI(), mcp, mount_path="/mcp")) as s:
        u = f"http://127.0.0.1:{s.port}/mcp/"
        for tool, args in [
            ("run_agent", {"message": "x", "tenant_id": "victim", "user_id": "victim-u"}),
            ("list_agents", {}),
            ("list_catalog_tools", {}),
            ("describe_catalog_tool", {"tool_id": "a"}),
        ]:
            r = await call(u, tool, args)
            ok = r.isError and "MCP_PRINCIPAL_REQUIRED" in text(r)
            rec(f"H1 default/no-auth {tool}", "PASS" if ok else "FAIL", text(r))
        rec("H1 no execute", "PASS" if port.execute.await_count == 0 else "FAIL", port.execute.await_count)
        r = await call(
            u,
            "run_agent",
            {"message": "x", "tenant_id": "victim"},
            headers={"X-Forwarded-For": "127.0.0.1", "X-Tenant-Id": "victim", "X-Api-Key": ""},
        )
        rec("H1b spoof headers + empty key", "PASS" if r.isError else "FAIL", text(r))

    # H2 auth wired AFTER coupling
    port = host_port()
    app = couple_fastapi_with_mcp(
        FastAPI(),
        build_nexus_mcp_server(name="a2", host_execution=port, registry=AgentRegistry(), default_capability="c"),
        mount_path="/mcp",
    )
    app.state.harness_auth = HarnessAuthState(
        require_api_key=True, resolved_api_key="K", api_key_principal_tenant_id="T1"
    )
    with Srv(app) as s:
        u = f"http://127.0.0.1:{s.port}/mcp/"
        cases = [
            ("no key", {}, {}, False),
            ("wrong key", {"X-Api-Key": "bad"}, {}, False),
            ("empty key", {"X-Api-Key": ""}, {}, False),
            ("bearer empty", {"Authorization": "Bearer "}, {}, False),
            ("bearer wrong", {"Authorization": "Bearer K2"}, {}, False),
            ("key + forged tenant", {"X-Api-Key": "K"}, {"tenant_id": "victim"}, False),
            ("key + forged user (tenant matches)", {"X-Api-Key": "K"}, {"tenant_id": "T1", "user_id": "victim-u"}, None),
            ("key + matching tenant", {"X-Api-Key": "K"}, {"tenant_id": "T1"}, True),
            ("key + tenant padded variant", {"X-Api-Key": "K"}, {"tenant_id": " T1 "}, None),
            ("key + tenant case variant", {"X-Api-Key": "K"}, {"tenant_id": "t1"}, None),
            ("key only", {"X-Api-Key": "K"}, {}, True),
        ]
        for name, hdr, extra, expect_ok in cases:
            n0 = port.execute.await_count
            r = await call(u, "run_agent", {"message": "x", **extra}, headers=hdr)
            ran = port.execute.await_count > n0
            tenant = port.execute.await_args.args[0].tenant_id if ran else None
            user = port.execute.await_args.args[0].user_id if ran else None
            if expect_ok is True:
                ok = (not r.isError) and tenant == "T1"
            elif expect_ok is False:
                ok = r.isError and not ran
            else:
                ok = tenant in (None, "T1")
            rec(
                f"H2 after-coupling {name}",
                "PASS" if ok else "FAIL",
                f"isError={r.isError} ran={ran} tenant={tenant} user={user} :: {text(r)[:120]}",
            )
        r = await call(u, "list_agents", {}, headers={"X-Api-Key": "K"})
        rec("H2b list_agents with key", "PASS" if not r.isError else "FAIL", text(r))
        r = await call(u, "list_agents", {})
        rec("H2b list_agents no key", "PASS" if r.isError else "FAIL", text(r))

    # H3 LOCAL_DEV_EXPLICIT: loopback ok, LAN IP refused, configured auth never downgraded
    ip = lan_ip()
    rec("H3 lan ip", "INFO", ip)
    port = host_port()
    mcp = build_nexus_mcp_server(
        name="a3",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="c",
        identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT,
    )
    app = couple_fastapi_with_mcp(FastAPI(), mcp, mount_path="/mcp")
    with Srv(app, host="0.0.0.0") as s:
        r = await call(f"http://127.0.0.1:{s.port}/mcp/", "run_agent", {"message": "x", "tenant_id": "dev-t"})
        ok = (not r.isError) and port.execute.await_args.args[0].tenant_id == "dev-t"
        rec("H3 local_dev loopback admitted", "PASS" if ok else "FAIL", text(r))
        n0 = port.execute.await_count
        r = await call(f"http://{ip}:{s.port}/mcp/", "run_agent", {"message": "x", "tenant_id": "victim"})
        rec("H3 local_dev via LAN ip refused", "PASS" if r.isError and port.execute.await_count == n0 else "FAIL", text(r))
        r = await call(f"http://{ip}:{s.port}/mcp/", "list_agents", {})
        rec("H3 local_dev list_agents via LAN refused", "PASS" if r.isError else "FAIL", text(r))
        r = await call(
            f"http://{ip}:{s.port}/mcp/",
            "run_agent",
            {"message": "x", "tenant_id": "victim"},
            headers={"X-Forwarded-For": "127.0.0.1", "Forwarded": "for=127.0.0.1"},
        )
        rec("H3 local_dev via LAN + XFF spoof refused", "PASS" if r.isError and port.execute.await_count == n0 else "FAIL", text(r))
    port = host_port()
    mcp = build_nexus_mcp_server(
        name="a3b",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="c",
        identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT,
    )
    app = couple_fastapi_with_mcp(FastAPI(), mcp, mount_path="/mcp")
    app.state.harness_auth = HarnessAuthState(
        require_api_key=True, resolved_api_key="K", api_key_principal_tenant_id="T1"
    )
    with Srv(app) as s:
        u = f"http://127.0.0.1:{s.port}/mcp/"
        r = await call(u, "run_agent", {"message": "x", "tenant_id": "victim"})
        rec("H3b local_dev + configured auth, no key (no downgrade)", "PASS" if r.isError and port.execute.await_count == 0 else "FAIL", text(r))
        r = await call(u, "run_agent", {"message": "x", "tenant_id": "victim"}, headers={"X-Api-Key": "K"})
        rec("H3b local_dev + auth + key + forged tenant -> conflict", "PASS" if r.isError and port.execute.await_count == 0 else "FAIL", text(r))

    # H4 custom host tool gating in default mode (no auth)
    async def secret_tool(q: str = "x") -> dict[str, object]:
        return {"tenant_data": "SECRET-" + q}

    mcp = compose_application_mcp_server(
        display_name="a4",
        host_execution=host_port(),
        registry=AgentRegistry(),
        default_capability="c",
        extra_bindings=(McpHostCustomBinding(binding_id="sec", tool_name="secret_tool"),),
        custom_host_tools={"secret_tool": secret_tool},
    )
    with Srv(couple_fastapi_with_mcp(FastAPI(), mcp, mount_path="/mcp")) as s:
        r = await call(f"http://127.0.0.1:{s.port}/mcp/", "secret_tool", {"q": "1"})
        rec("H4 custom host tool reachable w/o principal in HARNESS_BOUND (GAP if reachable)", "GAP" if not r.isError else "PASS", text(r))
    (ROOT / ".tmp/session/mcp-rc-audit/adv/adv_identity_results.json").write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")


asyncio.run(main())
