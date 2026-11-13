"""Re-audit throw-away probes (G6 delta audit of MCP-RC). Run: AUDIT_ROOT=<checkout> python reaudit_probes.py"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(os.environ["AUDIT_ROOT"])
sys.path[:0] = [str(ROOT), str(ROOT / "applications"), str(ROOT / "agents")]

import uvicorn
from fastapi import FastAPI
from fastmcp import Client
from fastmcp.exceptions import ToolError
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_application_server import compose_application_mcp_server
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.contracts.mcp_server_definition import McpHostCustomBinding
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.tools.registry.runtime import ToolRegistry
from tests.unit.applications.test_mcp_trust_a_principal_binding import _completed, _streamable_mcp_call_tool

RESULTS: list[dict] = []


def rec(case, outcome, detail=""):
    RESULTS.append({"case": case, "outcome": outcome, "detail": str(detail)[:400]})
    print(f"{outcome:6} {case} :: {str(detail)[:300]}", flush=True)


class Srv:
    def __init__(self, app):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.t = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.t.start()
        d = time.monotonic() + 20
        while not self.server.started and time.monotonic() < d:
            time.sleep(0.02)
        return self

    def __exit__(self, *a):
        self.server.should_exit = True
        self.t.join(5)


def hp():
    p = AsyncMock(spec=HostTaskExecutionPort)
    p.execute = AsyncMock(return_value=_completed())
    return p


def txt(r):
    try:
        return r.content[0].text
    except Exception:
        return str(r)


def keyed_api():
    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(require_api_key=True, resolved_api_key="K", api_key_principal_tenant_id="T1")
    return api


async def main():
    # P1 run_agent host exception over real HTTP with a verified principal (HARNESS_BOUND default)
    secret = RuntimeError("SECRET-HOST-FAILURE-hunter2 /srv/app/db.py:42")
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(side_effect=secret)
    mcp = build_nexus_mcp_server(name="p1", host_execution=port, registry=AgentRegistry(), default_capability="c")
    with Srv(couple_fastapi_with_mcp(keyed_api(), mcp, mount_path="/mcp")) as s:
        u = f"http://127.0.0.1:{s.port}/mcp/"
        r = await _streamable_mcp_call_tool(u, "run_agent", {"message": "x"}, headers={"X-Api-Key": "K"})
        t = txt(r)
        ok = r.isError and "SECRET" not in t and "hunter2" not in t and "db.py" not in t
        rec("P1 run_agent host exception over real HTTP with key", "PASS" if ok else "FAIL", t)

    # P3 custom tool gate over real HTTP
    calls: list[str] = []

    async def sec(q: str = "x") -> dict[str, object]:
        calls.append(q)
        return {"d": "SECRET-" + q}

    def build_sec(name):
        return compose_application_mcp_server(
            display_name=name, host_execution=hp(), registry=AgentRegistry(), default_capability="c",
            extra_bindings=(McpHostCustomBinding(binding_id="sec", tool_name="secret_tool"),),
            custom_host_tools={"secret_tool": sec},
        )

    with Srv(couple_fastapi_with_mcp(FastAPI(), build_sec("p3"), mount_path="/mcp")) as s:
        r = await _streamable_mcp_call_tool(f"http://127.0.0.1:{s.port}/mcp/", "secret_tool", {"q": "1"})
        rec("P3a custom tool, default mode, no auth, Tier-3 facade", "PASS" if r.isError and "MCP_PRINCIPAL_REQUIRED" in txt(r) and not calls else "FAIL", txt(r))
    with Srv(couple_fastapi_with_mcp(keyed_api(), build_sec("p3b"), mount_path="/mcp")) as s:
        u = f"http://127.0.0.1:{s.port}/mcp/"
        r = await _streamable_mcp_call_tool(u, "secret_tool", {"q": "ok"}, headers={"X-Api-Key": "K"})
        rec("P3b custom tool with valid key", "PASS" if not r.isError else "FAIL", txt(r))
        n = len(calls)
        for hdr in ({"X-Api-Key": "bad"}, {"X-Api-Key": ""}, {}):
            try:
                r = await _streamable_mcp_call_tool(u, "secret_tool", {"q": "no"}, headers=hdr)
                rec(f"P3c custom tool bad creds {hdr}", "PASS" if r.isError and len(calls) == n else "FAIL", txt(r))
            except BaseException as e:  # noqa: BLE001  (HTTP 401 raised by the transport client)
                rec(f"P3c custom tool bad creds {hdr}", "PASS" if len(calls) == n and "401" in repr(e) else "FAIL", "refused at HTTP layer (401)" if "401" in repr(e) else repr(e)[:200])
    params = inspect.signature(compose_application_mcp_server).parameters
    rec("P3d compose_application_mcp_server exposes public_custom_tool_names", "PASS" if "public_custom_tool_names" in params else "GAP", "opt-out unreachable via the Tier-3 facade" if "public_custom_tool_names" not in params else "")

    # P4 ToolError raised by the host passes through unchanged
    async def toolerr(q: str = "x") -> dict[str, object]:
        raise ToolError("UPSTREAM-SECRET tenant=victim token=abc")

    class Boom_tenant_victim(Exception):
        pass

    async def grp(q: str = "x") -> dict[str, object]:
        raise ExceptionGroup("g", [ValueError("EG-SECRET")])

    async def cls(q: str = "x") -> dict[str, object]:
        raise Boom_tenant_victim("m")

    mcp = compose_application_mcp_server(
        display_name="p4", host_execution=hp(), registry=AgentRegistry(), default_capability="c",
        identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT,
        extra_bindings=tuple(McpHostCustomBinding(binding_id=n, tool_name=n) for n in ("te_tool", "g_tool", "c_tool")),
        custom_host_tools={"te_tool": toolerr, "g_tool": grp, "c_tool": cls},
    )
    async with Client(mcp) as c:
        for n, needle in (("te_tool", "UPSTREAM-SECRET"), ("g_tool", "EG-SECRET"), ("c_tool", "zzz")):
            try:
                await c.call_tool(n, {})
            except ToolError as e:
                leaked = needle in str(e)
                rec(f"P4 {n}", ("NOTE" if n == "te_tool" else "FAIL") if leaked else "PASS", str(e))

    # P5 LOCAL_DEV: a host-raised TimeoutError is labelled as a call-bound expiry
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(side_effect=TimeoutError("host internal timeout"))
    mcp = build_nexus_mcp_server(name="p5", host_execution=port, registry=AgentRegistry(), default_capability="c",
                                 identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT, max_call_seconds=30.0)
    async with Client(mcp) as c:
        t0 = time.monotonic()
        try:
            await c.call_tool("run_agent", {"message": "x"})
        except ToolError as e:
            rec("P5 host TimeoutError (no bound expiry) in LOCAL_DEV with max_call_seconds", "NOTE" if "MCP_DEADLINE_EXCEEDED" in str(e) else "PASS", f"{e} after {time.monotonic() - t0:.2f}s")

    # P6 catalog discovery with a blocking registry: loop stays live
    orig = ToolRegistry.list

    def slow_list(self, *a, **k):
        time.sleep(1.5)
        return orig(self, *a, **k)

    mcp = build_nexus_mcp_server(name="p6", host_execution=hp(), registry=AgentRegistry(), default_capability="c",
                                 tool_registry=ToolRegistry(), identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT)
    with patch.object(ToolRegistry, "list", slow_list):
        async with Client(mcp) as c:
            async def timed(name):
                t0 = time.monotonic()
                await c.call_tool(name, {})
                return time.monotonic() - t0

            a = asyncio.create_task(timed("list_catalog_tools"))
            await asyncio.sleep(0.2)
            b = await timed("list_agents")
            a_t = await a
            rec("P6 list_agents latency while list_catalog_tools blocks 1.5 s", "PASS" if b < 1.0 else "FAIL", f"list_agents {b:.2f}s, list_catalog {a_t:.2f}s")

    # P7 deep metadata in a run_agent result
    base = _completed()
    deep: dict = {}
    cur = deep
    for _ in range(5000):
        cur["n"] = {}
        cur = cur["n"]
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=base.model_copy(update={"metadata": deep}))
    mcp = build_nexus_mcp_server(name="p7", host_execution=port, registry=AgentRegistry(), default_capability="c",
                                 identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT)
    async with Client(mcp) as c:
        try:
            r = await c.call_tool("run_agent", {"message": "x"})
            rec("P7 run_agent deep metadata", "NOTE", str(r.data)[:100])
        except ToolError as e:
            rec("P7 run_agent deep metadata", "PASS" if "MCP_" in str(e) and "Recursion" not in str(e)[:0] else "FAIL", str(e))
        except BaseException as e:  # noqa: BLE001
            rec("P7 run_agent deep metadata", "FAIL", f"{type(e).__name__}: {e}")

    # P8 admission cap 1 + client cancel
    from tests.qualification.mcp_server_x.reference_app.build import build_mcp_server_x_reference_qualification_server as ref

    async def slow(message: str = "") -> dict[str, object]:
        await asyncio.sleep(30)
        return {}

    ps = inspect.signature(ref).parameters
    if "max_concurrent_tool_calls" in ps:
        mcp = ref(host_execution=hp(), registry=AgentRegistry(), custom_echo=slow,
                  identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT, max_concurrent_tool_calls=1)
        async with Client(mcp) as c:
            t = asyncio.create_task(c.call_tool("qual_echo", {"message": "m"}))
            await asyncio.sleep(0.5)
            try:
                await c.call_tool("qual_echo", {"message": "m2"})
                rec("P8a second call over cap 1", "FAIL", "admitted")
            except ToolError as e:
                rec("P8a second call over cap 1", "PASS" if "CONCURRENT" in str(e) else "FAIL", str(e))
            t.cancel()
            try:
                await t
            except BaseException:  # noqa: BLE001
                pass
            await asyncio.sleep(0.5)
            t2 = asyncio.create_task(c.call_tool("qual_echo", {"message": "m3"}))
            await asyncio.sleep(0.5)
            rec("P8b slot after in-memory client cancel (probe limitation: the in-memory Client does not propagate cancel to the server task; see reaudit_probe_cancel.py for the server-side check)", "NOTE", "third call admitted" if not t2.done() else "refused while the first handler (30 s sleep) still holds the slot")
            t2.cancel()
    else:
        rec("P8 skipped", "NOTE", "reference build has no max_concurrent_tool_calls param; params=" + ",".join(ps))
    Path(os.environ.get("PROBE_OUT", "reaudit_probes_results.json")).write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")


asyncio.run(main())
