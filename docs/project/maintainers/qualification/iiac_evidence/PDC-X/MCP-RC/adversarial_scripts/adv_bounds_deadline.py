"""Auditor throw-away adversarial script A2: result bounds, error text, concurrency, deadline wall-time, cancel (MCP-RC audit)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(os.environ["AUDIT_ROOT"])
sys.path[:0] = [str(ROOT), str(ROOT / "applications"), str(ROOT / "agents")]

from fastmcp import Client
from fastmcp.exceptions import ToolError
from intergrax.applications._shared import mcp_host_execution_transport_cancel as tc
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
    build_host_root_execution_transport_detachment_supervisor,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from tests.qualification.mcp_server_x.test_mcp_server_x_p7_failure_injection_l2 import (
    _boundary,
    _HangingHost,
    _identity_resolution,
    _reference_server,
)
from tests.unit.applications.test_mcp_deadline_and_resilience import _StubbornHost

RESULTS: list[dict] = []
NEXUS = "intergrax.applications._shared.mcp_nexus_server"


def rec(case, outcome, detail=""):
    RESULTS.append({"case": case, "outcome": outcome, "detail": str(detail)[:400]})
    print(f"{outcome:8} {case} :: {str(detail)[:260]}", flush=True)


async def err_of(client, tool, args):
    try:
        r = await client.call_tool(tool, args)
        return False, str(r.data)[:200]
    except ToolError as e:
        return True, str(e)[:300]
    except BaseException as e:
        return True, f"{type(e).__name__}: {e}"[:300]


async def bounds():
    nested = []
    cur = nested
    for _ in range(5000):
        nxt = []
        cur.append(nxt)
        cur = nxt
    circ: list = []
    circ.append(circ)

    async def deep() -> dict[str, object]:
        return {"x": nested}

    async def circular() -> dict[str, object]:
        return {"x": circ}

    async def huge() -> dict[str, object]:
        return {"x": "z" * 3_000_000}

    async def widelist() -> dict[str, object]:
        return {"x": list(range(2_000_000))}

    class Weird:
        def __str__(self):
            return "weird-object"

    async def weird() -> dict[str, object]:
        return {"x": Weird(), 1: "intkey"}

    async def boom() -> dict[str, object]:
        raise RuntimeError("SECRET-DB-PASSWORD=hunter2")

    async def toolerr() -> dict[str, object]:
        raise ToolError("author-controlled message")

    async def many_small() -> dict[str, object]:
        return {"x": [{"a": "b" * 10} for _ in range(300_000)]}

    handlers = {"deep": deep, "circular": circular, "huge": huge, "widelist": widelist, "weird": weird,
                "boom": boom, "toolerr": toolerr, "many_small": many_small}
    host = _HangingHost()
    mcp = _reference_server(host.port, extra_custom_handlers=handlers, identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT)
    async with Client(mcp) as c:
        for name in handlers:
            t0 = time.monotonic()
            is_err, text = await err_of(c, name, {})
            dt = time.monotonic() - t0
            leak = "hunter2" in text
            rec(f"B {name}", "FAIL" if leak else "INFO", f"isError={is_err} dt={dt:.2f}s :: {text}")
        # loop liveness while a huge bound check runs
        t0 = time.monotonic()
        res = await asyncio.gather(err_of(c, "many_small", {}), err_of(c, "qual_echo", {"message": "live"}))
        rec("B event-loop liveness during large bound", "INFO", f"echo={res[1]} total={time.monotonic()-t0:.2f}s")
        # unknown tool echo (catalog describe) - needs include invoke stub: use huge tool id
        is_err, text = await err_of(c, "describe_catalog_tool", {"tool_id": "A" * 500_000})
        rec("B describe unknown huge id", "INFO", f"isError={is_err} len={len(text)} :: {text[:160]}")


async def concurrency():
    started = asyncio.Event()
    gate = asyncio.Event()

    async def slow(delay_s: float = 0.0) -> dict[str, object]:
        await gate.wait()
        return {"ok": 1}

    host = _HangingHost()
    mcp = _reference_server(host.port, extra_custom_handlers={"slow": slow}, max_concurrent_tool_calls=2,
                            identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT)
    async with Client(mcp) as c:
        tasks = [asyncio.create_task(err_of(c, "slow", {})) for _ in range(8)]
        await asyncio.sleep(0.8)
        gate.set()
        out = await asyncio.gather(*tasks)
        refused = [o for o in out if o[0]]
        ok = [o for o in out if not o[0]]
        rec("C 8 calls cap 2", "PASS" if len(ok) == 2 and len(refused) == 6 and all("MCP_CONCURRENT_CALL_LIMIT_EXCEEDED" in r[1] for r in refused) else "FAIL", f"ok={len(ok)} refused={len(refused)} sample={refused[0][1] if refused else ''}")
        again = await asyncio.gather(*[err_of(c, "slow", {}) for _ in range(2)])
        rec("C slots free after", "PASS" if not any(a[0] for a in again) else "FAIL", again)
    # cancel while holding a slot
    gate.clear()
    async with Client(mcp) as c:
        t1 = asyncio.create_task(err_of(c, "slow", {}))
        t2 = asyncio.create_task(err_of(c, "slow", {}))
        await asyncio.sleep(0.3)
        t1.cancel(); t2.cancel()
        await asyncio.sleep(0.3)
        gate.set()
        again = await asyncio.gather(*[err_of(c, "slow", {}) for _ in range(2)])
        rec("C slots free after client-side cancel", "PASS" if not any(a[0] for a in again) else "FAIL", again)


async def deadline():
    ActiveTaskRegistry.clear_for_tests()
    # default drain (no patch): measure real wall time of a 1s bound with a non-cooperative host
    host = _StubbornHost()
    supervision = build_host_root_execution_transport_detachment_supervisor()
    mcp = _reference_server(host.port, task_control_mutation_boundary=_boundary(), max_call_seconds=1.0)
    with patch(f"{NEXUS}.resolve_mcp_run_agent_identity", return_value=_identity_resolution()), \
         patch(f"{NEXUS}.resolve_transport_detachment_supervision_from_request", return_value=supervision):
        async with Client(mcp) as c:
            t0 = time.monotonic()
            is_err, text = await err_of(c, "run_agent", {"message": "stubborn"})
            dt = time.monotonic() - t0
            rec("D1 max_call_seconds=1.0 non-cooperative host, DEFAULT drain", "INFO",
                f"wall={dt:.1f}s isError={is_err} :: {text[:160]} | drain_const={tc._HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S}")
            rec("D1 not success", "PASS" if is_err and "MCP_DEADLINE_EXCEEDED" in text else "FAIL", text[:120])
            # side effect attempt after the response: work continues under supervision (documented limit)
            host.release.set()
            await asyncio.sleep(1.0)
            rec("D1 late completion after reported deadline (documented limit)", "INFO",
                f"supervised_count={supervision.supervised_count()} cancel_seen={host.cancel_seen}")
    ActiveTaskRegistry.clear_for_tests()


async def main():
    which = sys.argv[1:] or ["bounds", "concurrency", "deadline"]
    for w in which:
        await {"bounds": bounds, "concurrency": concurrency, "deadline": deadline}[w]()
    (ROOT / ".tmp/session/mcp-rc-audit/adv/adv_bounds_deadline_results.json").write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")


asyncio.run(main())
