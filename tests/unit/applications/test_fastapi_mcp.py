# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import FastMCP

from intergrax.applications._shared.fastapi_mcp import (
    apply_lifespans,
    couple_fastapi_with_mcp,
    make_scheduler_lifespan,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_couple_fastapi_with_mcp_serves_both_apps() -> None:
    api = FastAPI()

    @api.get("/hello")
    def hello() -> dict[str, str]:
        return {"ok": "api"}

    mcp = FastMCP("TestMCP")

    @mcp.tool
    def ping() -> str:
        return "pong"

    combined = couple_fastapi_with_mcp(api, mcp, mount_path="/mcp")
    client = TestClient(combined)

    assert client.get("/hello").json() == {"ok": "api"}
    # MCP mount is registered (exact path depends on transport; API must stay reachable)
    assert any(
        getattr(r, "path", None) in {"/mcp", "/mcp/"}
        for r in combined.routes
        if hasattr(r, "path")
    )


def test_apply_lifespans_starts_scheduler_without_on_event() -> None:
    class _Scheduler:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

    scheduler = _Scheduler()
    app = FastAPI()
    apply_lifespans(app, make_scheduler_lifespan(scheduler))

    with TestClient(app) as client:
        assert client.app is app
        assert scheduler.started is True
        assert scheduler.stopped is False

    assert scheduler.stopped is True
