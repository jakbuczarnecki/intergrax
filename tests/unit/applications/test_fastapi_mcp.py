# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import FastMCP

from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp

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
