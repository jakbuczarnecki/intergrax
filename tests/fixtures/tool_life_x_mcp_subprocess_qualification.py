# © Artur Czarnecki. All rights reserved.

"""Shared subprocess MCP client helpers for TOOL-LIFE-X L3-R1/R2 qualification."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_MCP_SERVER = (
    Path(__file__).resolve().parent / "tool_life_x_l3_r1_mcp_qualification_server.py"
)
_FASTMCP = pytest.importorskip("fastmcp")

_STARTUP_READ_TIMEOUT_S = 15.0
_TEARDOWN_WAIT_S = 3.0


@contextmanager
def mcp_qualification_subprocess(mode: str):
    """Start qualification FastMCP server; yield ``(base_url | None, proc | exit_code)``."""
    proc_env = os.environ.copy()
    proc_env["TOOL_LIFE_X_MCP_MODE"] = mode
    proc = subprocess.Popen(
        [sys.executable, str(_MCP_SERVER)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=proc_env,
    )
    try:
        if mode == "unavailable":
            proc.wait(timeout=5.0)
            yield None, proc.returncode
            return
        assert proc.stdout is not None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            line_future = pool.submit(proc.stdout.readline)
            try:
                line = line_future.result(timeout=_STARTUP_READ_TIMEOUT_S).strip()
            except concurrent.futures.TimeoutError:
                proc.kill()
                proc.wait(timeout=_TEARDOWN_WAIT_S)
                raise TimeoutError("mcp_qualification_subprocess_startup_timeout") from None
        if not line:
            err = proc.stderr.read() if proc.stderr is not None else ""
            proc.kill()
            raise RuntimeError(f"mcp_qualification_subprocess_no_port_line stderr={err!r}")
        port = int(line)
        base = f"http://127.0.0.1:{port}/"
        yield base, proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_TEARDOWN_WAIT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=_TEARDOWN_WAIT_S)


async def mcp_call_tool(
    base_url: str,
    tool: str,
    arguments: dict,
    *,
    timeout_s: float,
) -> object:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _run() -> object:
        async with streamablehttp_client(base_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool, arguments)

    return await asyncio.wait_for(_run(), timeout=timeout_s)
