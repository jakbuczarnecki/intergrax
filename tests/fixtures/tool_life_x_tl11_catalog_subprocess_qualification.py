# © Artur Czarnecki. All rights reserved.

"""Subprocess helpers for TLX-TL11 Nexus catalog discovery physical qualification."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_TL11_SERVER = (
    Path(__file__).resolve().parent / "tool_life_x_tl11_nexus_catalog_qualification_server.py"
)
pytest.importorskip("fastmcp")

_STARTUP_READ_TIMEOUT_S = 15.0
_TEARDOWN_WAIT_S = 3.0


@contextmanager
def tl11_catalog_qualification_subprocess(
    mode: str,
    *,
    require_auth: bool = False,
):
    """Start TL11 Nexus catalog server; yield ``(base_url | None, proc | exit_code)``."""
    proc_env = os.environ.copy()
    proc_env["TOOL_LIFE_X_TL11_MODE"] = mode
    if require_auth:
        proc_env["TOOL_LIFE_X_TL11_REQUIRE_AUTH"] = "1"
    else:
        proc_env.pop("TOOL_LIFE_X_TL11_REQUIRE_AUTH", None)
    proc = subprocess.Popen(
        [sys.executable, str(_TL11_SERVER)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=proc_env,
    )
    try:
        if mode == "unavailable":
            proc.wait(timeout=60.0)
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
                raise TimeoutError("tl11_catalog_qualification_subprocess_startup_timeout") from None
        if not line:
            err = proc.stderr.read() if proc.stderr is not None else ""
            proc.kill()
            raise RuntimeError(f"tl11_catalog_qualification_subprocess_no_port_line stderr={err!r}")
        port = int(line)
        suffix = "/mcp/" if require_auth else "/"
        base = f"http://127.0.0.1:{port}{suffix}"
        yield base, proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_TEARDOWN_WAIT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=_TEARDOWN_WAIT_S)


async def mcp_call_catalog_tool(
    base_url: str,
    tool: str,
    arguments: dict,
    *,
    timeout_s: float,
    api_key: str | None = None,
) -> object:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"X-Api-Key": api_key} if api_key else None

    async def _run() -> object:
        async with streamablehttp_client(base_url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool, arguments)

    return await asyncio.wait_for(_run(), timeout=timeout_s)
