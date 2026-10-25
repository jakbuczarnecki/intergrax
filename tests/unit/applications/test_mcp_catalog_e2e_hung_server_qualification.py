# © Artur Czarnecki. All rights reserved.

"""TLX-TL11 — physical MCP catalog discovery + hung-server lifecycle (streamable HTTP subprocess)."""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.fixtures.tool_life_x_tl11_catalog_subprocess_qualification import (
    mcp_call_catalog_tool,
    tl11_catalog_qualification_subprocess,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_QUAL_API_KEY = "tl11-qual-secret"


def test_tl11_e2e_catalog_normal_list_and_describe_subprocess() -> None:
    with tl11_catalog_qualification_subprocess("ok") as (base, proc):
        assert base is not None
        listed = asyncio.run(
            mcp_call_catalog_tool(base, "list_catalog_tools", {}, timeout_s=8.0)
        )
        described = asyncio.run(
            mcp_call_catalog_tool(
                base,
                "describe_catalog_tool",
                {"tool_id": "tl11.qual.one"},
                timeout_s=8.0,
            )
        )
        assert proc.poll() is None
    assert "tl11.qual.one" in str(listed)
    assert "tl11.qual.one" in str(described)


def test_tl11_e2e_hung_list_catalog_client_timeout_not_definite_failure() -> None:
    """Client ``wait_for`` timeout — does not prove server process was killed."""
    with tl11_catalog_qualification_subprocess("hang_list") as (base, proc):
        assert base is not None
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(
                mcp_call_catalog_tool(base, "list_catalog_tools", {}, timeout_s=0.6)
            )
        proc.terminate()
        proc.wait(timeout=5.0)


def test_tl11_e2e_hung_describe_catalog_client_timeout() -> None:
    with tl11_catalog_qualification_subprocess("hang_describe") as (base, proc):
        assert base is not None
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(
                mcp_call_catalog_tool(
                    base,
                    "describe_catalog_tool",
                    {"tool_id": "tl11.qual.one"},
                    timeout_s=0.6,
                )
            )
        proc.terminate()
        proc.wait(timeout=5.0)


def test_tl11_e2e_slow_list_within_deadline() -> None:
    with tl11_catalog_qualification_subprocess("slow_list") as (base, _proc):
        assert base is not None
        result = asyncio.run(
            mcp_call_catalog_tool(base, "list_catalog_tools", {}, timeout_s=8.0)
        )
    assert "tl11.qual.one" in str(result)


def test_tl11_e2e_server_crash_mid_catalog_list() -> None:
    with tl11_catalog_qualification_subprocess("crash_list") as (base, proc):
        assert base is not None
        with pytest.raises(Exception):
            asyncio.run(
                mcp_call_catalog_tool(base, "list_catalog_tools", {}, timeout_s=5.0)
            )
        time.sleep(0.4)
        assert proc.poll() is not None


def test_tl11_e2e_server_crash_mid_catalog_describe() -> None:
    with tl11_catalog_qualification_subprocess("crash_describe") as (base, proc):
        assert base is not None
        with pytest.raises(Exception):
            asyncio.run(
                mcp_call_catalog_tool(
                    base,
                    "describe_catalog_tool",
                    {"tool_id": "tl11.qual.one"},
                    timeout_s=5.0,
                )
            )
        time.sleep(0.4)
        assert proc.poll() is not None


def test_tl11_e2e_client_cancel_during_hung_catalog_list() -> None:
    async def _cancelled() -> None:
        with tl11_catalog_qualification_subprocess("hang_list") as (base, proc):
            assert base is not None
            task = asyncio.create_task(
                mcp_call_catalog_tool(base, "list_catalog_tools", {}, timeout_s=30.0)
            )
            await asyncio.sleep(0.25)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            proc.terminate()

    asyncio.run(_cancelled())


def test_tl11_e2e_reconnect_after_catalog_timeout_fresh_session() -> None:
    with tl11_catalog_qualification_subprocess("slow_list") as (base, proc):
        assert base is not None
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(
                mcp_call_catalog_tool(base, "list_catalog_tools", {}, timeout_s=0.5)
            )
        proc.terminate()
        proc.wait(timeout=5.0)
    with tl11_catalog_qualification_subprocess("ok") as (base2, _proc2):
        assert base2 is not None
        result = asyncio.run(
            mcp_call_catalog_tool(base2, "list_catalog_tools", {}, timeout_s=8.0)
        )
    assert "tl11.qual.one" in str(result)


def test_tl11_e2e_subprocess_cleanup_after_hung_catalog() -> None:
    with tl11_catalog_qualification_subprocess("hang_list") as (base, proc):
        assert base is not None
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(
                mcp_call_catalog_tool(base, "list_catalog_tools", {}, timeout_s=0.5)
            )
    time.sleep(0.2)
    assert proc.poll() is not None


def test_tl11_e2e_no_indefinite_wait_on_never_ready_startup() -> None:
    with pytest.raises(TimeoutError, match="startup_timeout"):
        with tl11_catalog_qualification_subprocess("never_ready"):
            pass


def test_tl11_e2e_mcp_server_unavailable_exit_code() -> None:
    with tl11_catalog_qualification_subprocess("unavailable") as (base, code):
        assert base is None
        assert code == 2


def test_tl11_e2e_http_ingress_rejects_missing_api_key() -> None:
    with tl11_catalog_qualification_subprocess("ok", require_auth=True) as (base, proc):
        assert base is not None
        with pytest.raises(Exception):
            asyncio.run(
                mcp_call_catalog_tool(base, "list_catalog_tools", {}, timeout_s=5.0)
            )
        assert proc.poll() is None


def test_tl11_e2e_http_ingress_accepts_authenticated_catalog_discovery() -> None:
    with tl11_catalog_qualification_subprocess("ok", require_auth=True) as (base, proc):
        assert base is not None
        result = asyncio.run(
            mcp_call_catalog_tool(
                base,
                "list_catalog_tools",
                {},
                timeout_s=8.0,
                api_key=_QUAL_API_KEY,
            )
        )
        assert proc.poll() is None
    assert "tl11.qual.one" in str(result)


def test_tl11_e2e_concurrent_catalog_discovery_isolated() -> None:
    async def _parallel() -> tuple[object, object]:
        with tl11_catalog_qualification_subprocess("ok") as (base, proc):
            assert base is not None
            left, right = await asyncio.gather(
                mcp_call_catalog_tool(base, "list_catalog_tools", {}, timeout_s=8.0),
                mcp_call_catalog_tool(
                    base,
                    "describe_catalog_tool",
                    {"tool_id": "tl11.qual.one"},
                    timeout_s=8.0,
                ),
            )
            assert proc.poll() is None
            return left, right

    listed, described = asyncio.run(_parallel())
    assert "tl11.qual.one" in str(listed)
    assert "tl11.qual.one" in str(described)
