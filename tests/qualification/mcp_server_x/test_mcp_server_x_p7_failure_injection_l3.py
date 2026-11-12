# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P7 (MCP-P7-05, ticket T-IGX-035-H) - failure injection, level L3.

Level: **L3 - physical streamable HTTP on loopback** (``uvicorn`` on an OS-assigned port, the
official MCP streamable HTTP client, plus raw ``httpx`` for a socket-level disconnect). The
reference server is composed from the reference qualification builder and coupled to a wrapper app
with the production helper ``couple_fastapi_with_mcp`` (task-control mutation boundary and transport
detachment supervisor wired, API-key auth on the inner app). The host execution port is an
inspectable ``AsyncMock`` of ``HostTaskExecutionPort`` whose ``execute`` hangs until a governed
cancel is requested; the caller principal is therefore the real one resolved from the HTTP request.

Scenarios: MCP-ACC-13 (cancel), MCP-ACC-14 (timeout: client-side deadlines; the transport carries a
request timeout, the server has no per-request execution deadline - see the L2 module for the
N/A record of a canonical-execution deadline), MCP-ACC-15 (disconnect), plus limit exceeded and
unknown tool of AC-P7-04. Every call runs under a deadline and every scenario asserts a terminal
state, so nothing can hang the suite. Contract-stub / double scope: the host port is a double, so
these tests prove the transport, principal and governed-cancel wiring, not an agent run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_control_plane_mutation_boundary,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
    build_host_root_execution_transport_detachment_supervisor,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from mcp import types
from mcp.shared.exceptions import McpError
from tests.qualification.mcp_server_x.reference_app.build import (
    build_mcp_server_x_reference_qualification_server,
)
from tests.qualification.mcp_server_x.reference_app.http_serve import (
    ServedMcp,
    error_text,
    list_tool_names,
    served_mcp,
    streamable_session,
    tool_payload,
)
from tests.qualification.mcp_server_x.reference_app.variants import (
    CATALOG_TOOL_ID,
    QUAL_API_KEY,
    RecordingCatalogInvoker,
    api_key_auth,
    build_catalog_registry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.network, pytest.mark.qualification]

_KEY_HEADERS = {"X-Api-Key": QUAL_API_KEY}
_DEADLINE_S = 15.0
_TERMINAL_WAIT_S = 12.0
_TRANSPORT = "intergrax.applications._shared.mcp_host_execution_transport_cancel"
_LIMIT_CODE = "mcp_concurrent_call_limit_exceeded"


@pytest.fixture(autouse=True)
def _clean_active_tasks_and_fast_drain(monkeypatch: pytest.MonkeyPatch):
    ActiveTaskRegistry.clear_for_tests()
    monkeypatch.setattr(f"{_TRANSPORT}._HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S", 0.25)
    yield
    ActiveTaskRegistry.clear_for_tests()


# --------------------------------------------------------------------------- server probe + helpers


@dataclass
class _Probe:
    """Cross-thread observations of what happened on the server side (uvicorn runs in its own thread)."""

    hang_started: threading.Event = field(default_factory=threading.Event)
    governed_cancel_seen: threading.Event = field(default_factory=threading.Event)
    run_finished: threading.Event = field(default_factory=threading.Event)
    slow_started: threading.Event = field(default_factory=threading.Event)
    slow_done: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)  # frees qual_slow
    stop: threading.Event = field(default_factory=threading.Event)  # safety valve for the hanging host
    tasks: list[Task] = field(default_factory=list)
    run_outcomes: list[TaskState] = field(default_factory=list)
    slow_outcomes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Served:
    served: ServedMcp
    probe: _Probe
    port: AsyncMock
    supervision: object

    @property
    def url(self) -> str:
        return self.served.url


def _harness_boundary():
    env = ApplicationEnvironmentProfile(application_profile=ApplicationProfile.PRODUCT)
    return build_harness_control_plane_mutation_boundary(env)


@contextmanager
def _serve_reference(**builder_kwargs: object) -> Iterator[_Served]:
    """Reference server on loopback with a hanging ``run_agent`` host and a probe-instrumented ``qual_slow``."""
    probe = _Probe()

    async def _hang(task: Task, **_kwargs: object) -> TaskResult:
        run_id = mint_run_id()
        probe.tasks.append(task)
        await ActiveTaskRegistry.register(task, run_id)
        probe.hang_started.set()
        cancelled = False
        try:
            while not CancellationCoordinator.is_requested(task.metadata) and not probe.stop.is_set():
                await asyncio.sleep(0.02)
            cancelled = CancellationCoordinator.is_requested(task.metadata)
            if cancelled:
                probe.governed_cancel_seen.set()
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)
            probe.run_finished.set()
        state = TaskState.CANCELLED if cancelled else TaskState.COMPLETED
        probe.run_outcomes.append(state)
        return TaskResult(
            task_id=task.task_id,
            run_id=str(run_id),
            state=state,
            answer="",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )

    async def _qual_slow(delay_s: float = 0.0) -> dict[str, object]:
        probe.slow_started.set()
        deadline = time.monotonic() + delay_s
        try:
            while time.monotonic() < deadline and not probe.release.is_set():
                await asyncio.sleep(0.02)
            probe.slow_outcomes.append("released" if probe.release.is_set() else "completed")
        except asyncio.CancelledError:
            probe.slow_outcomes.append("cancelled")
            raise
        finally:
            probe.slow_done.set()
        return {"slept": delay_s}

    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(side_effect=_hang)
    boundary = _harness_boundary()
    supervision = build_host_root_execution_transport_detachment_supervisor()
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=port,
        registry=AgentRegistry(),
        task_control_mutation_boundary=boundary,
        extra_custom_handlers={"qual_slow": _qual_slow},
        **builder_kwargs,  # type: ignore[arg-type]
    )
    api = FastAPI()
    api.state.harness_auth = api_key_auth()
    app = couple_fastapi_with_mcp(
        api,
        mcp,
        mount_path="/mcp",
        task_control_mutation_boundary=boundary,
        transport_detachment_supervision=supervision,
    )
    try:
        with served_mcp(app) as served:
            try:
                yield _Served(served=served, probe=probe, port=port, supervision=supervision)
            finally:
                probe.release.set()
                probe.stop.set()
    finally:
        probe.release.set()
        probe.stop.set()


async def _await_event(event: threading.Event, what: str, *, timeout_s: float = _TERMINAL_WAIT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while not event.is_set():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout_s}s waiting for: {what}")
        await asyncio.sleep(0.02)


async def _await_supervision_drained(supervision: object, *, timeout_s: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_s
    while supervision.supervised_count() > 0 and time.monotonic() < deadline:  # type: ignore[attr-defined]
        await asyncio.sleep(0.02)
    assert supervision.supervised_count() == 0  # type: ignore[attr-defined]


async def _assert_terminal_cancelled(env: _Served) -> None:
    """The hanging run ended in a governed, terminal CANCELLED state and nothing stayed bound or supervised."""
    await _await_event(env.probe.governed_cancel_seen, "governed cancel to reach the running task")
    await _await_event(env.probe.run_finished, "the cancelled run to finish")
    (task,) = env.probe.tasks
    assert CancellationCoordinator.is_requested(task.metadata)
    assert env.probe.run_outcomes == [TaskState.CANCELLED]
    await _await_supervision_drained(env.supervision)


async def _assert_server_still_usable(url: str) -> None:
    """A fresh client session on the same server completes discovery and a tool call within the deadline."""
    names = await list_tool_names(url, headers=_KEY_HEADERS)
    assert "qual_echo" in names
    async with streamable_session(url, headers=_KEY_HEADERS) as (session, _init):
        result = await asyncio.wait_for(session.call_tool("qual_echo", {"message": "still-up"}), timeout=_DEADLINE_S)
    assert result.isError is not True, error_text(result)
    assert tool_payload(result)["echo"] == "still-up"


def _cancelled_notification(request_id: int, reason: str) -> types.ClientNotification:
    return types.ClientNotification(
        types.CancelledNotification(
            method="notifications/cancelled",
            params=types.CancelledNotificationParams(requestId=request_id, reason=reason),
        )
    )


async def _settle(call: asyncio.Task) -> None:
    """Retire a client call task that will never receive a response (cancelled by the caller)."""
    call.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(call, timeout=5.0)


# --------------------------------------------------------------------------- MCP-ACC-13 (cancel)


async def test_acc_13_client_cancel_notification_mid_invoke_reaches_governed_cancel() -> None:
    """The client sends ``notifications/cancelled`` for an in-flight ``run_agent``; the session stays open."""
    with _serve_reference() as env:
        async with streamable_session(env.url, headers=_KEY_HEADERS) as (session, _init):
            call = asyncio.create_task(session.call_tool("run_agent", {"message": "hang"}))
            await _await_event(env.probe.hang_started, "run_agent to start on the host")
            # initialize was request 0 on this fresh session, so the in-flight call is request 1
            await session.send_notification(_cancelled_notification(1, "p7 failure injection: client cancel"))
            await _assert_terminal_cancelled(env)
            # a cancelled request gets no response (MCP spec); the client-side call is retired explicitly
            await _settle(call)
        await _assert_server_still_usable(env.url)


async def test_acc_13_unknown_task_id_invoke_over_http_fails_closed_without_running_the_invoker() -> None:
    """``invoke_catalog_tool`` with an unknown task id over HTTP is a typed fail-closed error."""
    invoker = RecordingCatalogInvoker()
    with _serve_reference(tool_registry=build_catalog_registry(), catalog_tool_invoker=invoker) as env:
        async with streamable_session(env.url, headers=_KEY_HEADERS) as (session, _init):
            result = await asyncio.wait_for(
                session.call_tool(
                    "invoke_catalog_tool",
                    {"tool_id": CATALOG_TOOL_ID, "task_id": str(mint_task_id()), "run_id": str(mint_run_id())},
                ),
                timeout=_DEADLINE_S,
            )
    assert result.isError is True
    assert "missing_execution_binding" in (error_text(result) or "")
    assert invoker.requests == []
    assert not invoker.started.is_set()


# --------------------------------------------------------------------------- MCP-ACC-14 (timeout)


async def test_acc_14_client_deadline_on_slow_tool_ends_client_side_and_handler_reaches_terminal_state() -> None:
    """``asyncio.wait_for`` on a slow reference tool: the client call ends (TimeoutError), the server is not stuck."""
    with _serve_reference() as env:
        async with streamable_session(env.url, headers=_KEY_HEADERS) as (session, _init):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(session.call_tool("qual_slow", {"delay_s": 2.0}), timeout=0.5)
            assert env.probe.slow_started.is_set()
            # RECORDED BEHAVIOUR: a client-side deadline does not cancel server work (the SDK sends no
            # cancel notification when its local wait is abandoned), so the handler runs on to its own
            # terminal state ("completed"). There is no per-request server-side deadline on this tool.
            await _await_event(env.probe.slow_done, "the slow handler to reach a terminal state")
            assert env.probe.slow_outcomes == ["completed"]
            # the same session is still usable: no hang, no poisoned connection
            echoed = await asyncio.wait_for(
                session.call_tool("qual_echo", {"message": "same-session"}), timeout=_DEADLINE_S
            )
            assert tool_payload(echoed)["echo"] == "same-session"


async def test_acc_14_sdk_request_timeout_returns_mcp_request_timeout_error() -> None:
    """The MCP client request timeout (``read_timeout_seconds``) is carried by the transport as error 408."""
    with _serve_reference() as env:
        async with streamable_session(env.url, headers=_KEY_HEADERS) as (session, _init):
            with pytest.raises(McpError) as caught:
                await asyncio.wait_for(
                    session.call_tool("qual_slow", {"delay_s": 3.0}, read_timeout_seconds=timedelta(seconds=0.5)),
                    timeout=_DEADLINE_S,
                )
            assert caught.value.error.code == httpx.codes.REQUEST_TIMEOUT
            await _await_event(env.probe.slow_done, "the slow handler to reach a terminal state")
            follow_up = await asyncio.wait_for(
                session.call_tool("qual_echo", {"message": "after-408"}), timeout=_DEADLINE_S
            )
            assert tool_payload(follow_up)["echo"] == "after-408"


async def test_acc_14_client_deadline_on_hanging_run_agent_ends_in_governed_cancel() -> None:
    """A client deadline on a hanging ``run_agent`` plus session close ends the run as a governed CANCELLED."""
    with _serve_reference() as env:
        async with streamable_session(env.url, headers=_KEY_HEADERS) as (session, _init):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(session.call_tool("run_agent", {"message": "hang"}), timeout=1.0)
            assert env.probe.hang_started.is_set()
        # session closed after the deadline: the abandoned run must not stay live
        await _assert_terminal_cancelled(env)
        await _assert_server_still_usable(env.url)


# --------------------------------------------------------------------------- MCP-ACC-15 (disconnect)


async def test_acc_15_client_transport_teardown_mid_invoke_admits_governed_cancel() -> None:
    """The client abandons the in-flight ``run_agent`` by closing its transport; no cancel notification is sent."""
    with _serve_reference() as env:
        async with streamable_session(env.url, headers=_KEY_HEADERS) as (session, _init):
            call = asyncio.create_task(session.call_tool("run_agent", {"message": "hang"}))
            await _await_event(env.probe.hang_started, "run_agent to start on the host")
            call.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(call, timeout=5.0)
        await _assert_terminal_cancelled(env)
        await _assert_server_still_usable(env.url)


def _jsonrpc_messages(response: httpx.Response) -> list[dict[str, object]]:
    """JSON-RPC messages of a streamable HTTP response (plain JSON or an SSE ``data:`` stream)."""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        found: list[dict[str, object]] = []
        for line in response.text.splitlines():
            if line.startswith("data:"):
                with contextlib.suppress(ValueError):
                    found.append(json.loads(line[len("data:") :].strip()))
        return found
    return [response.json()]


async def _raw_initialize(http: httpx.AsyncClient, url: str) -> dict[str, str]:
    """Raw MCP handshake on a client that does NOT follow redirects; returns the session headers."""
    base = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json", **_KEY_HEADERS}
    init = await http.post(
        url,
        json={
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": types.LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "p7-fi-raw-client", "version": "0"},
            },
        },
        headers=base,
    )
    assert init.status_code == 200, init.text
    session_headers = {**base, "mcp-session-id": init.headers["mcp-session-id"]}
    ack = await http.post(url, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=session_headers)
    assert ack.status_code in {200, 202}, ack.text
    return session_headers


async def test_acc_15_raw_socket_close_mid_invoke_admits_governed_cancel() -> None:
    """A raw HTTP client closes the connection while the ``run_agent`` response is still open (no MCP-level goodbye)."""
    with _serve_reference() as env:
        async with httpx.AsyncClient(timeout=_DEADLINE_S, follow_redirects=False) as http:
            headers = await _raw_initialize(http, env.url)
            request = http.build_request(
                "POST",
                env.url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "run_agent", "arguments": {"message": "hang"}},
                },
                headers=headers,
            )
            response = await http.send(request, stream=True)
            assert response.status_code == 200
            await _await_event(env.probe.hang_started, "run_agent to start on the host")
            await response.aclose()  # drops the connection with the response unread
        await _assert_terminal_cancelled(env)
        await _assert_server_still_usable(env.url)


# --------------------------------------------------------------------------- AC-P7-04: limit exceeded


async def test_p7_05_limit_exceeded_second_concurrent_call_is_rejected_over_http_and_slot_is_released() -> None:
    """``max_concurrent_tool_calls=1``: the 2nd in-flight call is a typed error result, never a hang."""
    with _serve_reference(max_concurrent_tool_calls=1) as env:
        async with streamable_session(env.url, headers=_KEY_HEADERS) as (session, _init):
            first = asyncio.create_task(session.call_tool("qual_slow", {"delay_s": 30.0}))
            await _await_event(env.probe.slow_started, "qual_slow to hold the only admission slot")
            on_builtin = await asyncio.wait_for(session.call_tool("list_agents", {}), timeout=_DEADLINE_S)
            on_custom = await asyncio.wait_for(session.call_tool("qual_echo", {"message": "over"}), timeout=_DEADLINE_S)
            for rejected in (on_builtin, on_custom):
                assert rejected.isError is True
                assert _LIMIT_CODE in (error_text(rejected) or "")
            env.probe.release.set()
            held = await asyncio.wait_for(first, timeout=_DEADLINE_S)
            assert held.isError is not True
            assert env.probe.slow_outcomes == ["released"]
            again = await asyncio.wait_for(
                session.call_tool("qual_echo", {"message": "slot-free"}), timeout=_DEADLINE_S
            )
            assert tool_payload(again)["echo"] == "slot-free"


# --------------------------------------------------------------------------- AC-P7-04: unknown tool


async def test_p7_05_unknown_tool_over_http_is_an_error_result_and_server_stays_usable() -> None:
    """An unpublished tool name is a protocol-visible error; no other tool runs in its place."""
    with _serve_reference() as env:
        async with streamable_session(env.url, headers=_KEY_HEADERS) as (session, _init):
            result = await asyncio.wait_for(session.call_tool("no_such_tool", {"message": "x"}), timeout=_DEADLINE_S)
        assert result.isError is True
        assert "unknown tool" in (error_text(result) or "").lower()
        env.port.execute.assert_not_awaited()
        await _assert_server_still_usable(env.url)


async def test_p7_05_unknown_tool_over_raw_http_is_not_a_redirect() -> None:
    """Raw HTTP without redirect following: the unknown-tool call is a 200 JSON-RPC answer, never a 3xx."""
    with _serve_reference() as env:
        async with httpx.AsyncClient(timeout=_DEADLINE_S, follow_redirects=False) as http:
            headers = await _raw_initialize(http, env.url)
            response = await http.post(
                env.url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "no_such_tool", "arguments": {}},
                },
                headers=headers,
            )
        assert response.status_code == 200
        assert "location" not in response.headers
        (message,) = _jsonrpc_messages(response)
        # either a JSON-RPC error or a tool result flagged isError; in both cases a clear unknown-tool signal
        outcome = message.get("error") or message.get("result")
        assert outcome is not None
        assert "unknown tool" in json.dumps(outcome).lower()
        if "result" in message:
            assert message["result"]["isError"] is True  # type: ignore[index]
        env.port.execute.assert_not_awaited()
