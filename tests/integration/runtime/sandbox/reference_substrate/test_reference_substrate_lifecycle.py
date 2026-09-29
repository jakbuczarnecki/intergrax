# © Artur Czarnecki. All rights reserved.

"""Lifecycle and cleanup integrity for reference substrate qualification harness."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.integration.runtime.sandbox.reference_substrate.backend import (
    ReferenceSandboxBackend,
    _ReferenceSessionState,
)
from tests.integration.runtime.sandbox.reference_substrate.endpoints import ReferenceEndpointServers
from tests.integration.runtime.sandbox.reference_substrate.errors import (
    ReferenceSubstrateEndpointError,
    ReferenceSubstrateLifecycleError,
)
from tests.integration.runtime.sandbox.reference_substrate.firewall import ReferenceSubstratePolicyError
from tests.integration.runtime.sandbox.reference_substrate.preflight import ReferenceSubstratePreflight
from tests.integration.runtime.sandbox.reference_substrate.topology import NetnsSessionResources


def _ok_preflight() -> ReferenceSubstratePreflight:
    return ReferenceSubstratePreflight(
        ok=True,
        block_reason=None,
        wsl_detected=False,
        linux_kernel="test",
        ip_available=True,
        netns_usable=True,
        firewall_backend="nftables",
        privileged=True,
        python_available=True,
    )


class _FakeHttpServer:
    RequestHandlerClass: type

    def serve_forever(self) -> None:
        return

    def shutdown(self) -> None:
        return

    def server_close(self) -> None:
        return


def _fake_httpserver_factory(
    server_address: tuple[str, int],
    request_handler_class: type,
) -> _FakeHttpServer:
    server = _FakeHttpServer()
    server.RequestHandlerClass = request_handler_class
    return server


def _patch_ip_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.loopback_has_ipv4_address",
        lambda _addr: False,
    )

    def _ip(command: list[str], **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.args = command
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._run_ip",
        _ip,
    )


def test_address_add_failure_does_not_claim_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.loopback_has_ipv4_address",
        lambda _addr: False,
    )

    def _fail_add(command: list[str], **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.returncode = 1
        result.stderr = "simulated add failure"
        result.stdout = ""
        result.args = command
        return result

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._run_ip",
        _fail_add,
    )
    servers = ReferenceEndpointServers()
    with pytest.raises(ReferenceSubstrateEndpointError):
        servers.start()
    assert servers.denied_addr_owned is False
    assert servers._allowed is None  # noqa: SLF001 — lifecycle contract proof


def test_partial_endpoint_startup_rolls_back_owned_address(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ip_success(monkeypatch)
    del_calls = 0

    def _ip_with_del(command: list[str], **kwargs: object) -> MagicMock:
        nonlocal del_calls
        result = MagicMock()
        result.args = command
        result.stdout = ""
        result.stderr = ""
        if command[:4] == ["ip", "addr", "del", "10.200.42.3/32"]:
            del_calls += 1
        result.returncode = 0
        return result

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._run_ip",
        _ip_with_del,
    )
    bind_attempts = 0

    def _flaky_httpserver(
        server_address: tuple[str, int],
        request_handler_class: type,
    ) -> _FakeHttpServer:
        nonlocal bind_attempts
        bind_attempts += 1
        if bind_attempts >= 2:
            raise OSError("simulated bind failure")
        return _fake_httpserver_factory(server_address, request_handler_class)

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.ThreadingHTTPServer",
        _flaky_httpserver,
    )
    servers = ReferenceEndpointServers()
    with pytest.raises(ReferenceSubstrateEndpointError):
        servers.start()
    assert del_calls == 1
    assert servers.denied_addr_owned is False


def test_normal_stop_removes_harness_owned_address(monkeypatch: pytest.MonkeyPatch) -> None:
    del_calls = 0

    def _ip(command: list[str], **kwargs: object) -> MagicMock:
        nonlocal del_calls
        result = MagicMock()
        result.args = command
        result.stdout = ""
        result.stderr = ""
        if command[:4] == ["ip", "addr", "del", "10.200.42.3/32"]:
            del_calls += 1
        result.returncode = 0
        return result

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.loopback_has_ipv4_address",
        lambda _addr: False,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._run_ip",
        _ip,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.ThreadingHTTPServer",
        _fake_httpserver_factory,
    )
    servers = ReferenceEndpointServers()
    servers.start()
    assert servers.denied_addr_owned is True
    servers.stop()
    assert servers.denied_addr_owned is False
    assert del_calls == 1


def test_double_stop_is_idempotent_without_extra_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    del_calls = 0

    def _ip(command: list[str], **kwargs: object) -> MagicMock:
        nonlocal del_calls
        result = MagicMock()
        result.args = command
        result.stdout = ""
        result.stderr = ""
        if command[:4] == ["ip", "addr", "del", "10.200.42.3/32"]:
            del_calls += 1
        result.returncode = 0
        return result

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.loopback_has_ipv4_address",
        lambda _addr: False,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._run_ip",
        _ip,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.ThreadingHTTPServer",
        _fake_httpserver_factory,
    )
    servers = ReferenceEndpointServers()
    servers.start()
    servers.stop()
    servers.stop()
    assert del_calls == 1


def test_preexisting_denied_address_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.loopback_has_ipv4_address",
        lambda _addr: True,
    )
    del_calls = 0

    def _ip(command: list[str], **kwargs: object) -> MagicMock:
        nonlocal del_calls
        result = MagicMock()
        result.args = command
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if command[:4] == ["ip", "addr", "del", "10.200.42.3/32"]:
            del_calls += 1
        return result

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._run_ip",
        _ip,
    )
    servers = ReferenceEndpointServers()
    with pytest.raises(ReferenceSubstrateEndpointError):
        servers.start()
    servers.stop()
    assert del_calls == 0
    assert servers.denied_addr_owned is False


def test_backend_close_drains_sessions_and_stops_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoints = ReferenceEndpointServers()
    monkeypatch.setattr(endpoints, "start", lambda: None)
    stop_mock = MagicMock()
    monkeypatch.setattr(endpoints, "stop", stop_mock)
    backend = ReferenceSandboxBackend(
        preflight=_ok_preflight(),
        endpoint_servers=endpoints,
    )
    resources = NetnsSessionResources(netns="igx-qual-test", veth_host="veth1h", veth_peer="veth1p")
    backend._sessions[resources.netns] = _ReferenceSessionState(resources=resources)  # noqa: SLF001
    destroy_mock = MagicMock()
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        destroy_mock,
    )
    backend.close()
    destroy_mock.assert_called_once_with(resources)
    stop_mock.assert_called_once()
    assert backend._lifecycle_closed is True  # noqa: SLF001


def test_backend_close_continues_after_session_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints = ReferenceEndpointServers()
    monkeypatch.setattr(endpoints, "start", lambda: None)
    stop_mock = MagicMock()
    monkeypatch.setattr(endpoints, "stop", stop_mock)
    backend = ReferenceSandboxBackend(
        preflight=_ok_preflight(),
        endpoint_servers=endpoints,
    )
    r1 = NetnsSessionResources(netns="igx-qual-a", veth_host="va", veth_peer="vb")
    r2 = NetnsSessionResources(netns="igx-qual-b", veth_host="vc", veth_peer="vd")
    backend._sessions[r1.netns] = _ReferenceSessionState(resources=r1)  # noqa: SLF001
    backend._sessions[r2.netns] = _ReferenceSessionState(resources=r2)  # noqa: SLF001

    def _destroy(resources: NetnsSessionResources) -> None:
        if resources.netns == "igx-qual-a":
            raise ReferenceSubstratePolicyError("simulated session cleanup failure")
        return None

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        _destroy,
    )
    with pytest.raises(ReferenceSubstrateLifecycleError):
        backend.close()
    stop_mock.assert_called_once()
    assert backend._lifecycle_closed is False  # noqa: SLF001
