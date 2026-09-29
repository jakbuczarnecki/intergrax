# © Artur Czarnecki. All rights reserved.

"""Lifecycle and cleanup integrity for reference substrate qualification harness."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from tests.integration.runtime.sandbox.reference_substrate.backend import ReferenceSandboxBackend
from tests.integration.runtime.sandbox.reference_substrate.endpoints import ReferenceEndpointServers
from tests.integration.runtime.sandbox.reference_substrate.errors import (
    ReferenceSubstrateEndpointCleanupError,
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
    shutdown_calls = 0
    close_calls = 0
    close_fail_once = False
    shutdown_fail_once = False

    def serve_forever(self) -> None:
        return

    def shutdown(self) -> None:
        if self.shutdown_fail_once:
            self.shutdown_fail_once = False
            raise OSError("simulated shutdown failure")
        type(self).shutdown_calls += 1

    def server_close(self) -> None:
        if self.close_fail_once:
            self.close_fail_once = False
            raise OSError("simulated close failure")
        type(self).close_calls += 1


def _reset_fake_server_metrics() -> None:
    _FakeHttpServer.shutdown_calls = 0
    _FakeHttpServer.close_calls = 0


def _fake_httpserver_factory(
    server_address: tuple[str, int],
    request_handler_class: type,
) -> _FakeHttpServer:
    server = _FakeHttpServer()
    server.RequestHandlerClass = request_handler_class
    return server


def _patch_denied_thread_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the second Thread.start() (denied serve_forever), not Thread construction."""
    original_thread = threading.Thread
    start_calls = 0

    class _PatchedThread(original_thread):
        def start(self) -> None:
            nonlocal start_calls
            start_calls += 1
            if start_calls >= 2:
                raise RuntimeError("simulated denied thread start failure")
            super().start()

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.threading.Thread",
        _PatchedThread,
    )


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


def _netns_resources(netns: str) -> NetnsSessionResources:
    return NetnsSessionResources(
        netns=netns,
        veth_host=f"{netns}-h",
        veth_peer=f"{netns}-p",
    )


def _backend_with_stub_endpoints(monkeypatch: pytest.MonkeyPatch) -> ReferenceSandboxBackend:
    endpoints = ReferenceEndpointServers()
    monkeypatch.setattr(endpoints, "start", lambda: None)
    return ReferenceSandboxBackend(
        preflight=_ok_preflight(),
        endpoint_servers=endpoints,
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


def test_partial_endpoint_startup_second_server_construct_fails_close_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1: unstarted server rollback uses server_close only, never shutdown."""
    _reset_fake_server_metrics()
    _patch_ip_success(monkeypatch)
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
    assert _FakeHttpServer.shutdown_calls == 0
    assert _FakeHttpServer.close_calls == 1
    assert servers.denied_addr_owned is False


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


def test_partial_startup_second_thread_start_fails_started_server_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2: started server gets shutdown+close; unstarted gets close-only."""
    _reset_fake_server_metrics()
    _patch_ip_success(monkeypatch)
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.ThreadingHTTPServer",
        _fake_httpserver_factory,
    )
    _patch_denied_thread_start_failure(monkeypatch)
    servers = ReferenceEndpointServers()
    with pytest.raises(ReferenceSubstrateEndpointError) as exc_info:
        servers.start()
    assert "simulated denied thread start failure" in str(exc_info.value)
    assert _FakeHttpServer.shutdown_calls == 1
    assert _FakeHttpServer.close_calls == 2
    assert servers.denied_addr_owned is False


def test_partial_startup_unstarted_server_close_failure_retains_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1: failed close on unstarted allowed server stays owned; stop() retries close-only."""
    _reset_fake_server_metrics()
    _patch_ip_success(monkeypatch)
    bind_attempts = 0
    retained_allowed: _FakeHttpServer | None = None

    def _flaky_httpserver(
        server_address: tuple[str, int],
        request_handler_class: type,
    ) -> _FakeHttpServer:
        nonlocal bind_attempts, retained_allowed
        bind_attempts += 1
        if bind_attempts >= 2:
            raise OSError("simulated bind failure")
        server = _fake_httpserver_factory(server_address, request_handler_class)
        server.close_fail_once = True
        retained_allowed = server
        return server

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.ThreadingHTTPServer",
        _flaky_httpserver,
    )
    servers = ReferenceEndpointServers()
    with pytest.raises(ReferenceSubstrateEndpointError) as exc_info:
        servers.start()
    assert "rollback incomplete" in str(exc_info.value)
    assert _FakeHttpServer.shutdown_calls == 0
    assert _FakeHttpServer.close_calls == 0
    assert retained_allowed is not None
    servers.stop()
    assert _FakeHttpServer.shutdown_calls == 0
    assert _FakeHttpServer.close_calls == 1


def test_partial_startup_started_server_cleanup_failure_retains_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2: failed shutdown on started allowed server stays owned; stop() retries shutdown+close."""
    _reset_fake_server_metrics()
    _patch_ip_success(monkeypatch)
    allowed_server: _FakeHttpServer | None = None

    def _tracking_httpserver(
        server_address: tuple[str, int],
        request_handler_class: type,
    ) -> _FakeHttpServer:
        nonlocal allowed_server
        server = _fake_httpserver_factory(server_address, request_handler_class)
        if server_address[0] != "10.200.42.3":
            allowed_server = server
            server.shutdown_fail_once = True
        return server

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.ThreadingHTTPServer",
        _tracking_httpserver,
    )
    _patch_denied_thread_start_failure(monkeypatch)
    servers = ReferenceEndpointServers()
    with pytest.raises(ReferenceSubstrateEndpointError) as exc_info:
        servers.start()
    assert "rollback incomplete" in str(exc_info.value)
    assert allowed_server is not None
    assert _FakeHttpServer.shutdown_calls == 0
    assert _FakeHttpServer.close_calls == 2
    servers.stop()
    assert _FakeHttpServer.shutdown_calls == 1
    assert _FakeHttpServer.close_calls == 3


def test_partial_startup_rollback_independent_server_cleanup_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3: sibling server cleanup success does not retain; failed server does."""
    _reset_fake_server_metrics()
    _patch_ip_success(monkeypatch)
    denied_server: _FakeHttpServer | None = None

    def _tracking_httpserver(
        server_address: tuple[str, int],
        request_handler_class: type,
    ) -> _FakeHttpServer:
        nonlocal denied_server
        server = _fake_httpserver_factory(server_address, request_handler_class)
        if server_address[0] == "10.200.42.3":
            denied_server = server
            server.close_fail_once = True
        return server

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.ThreadingHTTPServer",
        _tracking_httpserver,
    )
    _patch_denied_thread_start_failure(monkeypatch)
    servers = ReferenceEndpointServers()
    with pytest.raises(ReferenceSubstrateEndpointError):
        servers.start()
    assert _FakeHttpServer.shutdown_calls == 1
    assert _FakeHttpServer.close_calls == 1
    assert denied_server is not None
    servers.stop()
    assert _FakeHttpServer.shutdown_calls == 1
    assert _FakeHttpServer.close_calls == 2


def test_partial_startup_server_and_address_cleanup_failure_both_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T4: server close failure and address delete failure both retained for stop()."""
    _reset_fake_server_metrics()
    del_failures_remaining = 2

    def _ip(command: list[str], **kwargs: object) -> MagicMock:
        nonlocal del_failures_remaining
        result = MagicMock()
        result.args = command
        result.stdout = ""
        result.stderr = ""
        if command[:4] == ["ip", "addr", "del", "10.200.42.3/32"] and del_failures_remaining > 0:
            del_failures_remaining -= 1
            result.returncode = 1
            result.stderr = "rollback del failure"
            return result
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
    bind_attempts = 0

    def _flaky_httpserver(
        server_address: tuple[str, int],
        request_handler_class: type,
    ) -> _FakeHttpServer:
        nonlocal bind_attempts
        bind_attempts += 1
        if bind_attempts >= 2:
            raise OSError("simulated bind failure")
        server = _fake_httpserver_factory(server_address, request_handler_class)
        server.close_fail_once = True
        return server

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.ThreadingHTTPServer",
        _flaky_httpserver,
    )
    servers = ReferenceEndpointServers()
    with pytest.raises(ReferenceSubstrateEndpointError) as exc_info:
        servers.start()
    assert "rollback incomplete" in str(exc_info.value)
    assert servers.denied_addr_owned is True
    with pytest.raises(ReferenceSubstrateEndpointCleanupError):
        servers.stop()
    assert servers.denied_addr_owned is True
    servers.stop()
    assert servers.denied_addr_owned is False


def test_partial_startup_successful_rollback_no_owned_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T5: successful rollback leaves no servers; stop() is idempotent."""
    _reset_fake_server_metrics()
    _patch_ip_success(monkeypatch)
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
    assert _FakeHttpServer.shutdown_calls == 0
    assert _FakeHttpServer.close_calls == 1
    servers.stop()
    assert _FakeHttpServer.shutdown_calls == 0
    assert _FakeHttpServer.close_calls == 1
    servers.stop()


def test_stop_address_delete_failure_preserves_ownership_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3: failed address delete keeps ownership; second stop retries delete."""
    del_calls = 0
    del_fail_once = True

    def _ip(command: list[str], **kwargs: object) -> MagicMock:
        nonlocal del_calls, del_fail_once
        result = MagicMock()
        result.args = command
        result.stdout = ""
        result.stderr = ""
        if command[:4] == ["ip", "addr", "del", "10.200.42.3/32"]:
            del_calls += 1
            if del_fail_once:
                del_fail_once = False
                result.returncode = 1
                result.stderr = "simulated del failure"
                return result
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
    with pytest.raises(ReferenceSubstrateEndpointCleanupError):
        servers.stop()
    assert servers.denied_addr_owned is True
    assert del_calls == 1
    servers.stop()
    assert servers.denied_addr_owned is False
    assert del_calls == 2


def test_partial_startup_rollback_address_delete_failure_surfaces_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T4: rollback delete failure is visible; later stop() retries."""
    _patch_ip_success(monkeypatch)
    del_calls = 0
    del_fail_once = True

    def _ip(command: list[str], **kwargs: object) -> MagicMock:
        nonlocal del_calls, del_fail_once
        result = MagicMock()
        result.args = command
        result.stdout = ""
        result.stderr = ""
        if command[:4] == ["ip", "addr", "del", "10.200.42.3/32"]:
            del_calls += 1
            if del_fail_once:
                del_fail_once = False
                result.returncode = 1
                result.stderr = "rollback del failure"
                return result
        result.returncode = 0
        return result

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._run_ip",
        _ip,
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
    with pytest.raises(ReferenceSubstrateEndpointError) as exc_info:
        servers.start()
    assert "rollback incomplete" in str(exc_info.value)
    assert servers.denied_addr_owned is True
    assert del_calls == 1
    servers.stop()
    assert servers.denied_addr_owned is False
    assert del_calls == 2


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
    """T7: second stop after success does not repeat address deletion."""
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


def test_destroy_session_failure_preserves_ownership_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T5: failed destroy keeps session registered for retry."""
    backend = _backend_with_stub_endpoints(monkeypatch)
    resources = _netns_resources("igx-qual-destroy-retry")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.create_netns_session",
        lambda: resources,
    )
    session = backend.create_session()
    destroy_calls = 0

    def _destroy(res: NetnsSessionResources) -> None:
        nonlocal destroy_calls
        destroy_calls += 1
        if destroy_calls == 1:
            raise ReferenceSubstratePolicyError("simulated destroy failure")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        _destroy,
    )
    with pytest.raises(ReferenceSubstratePolicyError):
        backend.destroy_session(session.session_id)
    assert destroy_calls == 1
    backend.destroy_session(session.session_id)
    assert destroy_calls == 2


def test_backend_close_drains_sessions_and_stops_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoints = ReferenceEndpointServers()
    monkeypatch.setattr(endpoints, "start", lambda: None)
    stop_mock = MagicMock()
    monkeypatch.setattr(endpoints, "stop", stop_mock)
    backend = ReferenceSandboxBackend(
        preflight=_ok_preflight(),
        endpoint_servers=endpoints,
    )
    resources = _netns_resources("igx-qual-test")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.create_netns_session",
        lambda: resources,
    )
    backend.create_session()
    destroy_mock = MagicMock()
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        destroy_mock,
    )
    backend.close()
    destroy_mock.assert_called_once_with(resources)
    stop_mock.assert_called_once()
    destroy_mock.reset_mock()
    stop_mock.reset_mock()
    backend.close()
    destroy_mock.assert_not_called()
    stop_mock.assert_not_called()


def test_backend_close_mixed_session_cleanup_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """T6: successful session removed; failed session retried on second close."""
    endpoints = ReferenceEndpointServers()
    monkeypatch.setattr(endpoints, "start", lambda: None)
    stop_mock = MagicMock()
    monkeypatch.setattr(endpoints, "stop", stop_mock)
    backend = ReferenceSandboxBackend(
        preflight=_ok_preflight(),
        endpoint_servers=endpoints,
    )
    r1 = _netns_resources("igx-qual-a")
    r2 = _netns_resources("igx-qual-b")
    created = iter((r1, r2))

    def _create() -> NetnsSessionResources:
        return next(created)

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.create_netns_session",
        _create,
    )
    s1 = backend.create_session()
    s2 = backend.create_session()
    assert s1.session_id == r1.netns
    assert s2.session_id == r2.netns
    destroy_attempts: dict[str, int] = {"igx-qual-a": 0, "igx-qual-b": 0}

    def _destroy(resources: NetnsSessionResources) -> None:
        destroy_attempts[resources.netns] += 1
        if resources.netns == "igx-qual-a" and destroy_attempts[resources.netns] == 1:
            raise ReferenceSubstratePolicyError("simulated session cleanup failure")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        _destroy,
    )
    with pytest.raises(ReferenceSubstrateLifecycleError):
        backend.close()
    stop_mock.assert_called_once()
    assert destroy_attempts["igx-qual-a"] == 1
    assert destroy_attempts["igx-qual-b"] == 1
    stop_mock.reset_mock()
    backend.close()
    assert destroy_attempts["igx-qual-a"] == 2
    assert destroy_attempts["igx-qual-b"] == 1
    stop_mock.assert_called_once()
    stop_mock.reset_mock()
    backend.close()
    stop_mock.assert_not_called()


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
    r1 = _netns_resources("igx-qual-a")
    r2 = _netns_resources("igx-qual-b")
    created = iter((r1, r2))

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.create_netns_session",
        lambda: next(created),
    )
    backend.create_session()
    backend.create_session()

    def _destroy(resources: NetnsSessionResources) -> None:
        if resources.netns == "igx-qual-a":
            raise ReferenceSubstratePolicyError("simulated session cleanup failure")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        _destroy,
    )
    with pytest.raises(ReferenceSubstrateLifecycleError):
        backend.close()
    stop_mock.assert_called_once()
