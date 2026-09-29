# © Artur Czarnecki. All rights reserved.

"""Lifecycle and cleanup integrity for reference substrate qualification harness."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from intergrax.runtime.sandbox.contracts import SandboxSecurityRequirements

from tests.integration.runtime.sandbox.reference_substrate.backend import (
    ReferenceSandboxBackend,
    ReferenceSubstrateSecurityError,
)
from tests.integration.runtime.sandbox.reference_substrate.constants import (
    ALLOWED_ADDR,
    ALLOWED_LISTEN_BIND,
)
from tests.integration.runtime.sandbox.reference_substrate.endpoints import ReferenceEndpointServers
from tests.integration.runtime.sandbox.reference_substrate.errors import (
    ReferenceSubstrateEndpointCleanupError,
    ReferenceSubstrateEndpointError,
    ReferenceSubstrateLifecycleError,
    ReferenceSubstrateSecuritySetupLifecycleError,
)
from tests.integration.runtime.sandbox.reference_substrate.firewall import ReferenceSubstratePolicyError
from tests.integration.runtime.sandbox.reference_substrate.preflight import ReferenceSubstratePreflight
from tests.integration.runtime.sandbox.reference_substrate.topology import (
    NetnsSessionResources,
    ReferenceSubstrateTopologyError,
    _PartialTopologyCreation,
    _rollback_allocated_topology,
    destroy_netns_session,
)


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


class _LoopbackPresenceTracker:
    def __init__(self, *, present: bool = True) -> None:
        self.present = present


def _patch_loopback_presence_tracker(
    monkeypatch: pytest.MonkeyPatch,
    tracker: _LoopbackPresenceTracker,
) -> None:
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._loopback_address_presence",
        lambda _addr: tracker.present,
    )


def _patch_ip_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.loopback_has_ipv4_address",
        lambda _addr: False,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._loopback_address_presence",
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


def test_address_add_failure_retains_ownership_for_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.loopback_has_ipv4_address",
        lambda _addr: False,
    )

    def _fail_add() -> None:
        raise ReferenceSubstrateEndpointError("simulated add failure")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._add_denied_loopback_address",
        _fail_add,
    )
    remove_calls = 0

    def _remove() -> None:
        nonlocal remove_calls
        remove_calls += 1

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._remove_owned_denied_loopback_address",
        _remove,
    )
    servers = ReferenceEndpointServers()
    with pytest.raises(ReferenceSubstrateEndpointError):
        servers.start()
    assert remove_calls == 1
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
    presence = _LoopbackPresenceTracker(present=True)
    _patch_loopback_presence_tracker(monkeypatch, presence)

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
        if command[:4] == ["ip", "addr", "del", "10.200.42.3/32"]:
            presence.present = False
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
    presence = _LoopbackPresenceTracker(present=True)
    _patch_loopback_presence_tracker(monkeypatch, presence)

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
            presence.present = False
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
    presence = _LoopbackPresenceTracker(present=True)
    _patch_loopback_presence_tracker(monkeypatch, presence)

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
            presence.present = False
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
    presence = _LoopbackPresenceTracker(present=True)
    _patch_loopback_presence_tracker(monkeypatch, presence)

    def _ip(command: list[str], **kwargs: object) -> MagicMock:
        nonlocal del_calls
        result = MagicMock()
        result.args = command
        result.stdout = ""
        result.stderr = ""
        if command[:4] == ["ip", "addr", "del", "10.200.42.3/32"]:
            del_calls += 1
            presence.present = False
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
    presence = _LoopbackPresenceTracker(present=True)
    _patch_loopback_presence_tracker(monkeypatch, presence)

    def _ip(command: list[str], **kwargs: object) -> MagicMock:
        nonlocal del_calls
        result = MagicMock()
        result.args = command
        result.stdout = ""
        result.stderr = ""
        if command[:4] == ["ip", "addr", "del", "10.200.42.3/32"]:
            del_calls += 1
            presence.present = False
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


def _cloud_allowlist_requirements() -> SandboxSecurityRequirements:
    from tests.integration.runtime.sandbox.reference_substrate.endpoints import default_reference_scenario

    scenario = default_reference_scenario()
    return SandboxSecurityRequirements(
        isolation_tier="cloud",
        network_egress="allowlist",
        network_egress_allowlist=scenario.allowlist,
    )


def test_backend_constructor_start_retained_ownership_invokes_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1: failed endpoint start with retained ownership triggers constructor stop()."""
    endpoints = ReferenceEndpointServers()
    stop_calls = 0

    def _start_fail() -> None:
        monkeypatch.setattr(endpoints, "_allowed", _FakeHttpServer())
        raise ReferenceSubstrateEndpointError("simulated startup with retained allowed server")

    def _stop() -> None:
        nonlocal stop_calls
        stop_calls += 1
        endpoints._allowed = None

    monkeypatch.setattr(endpoints, "start", _start_fail)
    monkeypatch.setattr(endpoints, "stop", _stop)
    with pytest.raises(ReferenceSubstrateEndpointError):
        ReferenceSandboxBackend(preflight=_ok_preflight(), endpoint_servers=endpoints)
    assert stop_calls == 1


def test_backend_constructor_start_fail_stop_succeeds_no_retained_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2: constructor cleanup succeeds; primary startup error preserved."""

    def _start_fail() -> None:
        raise ReferenceSubstrateEndpointError("simulated startup failure")

    endpoints = ReferenceEndpointServers()
    monkeypatch.setattr(endpoints, "start", _start_fail)
    stop_calls = 0

    def _stop() -> None:
        nonlocal stop_calls
        stop_calls += 1

    monkeypatch.setattr(endpoints, "stop", _stop)
    with pytest.raises(ReferenceSubstrateEndpointError, match="simulated startup failure"):
        ReferenceSandboxBackend(preflight=_ok_preflight(), endpoint_servers=endpoints)
    assert stop_calls == 1


def test_backend_constructor_start_and_stop_fail_aggregate_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3: constructor raises lifecycle error with startup and cleanup failures."""
    endpoints = ReferenceEndpointServers()
    monkeypatch.setattr(
        endpoints,
        "start",
        lambda: (_ for _ in ()).throw(ReferenceSubstrateEndpointError("simulated startup failure")),
    )
    monkeypatch.setattr(
        endpoints,
        "stop",
        lambda: (_ for _ in ()).throw(
            ReferenceSubstrateEndpointCleanupError("simulated constructor cleanup failure"),
        ),
    )
    with pytest.raises(ReferenceSubstrateLifecycleError) as exc_info:
        ReferenceSandboxBackend(preflight=_ok_preflight(), endpoint_servers=endpoints)
    message = str(exc_info.value)
    assert "simulated startup failure" in message
    assert "simulated constructor cleanup failure" in message


def test_backend_constructor_success_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """C4: successful construction still starts endpoints."""
    endpoints = ReferenceEndpointServers()
    start_calls = 0

    def _start() -> None:
        nonlocal start_calls
        start_calls += 1

    monkeypatch.setattr(endpoints, "start", _start)
    monkeypatch.setattr(endpoints, "stop", lambda: None)
    backend = ReferenceSandboxBackend(preflight=_ok_preflight(), endpoint_servers=endpoints)
    assert start_calls == 1
    backend.close()


def test_endpoint_startup_does_not_require_preexisting_allowed_addr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean-host: listener bind does not require ALLOWED_ADDR on host before sessions."""
    _patch_ip_success(monkeypatch)
    bind_addresses: list[str] = []

    def _tracking_httpserver(
        server_address: tuple[str, int],
        request_handler_class: type,
    ) -> _FakeHttpServer:
        bind_addresses.append(server_address[0])
        return _fake_httpserver_factory(server_address, request_handler_class)

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.ThreadingHTTPServer",
        _tracking_httpserver,
    )
    servers = ReferenceEndpointServers()
    servers.start()
    assert ALLOWED_LISTEN_BIND in bind_addresses
    assert ALLOWED_ADDR not in bind_addresses
    servers.stop()


def test_topology_netns_created_veth_add_fails_rollback_netns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1: veth creation failure rolls back netns."""
    from tests.integration.runtime.sandbox.reference_substrate.topology import create_netns_session

    delete_calls: list[list[str]] = []

    def _run(command: list[str], *, timeout: float = 10.0) -> None:
        if command[:3] == ["ip", "link", "add"]:
            raise ReferenceSubstrateTopologyError("simulated veth add failure")
        if command[:3] == ["ip", "netns", "add"]:
            return
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._run",
        _run,
    )

    def _try_run(command: list[str], *, timeout: float = 10.0) -> str | None:
        delete_calls.append(command)
        return None

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._try_run",
        _try_run,
    )
    with pytest.raises(ReferenceSubstrateTopologyError, match="simulated veth add failure"):
        create_netns_session()
    assert any(cmd[:3] == ["ip", "netns", "delete"] for cmd in delete_calls)


def test_topology_later_setup_failure_rolls_back_netns_and_veth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2: address setup failure rolls back veth and netns."""
    from tests.integration.runtime.sandbox.reference_substrate.topology import create_netns_session

    delete_calls: list[list[str]] = []

    def _run(command: list[str], *, timeout: float = 10.0) -> None:
        if len(command) >= 4 and command[:3] == ["ip", "addr", "add"] and command[3] == f"{ALLOWED_ADDR}/24":
            raise ReferenceSubstrateTopologyError("simulated host addr failure")
        return None

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._run",
        _run,
    )

    def _try_run(command: list[str], *, timeout: float = 10.0) -> str | None:
        delete_calls.append(command)
        return None

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._try_run",
        _try_run,
    )
    with pytest.raises(ReferenceSubstrateTopologyError, match="simulated host addr failure"):
        create_netns_session()
    assert any(cmd[:3] == ["ip", "netns", "delete"] for cmd in delete_calls)
    assert any(cmd[:3] == ["ip", "link", "delete"] for cmd in delete_calls)


def test_topology_hosts_write_failure_rolls_back_network_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3: /etc/netns hosts failure rolls back network resources."""
    from tests.integration.runtime.sandbox.reference_substrate.topology import create_netns_session

    delete_calls: list[list[str]] = []

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._run",
        lambda command, timeout=10.0: None,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._write_netns_hosts",
        lambda _netns: (_ for _ in ()).throw(ReferenceSubstrateTopologyError("simulated hosts write failure")),
    )

    def _try_run(command: list[str], *, timeout: float = 10.0) -> str | None:
        delete_calls.append(command)
        return None

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._try_run",
        _try_run,
    )
    with pytest.raises(ReferenceSubstrateTopologyError, match="simulated hosts write failure"):
        create_netns_session()
    assert any(cmd[:3] == ["ip", "netns", "delete"] for cmd in delete_calls)


def test_topology_rollback_partial_failure_surfaces_both_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T4: rollback incomplete remains visible alongside setup error."""
    partial = _PartialTopologyCreation(
        netns="igx-qual-rollback",
        veth_host="veth000h",
        veth_peer="veth000p",
        mutation_started=True,
    )

    def _try_run(command: list[str], *, timeout: float = 10.0) -> str | None:
        if command[:3] == ["ip", "netns", "delete"]:
            return "simulated netns delete failure"
        return None

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._try_run",
        _try_run,
    )
    rollback_errors = _rollback_allocated_topology(partial)
    assert "simulated netns delete failure" in "; ".join(rollback_errors)


def test_topology_successful_creation_does_not_invoke_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T5: successful path does not call rollback helper."""
    from tests.integration.runtime.sandbox.reference_substrate.topology import create_netns_session

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._run",
        lambda command, timeout=10.0: None,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._write_netns_hosts",
        lambda _netns: None,
    )
    rollback_calls = 0

    def _rollback(partial: _PartialTopologyCreation) -> list[str]:
        nonlocal rollback_calls
        rollback_calls += 1
        return []

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._rollback_allocated_topology",
        _rollback,
    )
    resources = create_netns_session()
    assert resources.netns.startswith("igx-qual-")
    assert rollback_calls == 0


def test_security_setup_policy_fail_cleanup_success_drops_provisional_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1: policy failure with successful topology cleanup removes provisional ownership."""
    backend = _backend_with_stub_endpoints(monkeypatch)
    resources = _netns_resources("igx-qual-sec-s1")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.create_netns_session",
        lambda: resources,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.apply_egress_policy_netns",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ReferenceSubstratePolicyError("simulated policy apply failure"),
        ),
    )
    destroy_calls = 0

    def _destroy(res: NetnsSessionResources) -> None:
        nonlocal destroy_calls
        destroy_calls += 1

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        _destroy,
    )
    with pytest.raises(ReferenceSubstrateSecurityError, match="simulated policy apply failure"):
        backend.create_session_with_security(_cloud_allowlist_requirements())
    assert destroy_calls == 1
    assert resources.netns not in backend._sessions  # noqa: SLF001


def test_security_setup_policy_fail_cleanup_fail_retains_session_for_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2: policy failure with cleanup failure keeps ownership for backend.close()."""
    backend = _backend_with_stub_endpoints(monkeypatch)
    resources = _netns_resources("igx-qual-sec-s2")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.create_netns_session",
        lambda: resources,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.apply_egress_policy_netns",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ReferenceSubstratePolicyError("simulated policy apply failure"),
        ),
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        lambda res: (_ for _ in ()).throw(
            ReferenceSubstratePolicyError("simulated topology cleanup failure"),
        ),
    )
    with pytest.raises(ReferenceSubstrateSecuritySetupLifecycleError) as exc_info:
        backend.create_session_with_security(_cloud_allowlist_requirements())
    message = str(exc_info.value)
    assert "simulated policy apply failure" in message
    assert "simulated topology cleanup failure" in message
    assert resources.netns in backend._sessions  # noqa: SLF001
    destroy_calls = 0

    def _destroy_on_close(res: NetnsSessionResources) -> None:
        nonlocal destroy_calls
        destroy_calls += 1

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        _destroy_on_close,
    )
    backend.close()
    assert destroy_calls == 1


def test_security_setup_verification_fail_cleanup_fail_retains_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: verification failure with cleanup failure retains ownership."""
    backend = _backend_with_stub_endpoints(monkeypatch)
    resources = _netns_resources("igx-qual-sec-s3")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.create_netns_session",
        lambda: resources,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.apply_egress_policy_netns",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.read_verified_egress_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ReferenceSubstratePolicyError("simulated verification failure"),
        ),
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        lambda res: (_ for _ in ()).throw(
            ReferenceSubstratePolicyError("simulated topology cleanup failure"),
        ),
    )
    with pytest.raises(ReferenceSubstrateSecuritySetupLifecycleError):
        backend.create_session_with_security(_cloud_allowlist_requirements())
    assert resources.netns in backend._sessions  # noqa: SLF001


def test_security_setup_success_updates_same_session_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4: admitted session updates the same ownership record."""
    from tests.integration.runtime.sandbox.reference_substrate.endpoints import default_reference_scenario

    backend = _backend_with_stub_endpoints(monkeypatch)
    resources = _netns_resources("igx-qual-sec-s4")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.create_netns_session",
        lambda: resources,
    )
    enforced = default_reference_scenario().allowlist

    class _Verified:
        enforced_hosts = enforced

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.apply_egress_policy_netns",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.read_verified_egress_policy",
        lambda *args, **kwargs: _Verified(),
    )
    session = backend.create_session_with_security(_cloud_allowlist_requirements())
    assert session.session_id == resources.netns
    state = backend._sessions[resources.netns]  # noqa: SLF001
    assert state.security is not None
    assert len(backend._sessions) == 1  # noqa: SLF001


def test_topology_netns_add_timeout_attempts_conservative_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T1: netns add timeout still rolls back allocated identifiers."""
    import subprocess

    from tests.integration.runtime.sandbox.reference_substrate.topology import create_netns_session

    delete_calls: list[list[str]] = []

    def _subprocess_run(command: list[str], **kwargs: object) -> MagicMock:
        if command[:3] == ["ip", "netns", "add"]:
            raise subprocess.TimeoutExpired(cmd=command, timeout=10.0)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology.subprocess.run",
        _subprocess_run,
    )

    def _try_run(command: list[str], *, timeout: float = 10.0) -> str | None:
        delete_calls.append(command)
        return None

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._try_run",
        _try_run,
    )
    with pytest.raises(ReferenceSubstrateTopologyError, match="classification=timeout"):
        create_netns_session()
    assert any(cmd[:3] == ["ip", "netns", "delete"] for cmd in delete_calls)
    assert any(cmd[:3] == ["ip", "link", "delete"] for cmd in delete_calls)


def test_topology_netns_add_os_error_attempts_conservative_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T2: netns add OSError still rolls back allocated identifiers."""
    from tests.integration.runtime.sandbox.reference_substrate.topology import create_netns_session

    delete_calls: list[list[str]] = []

    def _subprocess_run(command: list[str], **kwargs: object) -> MagicMock:
        if command[:3] == ["ip", "netns", "add"]:
            raise OSError("simulated netns add os error")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology.subprocess.run",
        _subprocess_run,
    )

    def _try_run(command: list[str], *, timeout: float = 10.0) -> str | None:
        delete_calls.append(command)
        return None

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._try_run",
        _try_run,
    )
    with pytest.raises(ReferenceSubstrateTopologyError, match="classification=os_error"):
        create_netns_session()
    assert delete_calls


def test_topology_hosts_write_os_error_attempts_network_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T3: hosts filesystem failure still attempts network rollback."""
    from tests.integration.runtime.sandbox.reference_substrate.topology import create_netns_session

    delete_calls: list[list[str]] = []

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._run",
        lambda command, timeout=10.0: None,
    )

    def _write(_netns: str) -> None:
        raise ReferenceSubstrateTopologyError(
            "cannot materialize /etc/netns/x/hosts (classification=filesystem_os_error; simulated)",
        )

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._write_netns_hosts",
        _write,
    )

    def _try_run(command: list[str], *, timeout: float = 10.0) -> str | None:
        delete_calls.append(command)
        return None

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._try_run",
        _try_run,
    )
    with pytest.raises(ReferenceSubstrateTopologyError, match="filesystem_os_error"):
        create_netns_session()
    assert any(cmd[:3] == ["ip", "netns", "delete"] for cmd in delete_calls)


def test_destroy_cleanup_delete_timeout_succeeds_when_verified_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T4: delete timeout is not failure when resources are absent."""
    resources = _netns_resources("igx-qual-absent")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._try_run",
        lambda command, timeout=10.0: f"timeout expired: {' '.join(command)}",
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._inspect_netns",
        lambda _netns: __import__(
            "tests.integration.runtime.sandbox.reference_substrate.topology",
            fromlist=["_ResidualPresence"],
        )._ResidualPresence.ABSENT,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._inspect_link",
        lambda _link: __import__(
            "tests.integration.runtime.sandbox.reference_substrate.topology",
            fromlist=["_ResidualPresence"],
        )._ResidualPresence.ABSENT,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._inspect_netns_hosts_material",
        lambda _netns: __import__(
            "tests.integration.runtime.sandbox.reference_substrate.topology",
            fromlist=["_ResidualPresence"],
        )._ResidualPresence.ABSENT,
    )
    destroy_netns_session(resources)


def test_destroy_cleanup_fails_when_resource_still_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T5: residual presence retains cleanup failure."""
    from tests.integration.runtime.sandbox.reference_substrate.topology import _ResidualPresence

    resources = _netns_resources("igx-qual-present")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._try_run",
        lambda command, timeout=10.0: None,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._inspect_netns",
        lambda _netns: _ResidualPresence.PRESENT,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._inspect_link",
        lambda _link: _ResidualPresence.ABSENT,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._inspect_netns_hosts_material",
        lambda _netns: _ResidualPresence.ABSENT,
    )
    with pytest.raises(ReferenceSubstratePolicyError, match="still present"):
        destroy_netns_session(resources)


def test_destroy_cleanup_fails_closed_when_residual_inspection_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T6: cannot prove absence → cleanup incomplete."""
    from tests.integration.runtime.sandbox.reference_substrate.topology import _ResidualPresence

    resources = _netns_resources("igx-qual-unknown")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._try_run",
        lambda command, timeout=10.0: None,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._inspect_netns",
        lambda _netns: _ResidualPresence.UNKNOWN,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._inspect_link",
        lambda _link: _ResidualPresence.ABSENT,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.topology._inspect_netns_hosts_material",
        lambda _netns: _ResidualPresence.ABSENT,
    )
    with pytest.raises(ReferenceSubstratePolicyError, match="cannot verify"):
        destroy_netns_session(resources)


def test_endpoint_add_timeout_with_address_present_invokes_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T7: ambiguous add timeout still owns and cleans address."""
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints.loopback_has_ipv4_address",
        lambda _addr: False,
    )
    remove_calls = 0

    def _remove() -> None:
        nonlocal remove_calls
        remove_calls += 1

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._remove_owned_denied_loopback_address",
        _remove,
    )

    def _add() -> None:
        raise ReferenceSubstrateEndpointError(
            "ip command timed out: ip addr add (classification=timeout)",
        )

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._add_denied_loopback_address",
        _add,
    )
    servers = ReferenceEndpointServers()
    with pytest.raises(ReferenceSubstrateEndpointError, match="classification=timeout"):
        servers.start()
    assert remove_calls == 1
    assert servers.denied_addr_owned is False


def test_endpoint_remove_timeout_succeeds_when_address_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T8: delete timeout with verified absence succeeds."""
    from tests.integration.runtime.sandbox.reference_substrate.endpoints import (
        _remove_owned_denied_loopback_address,
    )

    def _run_ip(command: list[str], **kwargs: object) -> MagicMock:
        raise ReferenceSubstrateEndpointError(
            "ip command timed out: ip addr del (classification=timeout)",
        )

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._run_ip",
        _run_ip,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._loopback_address_presence",
        lambda _addr: False,
    )
    _remove_owned_denied_loopback_address()


def test_endpoint_remove_error_retains_when_address_still_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T9: delete error with address present → cleanup failure."""
    from tests.integration.runtime.sandbox.reference_substrate.endpoints import (
        _remove_owned_denied_loopback_address,
    )

    def _run_ip(command: list[str], **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.returncode = 1
        result.stderr = "simulated delete failure"
        result.stdout = ""
        return result

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._run_ip",
        _run_ip,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.endpoints._loopback_address_presence",
        lambda _addr: True,
    )
    with pytest.raises(ReferenceSubstrateEndpointCleanupError, match="cannot remove"):
        _remove_owned_denied_loopback_address()


def test_firewall_command_timeout_becomes_policy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T10: firewall timeout normalizes to policy error."""
    import subprocess

    from tests.integration.runtime.sandbox.reference_substrate.firewall import _run_in_netns

    def _run(command: list[str], **kwargs: object) -> MagicMock:
        raise subprocess.TimeoutExpired(cmd=command, timeout=10.0)

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.firewall.subprocess.run",
        _run,
    )
    with pytest.raises(ReferenceSubstratePolicyError, match="classification=timeout"):
        _run_in_netns("igx-qual-fw", ["nft", "list", "ruleset"])


def test_firewall_command_os_error_becomes_policy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T11: firewall OSError normalizes to policy error."""
    from tests.integration.runtime.sandbox.reference_substrate.firewall import _run_in_netns

    def _run(command: list[str], **kwargs: object) -> MagicMock:
        raise OSError("simulated firewall os error")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.firewall.subprocess.run",
        _run,
    )
    with pytest.raises(ReferenceSubstratePolicyError, match="classification=os_error"):
        _run_in_netns("igx-qual-fw", ["nft", "list", "ruleset"])


def test_malformed_kernel_evidence_fails_closed() -> None:
    """Unknown-outcome T12: malformed attestation cannot pass verification."""
    from tests.integration.runtime.sandbox.reference_substrate.firewall import (
        read_verified_egress_policy,
    )

    def _bad_nft(_netns: str, _command: list[str], *, timeout: float = 10.0) -> str:
        return 'ip daddr 10.200.42.1 tcp dport not-a-port accept'

    import tests.integration.runtime.sandbox.reference_substrate.firewall as firewall_mod

    original = firewall_mod._run_in_netns
    firewall_mod._run_in_netns = _bad_nft  # type: ignore[method-assign]
    try:
        with pytest.raises(ReferenceSubstratePolicyError, match="malformed"):
            read_verified_egress_policy("igx-qual-parse", use_nftables=True)
    finally:
        firewall_mod._run_in_netns = original  # type: ignore[method-assign]


def test_security_setup_firewall_timeout_triggers_topology_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T10 integration: policy timeout enters destroy path."""
    backend = _backend_with_stub_endpoints(monkeypatch)
    resources = _netns_resources("igx-qual-sec-timeout")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.create_netns_session",
        lambda: resources,
    )

    def _apply(*args: object, **kwargs: object) -> None:
        raise ReferenceSubstratePolicyError(
            "command timed out in netns igx-qual-sec-timeout: nft add table "
            "(classification=timeout)",
        )

    destroy_calls = 0

    def _destroy(res: NetnsSessionResources) -> None:
        nonlocal destroy_calls
        destroy_calls += 1

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.apply_egress_policy_netns",
        _apply,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.destroy_netns_session",
        _destroy,
    )
    with pytest.raises(ReferenceSubstrateSecurityError, match="classification=timeout"):
        backend.create_session_with_security(_cloud_allowlist_requirements())
    assert destroy_calls == 1
    assert resources.netns not in backend._sessions  # noqa: SLF001


def test_preflight_netns_inspection_timeout_blocks_without_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T14: preflight classifies timeout as explicit block."""
    import subprocess

    from tests.integration.runtime.sandbox.reference_substrate.preflight import (
        evaluate_reference_substrate_preflight,
    )

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.preflight.shutil.which",
        lambda name: "ip" if name in {"ip", "nft", "python3"} else None,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.preflight.os.geteuid",
        lambda: 0,
        raising=False,
    )

    def _run_ip(args: list[str], *, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["ip", *args], timeout=timeout)

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.preflight._run_ip",
        _run_ip,
    )
    result = evaluate_reference_substrate_preflight()
    assert result.ok is False
    assert result.block_reason is not None
    assert "timed out" in result.block_reason


def test_preflight_netns_inspection_os_error_blocks_without_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-outcome T14: preflight classifies OSError as explicit block."""
    from tests.integration.runtime.sandbox.reference_substrate.preflight import (
        evaluate_reference_substrate_preflight,
    )

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.preflight.shutil.which",
        lambda name: "ip" if name in {"ip", "nft", "python3"} else None,
    )
    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.preflight.os.geteuid",
        lambda: 0,
        raising=False,
    )

    def _run_ip(args: list[str], *, timeout: float = 5.0) -> None:
        raise OSError("simulated preflight os error")

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.preflight._run_ip",
        _run_ip,
    )
    result = evaluate_reference_substrate_preflight()
    assert result.ok is False
    assert result.block_reason is not None
    assert "OS error" in result.block_reason


def test_probe_python_code_disables_ambient_http_proxies() -> None:
    """Unknown-outcome T15: qualification probe must not use ambient proxy env."""
    from tests.integration.providers.sandbox_host.qualification.probes import _probe_python_code

    code = _probe_python_code("http://allowed.test:18080/")
    assert "ProxyHandler({})" in code
    assert "build_opener" in code
