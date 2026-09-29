# © Artur Czarnecki. All rights reserved.

"""Deterministic local HTTP endpoints for reference substrate qualification."""

from __future__ import annotations

import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.integration.providers.sandbox_host.qualification.scenarios import PhysicalEgressScenario

from .constants import (
    ALLOWED_ADDR,
    ALLOWED_HOSTNAME,
    ALLOWED_PORT,
    DENIED_ADDR,
    DENIED_HOSTNAME,
    DENIED_PORT,
    REDIRECT_PATH,
)
from .errors import ReferenceSubstrateEndpointCleanupError, ReferenceSubstrateEndpointError


class _AllowedHandler(BaseHTTPRequestHandler):
    denied_redirect_url: str = f"http://{DENIED_HOSTNAME}:{DENIED_PORT}/"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == REDIRECT_PATH.rstrip("/"):
            self.send_response(302)
            self.send_header("Location", self.denied_redirect_url)
            self.end_headers()
            return
        body = b"allowed-ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _DeniedHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        body = b"denied-ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _run_ip(command: list[str], *, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def loopback_has_ipv4_address(address: str) -> bool:
    completed = _run_ip(["ip", "-4", "addr", "show", "dev", "lo"])
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ReferenceSubstrateEndpointError(
            f"cannot inspect loopback addresses: {' '.join(completed.args)} ({stderr})",
        )
    needle = f"inet {address}/"
    return needle in completed.stdout


def _add_denied_loopback_address() -> None:
    completed = _run_ip(["ip", "addr", "add", f"{DENIED_ADDR}/32", "dev", "lo"])
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ReferenceSubstrateEndpointError(
            f"cannot add qualification denied address {DENIED_ADDR}/32: {stderr}",
        )


def _remove_owned_denied_loopback_address() -> None:
    completed = _run_ip(["ip", "addr", "del", f"{DENIED_ADDR}/32", "dev", "lo"])
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ReferenceSubstrateEndpointCleanupError(
            f"cannot remove harness-owned address {DENIED_ADDR}/32: {stderr}",
        )


def _cleanup_http_server(
    server: ThreadingHTTPServer,
    *,
    serve_forever_started: bool,
) -> list[str]:
    cleanup_errors: list[str] = []
    if serve_forever_started:
        try:
            server.shutdown()
        except OSError as exc:
            cleanup_errors.append(f"server shutdown: {exc}")
    try:
        server.server_close()
    except OSError as exc:
        cleanup_errors.append(f"server close: {exc}")
    return cleanup_errors


class ReferenceEndpointServers:
    """Host-local HTTP services bound to distinct qualification addresses."""

    def __init__(self) -> None:
        self._allowed: ThreadingHTTPServer | None = None
        self._denied: ThreadingHTTPServer | None = None
        self._allowed_serve_forever_started = False
        self._denied_serve_forever_started = False
        self._threads: list[threading.Thread] = []
        self._denied_addr_owned = False

    @property
    def denied_addr_owned(self) -> bool:
        return self._denied_addr_owned

    def start(self) -> None:
        if self._allowed is not None:
            return
        if loopback_has_ipv4_address(DENIED_ADDR):
            raise ReferenceSubstrateEndpointError(
                f"qualification denied address {DENIED_ADDR}/32 already present on loopback; "
                "ambient host state is not harness-owned",
            )
        _add_denied_loopback_address()
        self._denied_addr_owned = True
        allowed: ThreadingHTTPServer | None = None
        denied: ThreadingHTTPServer | None = None
        allowed_serve_forever_started = False
        denied_serve_forever_started = False
        threads: list[threading.Thread] = []
        startup_exc: BaseException | None = None
        try:
            allowed = ThreadingHTTPServer((ALLOWED_ADDR, ALLOWED_PORT), _AllowedHandler)
            allowed.RequestHandlerClass.denied_redirect_url = (  # type: ignore[attr-defined]
                f"http://{DENIED_HOSTNAME}:{DENIED_PORT}/"
            )
            denied = ThreadingHTTPServer((DENIED_ADDR, DENIED_PORT), _DeniedHandler)
            allowed_thread = threading.Thread(target=allowed.serve_forever, daemon=True)
            allowed_thread.start()
            allowed_serve_forever_started = True
            threads.append(allowed_thread)
            denied_thread = threading.Thread(target=denied.serve_forever, daemon=True)
            denied_thread.start()
            denied_serve_forever_started = True
            threads.append(denied_thread)
        except BaseException as exc:
            startup_exc = exc
        if startup_exc is not None:
            rollback_errors = self._rollback_partial_startup(
                allowed,
                denied,
                allowed_serve_forever_started=allowed_serve_forever_started,
                denied_serve_forever_started=denied_serve_forever_started,
            )
            message = f"reference endpoint servers failed to start: {startup_exc}"
            if rollback_errors:
                message += f"; rollback incomplete: {rollback_errors}"
            raise ReferenceSubstrateEndpointError(message) from startup_exc
        self._allowed = allowed
        self._denied = denied
        self._allowed_serve_forever_started = True
        self._denied_serve_forever_started = True
        self._threads = threads

    def _rollback_partial_startup(
        self,
        allowed: ThreadingHTTPServer | None,
        denied: ThreadingHTTPServer | None,
        *,
        allowed_serve_forever_started: bool,
        denied_serve_forever_started: bool,
    ) -> str:
        rollback_errors: list[str] = []
        if allowed is not None:
            rollback_errors.extend(
                _cleanup_http_server(
                    allowed,
                    serve_forever_started=allowed_serve_forever_started,
                ),
            )
        if denied is not None:
            rollback_errors.extend(
                _cleanup_http_server(
                    denied,
                    serve_forever_started=denied_serve_forever_started,
                ),
            )
        if self._denied_addr_owned:
            try:
                _remove_owned_denied_loopback_address()
                self._denied_addr_owned = False
            except ReferenceSubstrateEndpointCleanupError as exc:
                rollback_errors.append(str(exc))
        return "; ".join(rollback_errors)

    def stop(self) -> None:
        cleanup_errors: list[str] = []
        if self._allowed is not None:
            server_errors = _cleanup_http_server(
                self._allowed,
                serve_forever_started=self._allowed_serve_forever_started,
            )
            if server_errors:
                cleanup_errors.extend(server_errors)
            else:
                self._allowed = None
                self._allowed_serve_forever_started = False
        if self._denied is not None:
            server_errors = _cleanup_http_server(
                self._denied,
                serve_forever_started=self._denied_serve_forever_started,
            )
            if server_errors:
                cleanup_errors.extend(server_errors)
            else:
                self._denied = None
                self._denied_serve_forever_started = False
        if self._allowed is None and self._denied is None:
            self._threads.clear()
        if self._denied_addr_owned:
            try:
                _remove_owned_denied_loopback_address()
                self._denied_addr_owned = False
            except ReferenceSubstrateEndpointCleanupError as exc:
                cleanup_errors.append(str(exc))
        if cleanup_errors:
            raise ReferenceSubstrateEndpointCleanupError(
                "reference endpoint cleanup incomplete: " + "; ".join(cleanup_errors),
            )


def default_reference_scenario() -> PhysicalEgressScenario:
    from .constants import REFERENCE_PROVIDER_ID

    return PhysicalEgressScenario(
        scenario_id="reference-physical-egress-causal-proof",
        provider=REFERENCE_PROVIDER_ID,
        allowed_host=f"http://{ALLOWED_HOSTNAME}:{ALLOWED_PORT}",
        denied_host=f"http://{DENIED_HOSTNAME}:{DENIED_PORT}",
        redirect_url=f"http://{ALLOWED_HOSTNAME}:{ALLOWED_PORT}{REDIRECT_PATH}",
    )
