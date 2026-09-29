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


class ReferenceEndpointServers:
    """Host-local HTTP services bound to distinct qualification addresses."""

    def __init__(self) -> None:
        self._allowed: ThreadingHTTPServer | None = None
        self._denied: ThreadingHTTPServer | None = None
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
        threads: list[threading.Thread] = []
        try:
            allowed = ThreadingHTTPServer((ALLOWED_ADDR, ALLOWED_PORT), _AllowedHandler)
            allowed.RequestHandlerClass.denied_redirect_url = (  # type: ignore[attr-defined]
                f"http://{DENIED_HOSTNAME}:{DENIED_PORT}/"
            )
            denied = ThreadingHTTPServer((DENIED_ADDR, DENIED_PORT), _DeniedHandler)
            for server in (allowed, denied):
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                threads.append(thread)
        except Exception as exc:
            self._rollback_partial_startup(allowed, denied, threads)
            raise ReferenceSubstrateEndpointError(
                f"reference endpoint servers failed to start: {exc}",
            ) from exc
        self._allowed = allowed
        self._denied = denied
        self._threads = threads

    def _rollback_partial_startup(
        self,
        allowed: ThreadingHTTPServer | None,
        denied: ThreadingHTTPServer | None,
        threads: list[threading.Thread],
    ) -> None:
        for server in (allowed, denied):
            if server is not None:
                try:
                    server.shutdown()
                except OSError:
                    pass
                try:
                    server.server_close()
                except OSError:
                    pass
        threads.clear()
        if self._denied_addr_owned:
            try:
                _remove_owned_denied_loopback_address()
            except ReferenceSubstrateEndpointCleanupError:
                pass
            self._denied_addr_owned = False

    def stop(self) -> None:
        cleanup_errors: list[str] = []
        for server in (self._allowed, self._denied):
            if server is not None:
                try:
                    server.shutdown()
                except OSError as exc:
                    cleanup_errors.append(f"server shutdown: {exc}")
                try:
                    server.server_close()
                except OSError as exc:
                    cleanup_errors.append(f"server close: {exc}")
        self._allowed = None
        self._denied = None
        self._threads.clear()
        if self._denied_addr_owned:
            try:
                _remove_owned_denied_loopback_address()
            except ReferenceSubstrateEndpointCleanupError as exc:
                cleanup_errors.append(str(exc))
            self._denied_addr_owned = False
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
