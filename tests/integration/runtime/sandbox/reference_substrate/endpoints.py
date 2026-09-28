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


class ReferenceEndpointServers:
    """Host-local HTTP services bound to distinct qualification addresses."""

    def __init__(self) -> None:
        self._allowed: ThreadingHTTPServer | None = None
        self._denied: ThreadingHTTPServer | None = None
        self._threads: list[threading.Thread] = []
        self._denied_addr_added = False

    def start(self) -> None:
        if self._allowed is not None:
            return
        subprocess.run(
            ["ip", "addr", "add", f"{DENIED_ADDR}/32", "dev", "lo"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        self._denied_addr_added = True
        allowed = ThreadingHTTPServer((ALLOWED_ADDR, ALLOWED_PORT), _AllowedHandler)
        allowed.RequestHandlerClass.denied_redirect_url = (  # type: ignore[attr-defined]
            f"http://{DENIED_HOSTNAME}:{DENIED_PORT}/"
        )
        denied = ThreadingHTTPServer((DENIED_ADDR, DENIED_PORT), _DeniedHandler)
        for server in (allowed, denied):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self._threads.append(thread)
        self._allowed = allowed
        self._denied = denied

    def stop(self) -> None:
        for server in (self._allowed, self._denied):
            if server is not None:
                server.shutdown()
                server.server_close()
        self._allowed = None
        self._denied = None
        self._threads.clear()


def default_reference_scenario() -> PhysicalEgressScenario:
    from .constants import REFERENCE_PROVIDER_ID

    return PhysicalEgressScenario(
        scenario_id="reference-physical-egress-causal-proof",
        provider=REFERENCE_PROVIDER_ID,
        allowed_host=f"http://{ALLOWED_HOSTNAME}:{ALLOWED_PORT}",
        denied_host=f"http://{DENIED_HOSTNAME}:{DENIED_PORT}",
        redirect_url=f"http://{ALLOWED_HOSTNAME}:{ALLOWED_PORT}{REDIRECT_PATH}",
    )
