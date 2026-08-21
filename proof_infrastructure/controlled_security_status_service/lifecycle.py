# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import httpx
import uvicorn

from proof_infrastructure.controlled_security_status_service.app import create_app
from proof_infrastructure.controlled_security_status_service.state import SecurityStatusStore


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True, slots=True)
class ControlledSecurityStatusServer:
    """Start and stop a real loopback HTTP Security Status service for proof tests."""

    base_url: str
    port: int
    store: SecurityStatusStore
    server: uvicorn.Server
    thread: threading.Thread

    @classmethod
    def start(cls, *, store: SecurityStatusStore | None = None) -> ControlledSecurityStatusServer:
        security_store = store or SecurityStatusStore()
        port = _free_port()
        app = create_app(store=security_store)
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run,
            daemon=True,
            name="security-status-proof",
        )
        thread.start()
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 10.0
        last_error = "startup_timeout"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{base_url}/health", timeout=1.0)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                time.sleep(0.02)
                continue
            if response.status_code == 200:
                return cls(
                    base_url=base_url,
                    port=port,
                    store=security_store,
                    server=server,
                    thread=thread,
                )
            last_error = f"status={response.status_code}"
            time.sleep(0.02)
        server.should_exit = True
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("security_status_service_startup_shutdown_timeout")
        raise RuntimeError(f"security_status_service_startup_failed: {last_error}")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise RuntimeError("security_status_service_shutdown_timeout")
