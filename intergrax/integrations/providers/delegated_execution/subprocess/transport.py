# © Artur Czarnecki. All rights reserved.

"""TCP transport to subprocess delegated execution worker."""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from intergrax.contracts.delegated_execution_provider import DelegatedExecutionTransportError


class DelegatedExecutionPostDispatchTransportError(DelegatedExecutionTransportError):
    """Transport failed after the worker accepted the request frame."""
from intergrax.integrations.providers.delegated_execution.subprocess.config import (
    SubprocessDelegatedExecutionProviderConfig,
)
from intergrax.integrations.providers.delegated_execution.subprocess.protocol import (
    WorkerRequest,
    WorkerResponse,
)

_MAX_FRAME_BYTES: Final = 1_048_576
_AUTH_ENV: Final = "SUBPROCESS_DELEGATED_AUTH"
_MAX_CONCURRENT_ENV: Final = "SUBPROCESS_DELEGATED_MAX_CONCURRENT"


@runtime_checkable
class SubprocessDelegatedExecutionTransport(Protocol):
    """Transport abstraction — platform service never opens sockets directly."""

    @property
    def host(self) -> str:
        ...

    @property
    def port(self) -> int:
        ...

    def roundtrip(self, request: WorkerRequest, *, timeout_seconds: float) -> WorkerResponse:
        ...

    def close(self) -> None:
        ...


@dataclass
class TcpSubprocessDelegatedExecutionTransport(SubprocessDelegatedExecutionTransport):
    """Real subprocess + TCP boundary transport."""

    _host: str
    _port: int
    _process: subprocess.Popen[str] | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @classmethod
    def spawn(cls, config: SubprocessDelegatedExecutionProviderConfig) -> TcpSubprocessDelegatedExecutionTransport:
        env = os.environ.copy()
        if config.connection_auth_token:
            env[_AUTH_ENV] = config.connection_auth_token
        env[_MAX_CONCURRENT_ENV] = str(config.max_concurrent_requests)
        cmd = [
            sys.executable,
            "-m",
            "intergrax.integrations.providers.delegated_execution.subprocess.worker_main",
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if process.stdout is None:
            raise DelegatedExecutionTransportError("worker stdout unavailable")
        port_line = process.stdout.readline()
        if not port_line.startswith("PORT="):
            process.kill()
            raise DelegatedExecutionTransportError("worker failed to publish port")
        port = int(port_line.strip().removeprefix("PORT="))
        return cls(_host="127.0.0.1", _port=port, _process=process)

    def roundtrip(self, request: WorkerRequest, *, timeout_seconds: float) -> WorkerResponse:
        payload = json.dumps(request.model_dump(mode="json"), separators=(",", ":")).encode("utf-8")
        if len(payload) > _MAX_FRAME_BYTES:
            raise DelegatedExecutionTransportError("request frame too large")
        dispatched = False
        try:
            with socket.create_connection(
                (self._host, self._port),
                timeout=timeout_seconds,
            ) as sock:
                sock.sendall(struct.pack(">I", len(payload)) + payload)
                dispatched = True
                header = _recv_exact(sock, 4, timeout_seconds, post_dispatch=True)
                (length,) = struct.unpack(">I", header)
                if length <= 0 or length > _MAX_FRAME_BYTES:
                    raise DelegatedExecutionPostDispatchTransportError("invalid response frame")
                body = _recv_exact(sock, length, timeout_seconds, post_dispatch=True)
        except TimeoutError as exc:
            if dispatched:
                raise DelegatedExecutionPostDispatchTransportError(
                    "transport timed out after dispatch",
                ) from exc
            raise DelegatedExecutionTransportError("transport timed out") from exc
        except OSError as exc:
            if dispatched:
                raise DelegatedExecutionPostDispatchTransportError(
                    "transport I/O failed after dispatch",
                ) from exc
            raise DelegatedExecutionTransportError("transport I/O failed") from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
            return WorkerResponse.model_validate(decoded)
        except (json.JSONDecodeError, ValueError) as exc:
            raise DelegatedExecutionPostDispatchTransportError("invalid worker response") from exc

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None


@dataclass
class FailingConnectTransport(SubprocessDelegatedExecutionTransport):
    """Test transport that fails before dispatch."""

    _host: str = "127.0.0.1"
    _port: int = 0

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def roundtrip(self, request: WorkerRequest, *, timeout_seconds: float) -> WorkerResponse:
        raise DelegatedExecutionTransportError("connect refused")

    def close(self) -> None:
        return


def _recv_exact(
    sock: socket.socket,
    size: int,
    timeout_seconds: float,
    *,
    post_dispatch: bool = False,
) -> bytes:
    sock.settimeout(timeout_seconds)
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            if post_dispatch:
                raise DelegatedExecutionPostDispatchTransportError("truncated transport read")
            raise DelegatedExecutionTransportError("truncated transport read")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
