# © Artur Czarnecki. All rights reserved.

"""In-process worker server used by the subprocess worker entrypoint."""

from __future__ import annotations

import json
import socket
import struct
import threading
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from socketserver import StreamRequestHandler, ThreadingTCPServer
from typing import Final

from intergrax.integrations.providers.delegated_execution.subprocess.protocol import (
    WorkerExecutePayload,
    WorkerExecuteResult,
    WorkerRequest,
    WorkerResponse,
    WorkerResponseStatus,
    WorkerRpcMethod,
)

_MAX_FRAME_BYTES: Final = 1_048_576
_AUTH_ENV = "SUBPROCESS_DELEGATED_AUTH"


class _PhysicalStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class _OperationRecord:
    provider_request_id: str
    provider_operation_id: str
    invocation_id: str
    execution_id: str
    physical_status: _PhysicalStatus
    result_value: str | None = None
    status_spoof_field: str | None = None


@dataclass
class WorkerState:
    auth_token: str
    max_concurrent: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    operations: dict[str, _OperationRecord] = field(default_factory=dict)
    execute_count: int = 0
    active_requests: int = 0


def _read_frame(stream: socket.socket) -> bytes:
    header = stream.recv(4)
    if len(header) < 4:
        raise ConnectionError("truncated frame header")
    (length,) = struct.unpack(">I", header)
    if length <= 0 or length > _MAX_FRAME_BYTES:
        raise ValueError("invalid frame length")
    chunks: list[bytes] = []
    remaining = length
    while remaining > 0:
        chunk = stream.recv(remaining)
        if not chunk:
            raise ConnectionError("truncated frame body")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_frame(stream: socket.socket, payload: bytes) -> None:
    stream.sendall(struct.pack(">I", len(payload)) + payload)


def _error_response(request_id: str, *, code: str, message: str) -> WorkerResponse:
    return WorkerResponse(
        request_id=request_id,
        status=WorkerResponseStatus.ERROR,
        error_code=code,
        error_message=message,
    )


def handle_worker_request(state: WorkerState, raw: bytes) -> WorkerResponse | None:
    """Handle one worker RPC. Returns None to signal intentional disconnect (ambiguous path)."""
    data = json.loads(raw.decode("utf-8"))
    request = WorkerRequest.model_validate(data)
    if state.auth_token and request.auth_token != state.auth_token:
        return _error_response(request.request_id, code="AUTH_REJECTED", message="auth rejected")

    if request.method is WorkerRpcMethod.METRICS:
        with state.lock:
            return WorkerResponse(
                request_id=request.request_id,
                status=WorkerResponseStatus.OK,
                execute_count=state.execute_count,
            )

    if request.method is WorkerRpcMethod.STATUS:
        op_id = request.provider_operation_id
        if op_id is None:
            return _error_response(
                request.request_id,
                code="INVALID_REQUEST",
                message="provider_operation_id required",
            )
        with state.lock:
            record = state.operations.get(op_id)
        if record is None:
            return _error_response(
                request.request_id,
                code="NOT_FOUND",
                message="operation not found",
            )
        spoof = record.status_spoof_field
        req_id = record.provider_request_id
        op_id_out = record.provider_operation_id
        if spoof == "provider_request_id":
            req_id = "spoof-req"
        elif spoof == "provider_operation_id":
            op_id_out = "spoof-op"
        return WorkerResponse(
            request_id=request.request_id,
            status=WorkerResponseStatus.OK,
            provider_request_id=req_id,
            provider_operation_id=op_id_out,
            physical_status=record.physical_status.value,
        )

    if request.method is WorkerRpcMethod.CANCEL:
        op_id = request.provider_operation_id
        if op_id is None:
            return _error_response(
                request.request_id,
                code="INVALID_REQUEST",
                message="provider_operation_id required",
            )
        with state.lock:
            record = state.operations.get(op_id)
            if record is None:
                return _error_response(
                    request.request_id,
                    code="NOT_FOUND",
                    message="operation not found",
                )
            record.physical_status = _PhysicalStatus.CANCELLED
        return WorkerResponse(
            request_id=request.request_id,
            status=WorkerResponseStatus.OK,
            provider_request_id=record.provider_request_id,
            provider_operation_id=record.provider_operation_id,
        )

    if request.method is WorkerRpcMethod.REATTACH:
        op_id = request.provider_operation_id
        if op_id is None:
            return _error_response(
                request.request_id,
                code="INVALID_REQUEST",
                message="provider_operation_id required",
            )
        with state.lock:
            record = state.operations.get(op_id)
        if record is None:
            return _error_response(
                request.request_id,
                code="NOT_FOUND",
                message="operation not found",
            )
        kind = "reattached"
        if record.physical_status is _PhysicalStatus.SUCCEEDED:
            kind = "already_attached"
        return WorkerResponse(
            request_id=request.request_id,
            status=WorkerResponseStatus.OK,
            provider_request_id=record.provider_request_id,
            provider_operation_id=record.provider_operation_id,
            physical_status=record.physical_status.value,
            reattachment_kind=kind,
        )

    if request.method is not WorkerRpcMethod.EXECUTE:
        return _error_response(
            request.request_id,
            code="UNSUPPORTED",
            message="unsupported method",
        )

    if request.payload is None or request.execution_id is None:
        return _error_response(
            request.request_id,
            code="INVALID_REQUEST",
            message="execute requires payload and execution_id",
        )

    behavior = request.payload.behavior
    if behavior == "accept_then_disconnect":
        with state.lock:
            state.execute_count += 1
        return None

    if behavior == "error_auth_rejected":
        return _error_response(
            request.request_id,
            code="AUTH_REJECTED",
            message="auth rejected",
        )
    if behavior == "error_invalid_request":
        return _error_response(
            request.request_id,
            code="INVALID_REQUEST",
            message="invalid execute request",
        )
    if behavior == "error_capacity":
        return _error_response(
            request.request_id,
            code="CAPACITY_EXCEEDED",
            message="worker at capacity",
        )

    provider_request_id = request.provider_request_id or f"sub-req-{uuid.uuid4().hex}"
    provider_operation_id = request.provider_operation_id or f"sub-op-{uuid.uuid4().hex}"
    invocation_id = request.invocation_id or f"sub-inv-{uuid.uuid4().hex}"

    if behavior == "spoof_execution_id":
        provider_request_id = request.execution_id

    if behavior == "spoof_provider_request_id":
        provider_request_id = "spoof-req-id"

    if behavior == "spoof_provider_operation_id":
        provider_operation_id = "spoof-op-id"

    if behavior == "invalid_response":
        raise ValueError("worker forced invalid response path")

    if behavior == "worker_crash":
        raise SystemExit(3)

    with state.lock:
        if state.active_requests >= state.max_concurrent:
            return _error_response(
                request.request_id,
                code="CAPACITY_EXCEEDED",
                message="worker at capacity",
            )
        state.active_requests += 1
        state.execute_count += 1

    try:
        if behavior == "slow":
            threading.Event().wait(timeout=3600.0)

        record = _OperationRecord(
            provider_request_id=provider_request_id,
            provider_operation_id=provider_operation_id,
            invocation_id=invocation_id,
            execution_id=request.execution_id,
            physical_status=_PhysicalStatus.RUNNING,
            status_spoof_field=(
                "provider_request_id"
                if behavior == "status_spoof_provider_request_id"
                else None
            ),
        )
        with state.lock:
            state.operations[provider_operation_id] = record

        result_value = request.payload.value
        record.result_value = result_value
        record.physical_status = _PhysicalStatus.SUCCEEDED
        parent_id = request.parent_execution_id or request.execution_id

        return WorkerResponse(
            request_id=request.request_id,
            status=WorkerResponseStatus.OK,
            provider_request_id=provider_request_id,
            provider_operation_id=provider_operation_id,
            result=WorkerExecuteResult(
                value=result_value,
                child_execution_id=request.execution_id,
                parent_execution_id=parent_id,
            ),
        )
    finally:
        with state.lock:
            state.active_requests = max(0, state.active_requests - 1)


class _WorkerHandler(StreamRequestHandler):
    server: _WorkerServer

    def handle(self) -> None:
        try:
            frame = _read_frame(self.request)
        except (ConnectionError, ValueError, OSError):
            return
        try:
            response = handle_worker_request(self.server.state, frame)
        except (json.JSONDecodeError, ValueError):
            return
        if response is None:
            return
        encoded = response.model_dump(mode="json")
        try:
            _write_frame(self.request, json.dumps(encoded, separators=(",", ":")).encode("utf-8"))
        except OSError:
            return


class _WorkerServer(ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, host: str, port: int, state: WorkerState) -> None:
        super().__init__((host, port), _WorkerHandler)
        self.state = state


def run_worker_server(*, host: str = "127.0.0.1", port: int = 0, max_concurrent: int = 8) -> int:
    import os

    auth = os.environ.get(_AUTH_ENV, "")
    state = WorkerState(auth_token=auth, max_concurrent=max_concurrent)
    server = _WorkerServer(host, port, state)
    bound_port = server.server_address[1]
    print(f"PORT={bound_port}", flush=True)
    server.serve_forever(poll_interval=0.2)
    return bound_port
