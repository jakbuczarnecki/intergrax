# © Artur Czarnecki. All rights reserved.

"""Framed JSON wire protocol for subprocess delegated execution worker."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_NON_EMPTY = Field(min_length=1)


class WorkerRpcMethod(StrEnum):
    EXECUTE = "execute"
    STATUS = "status"
    CANCEL = "cancel"
    REATTACH = "reattach"
    METRICS = "metrics"


class WorkerExecutePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = _NON_EMPTY
    behavior: str | None = None


class WorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: WorkerRpcMethod
    request_id: str = _NON_EMPTY
    auth_token: str | None = None
    payload: WorkerExecutePayload | None = None
    provider_request_id: str | None = None
    provider_operation_id: str | None = None
    invocation_id: str | None = None
    execution_id: str | None = None
    parent_execution_id: str | None = None


class WorkerResponseStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class WorkerExecuteResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str
    child_execution_id: str
    parent_execution_id: str


class WorkerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = _NON_EMPTY
    status: WorkerResponseStatus
    result: WorkerExecuteResult | None = None
    provider_request_id: str | None = None
    provider_operation_id: str | None = None
    invocation_id: str | None = None
    physical_status: str | None = None
    reattachment_kind: str | None = None
    execute_count: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    wire_kind: Literal["worker_response.v1"] = "worker_response.v1"
