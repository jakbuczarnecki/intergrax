# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import base64

from pydantic import BaseModel, Field, field_validator

from intergrax.queueing.contracts.task_queue import TaskStatus


class MessageBusEnqueueInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    task_name: str = Field(..., min_length=1)
    payload_base64: str = Field(..., min_length=1)
    idempotency_key: str | None = None

    @field_validator("payload_base64")
    @classmethod
    def _validate_base64(cls, value: str) -> str:
        base64.b64decode(value, validate=True)
        return value


class MessageBusEnqueueOutput(BaseModel):
    task_id: str
    provider: str
    tenant_id: str | None = None


class MessageBusGetStatusInput(BaseModel):
    task_id: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    tenant_id: str | None = None


class MessageBusGetStatusOutput(BaseModel):
    task_id: str
    status: TaskStatus


class MessageBusGetResultInput(BaseModel):
    task_id: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    tenant_id: str | None = None


class MessageBusGetResultOutput(BaseModel):
    task_id: str
    completed: bool
    status: TaskStatus | None = None
    output_base64: str = ""
    error_message: str = ""
    attempts: int = 0
