# © Artur Czarnecki. All rights reserved.

"""First-class provider invocation identity (PC-3).

Intergrax ``invocation_id`` is created by the execution boundary before the
provider call. Optional provider-native request/operation ids are separate.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_PROVIDER_INVOCATION_V1: Final = "provider_invocation.v1"
SCHEMA_PROVIDER_INVOCATION_OUTCOME_V1: Final = "provider_invocation_outcome.v1"
_NON_EMPTY = Field(min_length=1)


class ProviderInvocationStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class ProviderInvocation(BaseModel):
    """Host/runtime invocation record created before the provider call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["provider_invocation.v1"] = SCHEMA_PROVIDER_INVOCATION_V1
    invocation_id: str = _NON_EMPTY
    provider_id: str = _NON_EMPTY
    operation: str = _NON_EMPTY
    task_id: str = _NON_EMPTY
    run_id: str = _NON_EMPTY
    external_task_id: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None
    request_digest: str = _NON_EMPTY
    started_at: datetime
    # Optional partner-native ids — never aliases of invocation_id.
    provider_request_id: str | None = None
    provider_operation_id: str | None = None

    @field_validator(
        "invocation_id",
        "provider_id",
        "operation",
        "task_id",
        "run_id",
        "request_digest",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ProviderInvocationOutcome(BaseModel):
    """Outcome bound to a prior ``ProviderInvocation``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["provider_invocation_outcome.v1"] = (
        SCHEMA_PROVIDER_INVOCATION_OUTCOME_V1
    )
    invocation_id: str = _NON_EMPTY
    status: ProviderInvocationStatus
    completed_at: datetime
    response_digest: str | None = None
    external_status: str | None = None
    error_code: str | None = None
    provider_request_id: str | None = None
    provider_operation_id: str | None = None

    @field_validator("invocation_id")
    @classmethod
    def _strip_invocation_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("invocation_id must be non-empty")
        return normalized
