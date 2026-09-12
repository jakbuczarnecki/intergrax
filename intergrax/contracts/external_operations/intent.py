# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""External operation intent — LLM proposes; admission governs execution (R1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution_identity import TaskId, validate_task_id

SCHEMA_EXTERNAL_OPERATION_INTENT_V1: Final = "external_operation_intent.v1"


class ExternalOperationType(StrEnum):
    """Typed external operation — never a raw command string."""

    LLM_PROVIDER_CALL = "LLM_PROVIDER_CALL"
    CONNECTOR_RESTART = "CONNECTOR_RESTART"
    CONNECTOR_UPDATE = "CONNECTOR_UPDATE"
    INTEGRATION_INVOKE = "INTEGRATION_INVOKE"
    CRM_UPDATE = "CRM_UPDATE"
    CLOUD_MUTATION = "CLOUD_MUTATION"
    MESSAGING_POST = "MESSAGING_POST"


class ExternalOperationIntent(BaseModel):
    """Mandatory scope for every external operation (who / why / what / tenant)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    task_id: TaskId
    operation_type: ExternalOperationType
    target_resource: str = Field(min_length=1, max_length=512)
    requested_by: str = Field(min_length=1, max_length=256)
    justification: str = Field(min_length=1, max_length=2048)
    created_at: datetime

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: TaskId) -> TaskId:
        return validate_task_id(value)

    @field_validator("target_resource", "requested_by", "justification")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field must be non-empty after strip")
        return stripped


def mint_external_operation_intent_id() -> str:
    return f"ext_op_intent_{uuid4().hex}"
