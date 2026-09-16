# © Artur Czarnecki. All rights reserved.

"""Meaningful external side-effect policy request (GEC-5).

Generic description of a proposed external action that may create commitments,
mutations, disclosures, or other irreversible consequences. Reuses
``PolicyDecision`` / ``PolicyAction`` for evaluation outcomes.

Not a quote model, payment model, or provider authorization layer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.decision_governance_material import DecisionGovernanceMaterialRef
from intergrax.contracts.validation import validate_content_digest

SCHEMA_MEANINGFUL_SIDE_EFFECT_REQUEST_V1: Final = "meaningful_side_effect_request.v1"

_NON_EMPTY = Field(min_length=1)


class MeaningfulSideEffectKind(StrEnum):
    """Coarse impact classes — not an action catalog."""

    COMMITMENT = "commitment"
    MUTATION = "mutation"
    DISCLOSURE = "disclosure"
    ACCESS = "access"


class MeaningfulSideEffectRequest(BaseModel):
    """Proposed external side effect for policy evaluation before execution.

    ``action`` identifies the proposed side effect. When Decision governance
    material is attached, ``action`` must equal ``bound_action_kind`` and
    satisfy ``DecisionExecutionActionKind`` (see ``decision_authorization``).
    Domain adapters define concrete kind strings (e.g. ``external_work.accept_quote``).
    Domain-specific payloads belong in ``context`` / ``correlation`` — not as
    quote- or provider-SDK-typed fields on this model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["meaningful_side_effect_request.v1"] = (
        SCHEMA_MEANINGFUL_SIDE_EFFECT_REQUEST_V1
    )
    action: str = _NON_EMPTY
    kinds: tuple[MeaningfulSideEffectKind, ...] = Field(min_length=1)
    side_effect_scope_id: str = _NON_EMPTY
    side_effect_scope_digest: str | None = None
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    principal_id: str | None = None
    tenant_id: str | None = None
    resource: str | None = None
    external_target: str | None = None
    correlation: Mapping[str, Any] = Field(default_factory=dict)
    context: Mapping[str, Any] = Field(default_factory=dict)
    decision_governance_material: DecisionGovernanceMaterialRef | None = None

    @field_validator("action", "side_effect_scope_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id(cls, value: object) -> TaskId:
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("principal_id", "tenant_id", "resource", "external_target")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("side_effect_scope_digest")
    @classmethod
    def _validate_side_effect_scope_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_content_digest(value)


def resolve_meaningful_side_effect_execution_identity(
    *,
    task_id: object,
    run_id: object,
    attempt_id: object | None = None,
    execution_id: object | None = None,
) -> tuple[TaskId, RunId, AttemptId, ExecutionId]:
    """Bind a coherent AttemptId + ExecutionId pair from caller or active execution context."""
    validated_task = validate_task_id(task_id)
    validated_run = validate_run_id(run_id)
    has_attempt = attempt_id is not None
    has_execution = execution_id is not None
    if has_attempt != has_execution:
        raise ValueError(
            "meaningful side effect attempt_id and execution_id must be supplied together",
        )
    if has_attempt:
        validated_attempt = validate_attempt_id(attempt_id)
        validated_execution = validate_execution_id(execution_id)
        active_identity = peek_active_execution_identity()
        if active_identity is not None:
            active_run, active_attempt = active_identity
            active_execution = peek_active_execution_id()
            if active_run != validated_run:
                raise ValueError(
                    "meaningful side effect run_id does not match active execution",
                )
            if active_execution is None:
                raise RuntimeError("active ExecutionId required")
            if active_attempt != validated_attempt or active_execution != validated_execution:
                raise ValueError(
                    "meaningful side effect execution identity does not match active execution",
                )
        return (
            validated_task,
            validated_run,
            validated_attempt,
            validated_execution,
        )
    active_run, active_attempt = require_active_execution_identity()
    active_execution = require_active_execution_id()
    if active_run != validated_run:
        raise ValueError("meaningful side effect run_id does not match active execution")
    return (
        validated_task,
        validated_run,
        active_attempt,
        active_execution,
    )
