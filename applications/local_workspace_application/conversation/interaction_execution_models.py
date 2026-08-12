# © Artur Czarnecki. All rights reserved.

"""Typed, frontend-neutral results for deterministic conversation execution."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningRequest,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationExecutionContextV1,
)


class ConversationActionExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_DEPENDENCY = "blocked_dependency"
    BLOCKED_CLARIFICATION = "blocked_clarification"
    SKIPPED = "skipped"


class ConversationInteractionOverallStatus(StrEnum):
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    CLARIFICATION_REQUIRED = "clarification_required"
    FAILED = "failed"


class ConversationExecutionError(BaseModel):
    """A stable error code without provider, path, tenant or exception details."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=128)
    action_id: str | None = Field(default=None, max_length=128)

    @field_validator("code", "action_id", mode="after")
    @classmethod
    def _normalize_identifier(cls, value: str | None) -> str | None:
        return _normalize_identifier(value) if value is not None else None


class ConversationExecutionArtifact(BaseModel):
    """Safe structured data produced by one action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_type: str = Field(min_length=1, max_length=128)
    data: Mapping[str, Any] = Field(default_factory=dict)


class ConversationExecutionClarification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    clarification_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2_000)
    blocks_action_ids: tuple[str, ...] = ()


class ConversationActionExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str = Field(min_length=1, max_length=128)
    action_type: str = Field(min_length=1, max_length=128)
    status: ConversationActionExecutionStatus
    artifact: ConversationExecutionArtifact | None = None
    error: ConversationExecutionError | None = None
    resolved_workspace_id: str | None = Field(default=None, max_length=128)
    started_at: datetime
    completed_at: datetime

    @field_validator("action_id", "action_type", "resolved_workspace_id", mode="after")
    @classmethod
    def _normalize_identifier(cls, value: str | None) -> str | None:
        return _normalize_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _validate_integrity(self) -> ConversationActionExecutionResult:
        _validate_utc_timestamp(self.started_at, "started_at")
        _validate_utc_timestamp(self.completed_at, "completed_at")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must be greater than or equal to started_at")
        if self.error is not None and self.error.action_id not in (None, self.action_id):
            raise ValueError("error.action_id must match action_id")
        if self.status is ConversationActionExecutionStatus.COMPLETED:
            if self.artifact is None or self.error is not None:
                raise ValueError("completed action results require an artifact and no error")
        elif self.status is ConversationActionExecutionStatus.FAILED:
            if self.error is None or self.artifact is not None:
                raise ValueError("failed action results require an error and no artifact")
        elif self.status is ConversationActionExecutionStatus.BLOCKED_DEPENDENCY:
            if self.error is None or self.error.code != "blocked_dependency" or self.artifact is not None:
                raise ValueError("blocked dependency results require the canonical error")
        elif self.status is ConversationActionExecutionStatus.BLOCKED_CLARIFICATION:
            if self.error is None or self.error.code != "blocked_clarification" or self.artifact is not None:
                raise ValueError("blocked clarification results require the canonical error")
        elif self.status is ConversationActionExecutionStatus.SKIPPED:
            if self.artifact is not None or self.error is not None:
                raise ValueError("skipped action results cannot contain artifacts or errors")
        return self


class ConversationInteractionExecutionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    tenant_id: str = Field(min_length=1, max_length=128)
    planning_request: ConversationPlanningRequest
    interaction_plan: ConversationInteractionPlan
    execution_context: ConversationExecutionContextV1
    execution_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("tenant_id")
    @classmethod
    def _normalize_tenant_id(cls, value: str) -> str:
        return value.strip()


class ConversationInteractionExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    plan_version: str = Field(min_length=1, max_length=128)
    started_at: datetime
    completed_at: datetime
    status: ConversationInteractionOverallStatus
    action_results: tuple[ConversationActionExecutionResult, ...] = ()
    clarifications: tuple[ConversationExecutionClarification, ...] = ()
    active_workspace_id: str | None = Field(default=None, max_length=128)
    created_resources: tuple[ConversationExecutionArtifact, ...] = ()
    ask_runs: tuple[ConversationExecutionArtifact, ...] = ()
    response_data: tuple[ConversationExecutionArtifact, ...] = ()
    thread_memory_user_text: str | None = Field(default=None, max_length=16_000)
    error: ConversationExecutionError | None = None

    @field_validator("execution_id", "tenant_id", "plan_version", mode="after")
    @classmethod
    def _normalize_identifier(cls, value: str) -> str:
        return _normalize_identifier(value)

    @model_validator(mode="after")
    def _validate_integrity(self) -> ConversationInteractionExecutionResult:
        _validate_utc_timestamp(self.started_at, "started_at")
        _validate_utc_timestamp(self.completed_at, "completed_at")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must be greater than or equal to started_at")

        action_ids = [item.action_id for item in self.action_results]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action result action_ids must be unique")

        if self.error is not None:
            if self.status is not ConversationInteractionOverallStatus.FAILED:
                raise ValueError("top-level error requires a failed result")
            if self.action_results:
                raise ValueError("top-level error is reserved for preflight failures")
            if self.error.action_id is not None:
                raise ValueError("top-level error.action_id must be None")

        statuses = {item.status for item in self.action_results}
        has_completed = ConversationActionExecutionStatus.COMPLETED in statuses
        has_non_completed = any(
            status is not ConversationActionExecutionStatus.COMPLETED
            for status in statuses
        )
        if self.status is ConversationInteractionOverallStatus.COMPLETED:
            if self.error is not None or has_non_completed:
                raise ValueError("completed results require only completed actions")
        elif self.status is ConversationInteractionOverallStatus.FAILED:
            has_failed = ConversationActionExecutionStatus.FAILED in statuses
            if self.error is None and (not has_failed or has_completed):
                raise ValueError("failed results require a preflight or all-actions failure")
            if self.error is not None and self.action_results:
                raise ValueError("preflight failures cannot contain action results")
        elif self.status is ConversationInteractionOverallStatus.CLARIFICATION_REQUIRED:
            if ConversationActionExecutionStatus.BLOCKED_CLARIFICATION not in statuses:
                raise ValueError("clarification-required results need a blocked action")
        elif self.status is ConversationInteractionOverallStatus.PARTIALLY_COMPLETED:
            if not has_completed or not has_non_completed:
                raise ValueError("partial results require completed and non-completed actions")

        completed_artifacts = tuple(
            item.artifact
            for item in self.action_results
            if item.status is ConversationActionExecutionStatus.COMPLETED
            and item.artifact is not None
        )
        if any(
            artifact not in tuple(
                item.artifact
                for item in self.action_results
                if item.status is ConversationActionExecutionStatus.COMPLETED
                and item.action_type == "workspace.create"
                and item.artifact is not None
            )
            for artifact in self.created_resources
        ):
            raise ValueError("created_resources contains an unrelated artifact")
        if any(
            artifact not in tuple(
                item.artifact
                for item in self.action_results
                if item.status is ConversationActionExecutionStatus.COMPLETED
                and item.action_type == "workspace.ask"
                and item.artifact is not None
            )
            for artifact in self.ask_runs
        ):
            raise ValueError("ask_runs contains an unrelated artifact")
        if any(artifact not in completed_artifacts for artifact in self.response_data):
            raise ValueError("response_data must contain completed action artifacts")
        return self


def _normalize_identifier(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("identifier must not be blank")
    if len(normalized) > 128:
        raise ValueError("identifier exceeds 128 characters")
    return normalized


def _validate_utc_timestamp(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
