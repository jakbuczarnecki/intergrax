# © Artur Czarnecki. All rights reserved.

"""Typed, frontend-neutral results for deterministic conversation execution."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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

    @field_validator("error")
    @classmethod
    def _error_requires_failed_or_blocked_status(
        cls,
        value: ConversationExecutionError | None,
    ) -> ConversationExecutionError | None:
        return value


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
    status: ConversationInteractionOverallStatus
    action_results: tuple[ConversationActionExecutionResult, ...] = ()
    clarifications: tuple[ConversationExecutionClarification, ...] = ()
    active_workspace_id: str | None = Field(default=None, max_length=128)
    created_resources: tuple[ConversationExecutionArtifact, ...] = ()
    ask_runs: tuple[ConversationExecutionArtifact, ...] = ()
    response_data: tuple[ConversationExecutionArtifact, ...] = ()
    error: ConversationExecutionError | None = None
