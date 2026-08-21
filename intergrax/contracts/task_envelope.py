# © Artur Czarnecki. All rights reserved.

"""Canonical task intake envelope (FAUDIT-INTAKE.1 / AUDIT-IDEAL-3.1).

``TaskEnvelope`` is the single normalized intake contract. ``Task`` and
``RuntimeRequest`` round-trip via ``to_envelope`` / ``from_envelope``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.delegation_authority import ParentExecutionAuthority


class TaskSlaClass(str, Enum):
    INTERACTIVE = "interactive"
    BATCH = "batch"
    BACKGROUND = "background"


class TaskRiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    REGULATED = "regulated"


class TaskEnvelope(BaseModel):
    """Unified intake contract across HTTP, CLI, worker, and interaction adapters."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    user_id: str
    message: str = ""
    session_id: str | None = None
    agent_id: str | None = None
    workspace_id: str | None = None
    sla_class: TaskSlaClass = TaskSlaClass.INTERACTIVE
    risk_tier: TaskRiskTier = TaskRiskTier.LOW
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    canonical_identity: RequestIdentity | None = None
    execution_authority: ParentExecutionAuthority | None = None

    @field_validator("tenant_id", "user_id")
    @classmethod
    def _non_empty_identity(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("tenant_id and user_id must be non-empty")
        return value.strip()

    def with_actor(self, *, actor_kind: str, actor_id: str) -> TaskEnvelope:
        meta = dict(self.metadata)
        meta["actor_kind"] = actor_kind
        meta["actor_id"] = actor_id
        return self.model_copy(update={"metadata": meta})
