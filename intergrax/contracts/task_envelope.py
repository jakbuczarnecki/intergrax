# © Artur Czarnecki. All rights reserved.

"""Canonical task intake envelope (FAUDIT-INTAKE.1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskEnvelope(BaseModel):
    """Unified intake contract across HTTP, CLI, worker, and interaction adapters."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    user_id: str
    message: str = ""
    session_id: str | None = None
    agent_id: str | None = None
    workspace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def with_actor(self, *, actor_kind: str, actor_id: str) -> TaskEnvelope:
        meta = dict(self.metadata)
        meta["actor_kind"] = actor_kind
        meta["actor_id"] = actor_id
        return self.model_copy(update={"metadata": meta})
