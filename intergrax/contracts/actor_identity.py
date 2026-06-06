# © Artur Czarnecki. All rights reserved.

"""Actor identity model for harness trust boundaries (FAUDIT-ID.1)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ActorKind(str, Enum):
    USER = "user"
    SERVICE = "service"
    AGENT = "agent"


class ActorIdentity(BaseModel):
    """Resolved actor on the execution path."""

    model_config = ConfigDict(extra="forbid")

    kind: ActorKind
    actor_id: str
    tenant_id: str
    delegated_from: str | None = None
    permission_scopes: tuple[str, ...] = Field(default_factory=tuple)

    def allows_scope(self, scope: str) -> bool:
        if not self.permission_scopes:
            return True
        return scope in self.permission_scopes
