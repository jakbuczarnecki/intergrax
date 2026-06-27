# © Artur Czarnecki. All rights reserved.

"""Neutral per-roster binding slice consumed by agent runtime merge."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_budget import AgentBudgetSlice
from intergrax.contracts.memory_scope import MemoryScope


class AgentRunBinding(BaseModel):
    """Runtime binding slice — no application manifest or import paths."""

    model_config = ConfigDict(extra="forbid")

    memory_scope_override: MemoryScope | None = None
    rag_collection_override: str | None = None
    tool_allowlist_extra: list[str] = Field(default_factory=list)
    tool_denylist: list[str] = Field(default_factory=list)
    org_role_id: str | None = None
    budget_slice: AgentBudgetSlice | None = None
