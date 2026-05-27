# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchRunRequestV1(BaseModel):
    tenant_id: str = "research-tenant"
    user_id: str = "research-user"
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ResearchRunResponseV1(BaseModel):
    task_id: str
    run_id: str | None = None
    state: str
    answer: str
    graph_id: str | None = None
    agent_ids: list[str] = Field(default_factory=list)
