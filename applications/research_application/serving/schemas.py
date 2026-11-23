# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ResearchRunRequestV1(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    session_id: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("tenant_id must not be blank")
        return stripped


class ResearchRunResponseV1(BaseModel):
    task_id: str
    run_id: str | None = None
    state: str
    answer: str
    graph_id: str | None = None
    agent_ids: list[str] = Field(default_factory=list)
