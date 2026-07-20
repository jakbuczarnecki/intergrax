# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class GovernedContractorRunRequestV1(BaseModel):
    tenant_id: str = "default"
    user_id: str = "default-user"
    session_id: Optional[str] = None
    message: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GovernedContractorRunResponseV1(BaseModel):
    task_id: str
    run_id: Optional[str] = None
    state: str
    answer: str = ""
    agent_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
