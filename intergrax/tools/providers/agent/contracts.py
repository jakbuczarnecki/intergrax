# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentListAgentsInput(BaseModel):
    limit: int = Field(default=100, ge=1, le=500)


class AgentSummaryOutput(BaseModel):
    agent_id: str
    capabilities: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)


class AgentListAgentsOutput(BaseModel):
    agents: list[AgentSummaryOutput] = Field(default_factory=list)
    total: int = 0


class AgentGetContractInput(BaseModel):
    agent_id: str = Field(..., min_length=1)


class AgentGetContractOutput(BaseModel):
    found: bool = False
    agent_id: str = ""
    contract: dict = Field(default_factory=dict)
