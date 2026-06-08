# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillResolveInput(BaseModel):
    skill_ids: list[str] = Field(..., min_length=1)


class SkillResolveOutput(BaseModel):
    skill_ids: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    prompt_instruction_ids: list[str] = Field(default_factory=list)
    policy_fragment_ids: list[str] = Field(default_factory=list)
    risk_tier: str = ""
