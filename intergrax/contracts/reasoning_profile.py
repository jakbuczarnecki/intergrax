# © Artur Czarnecki. All rights reserved.

"""Reasoning profile for Tier-3 hosts (COG-5.1)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.llm_adapters.registry.profile import LLMProfile


class ReasoningProfile(BaseModel):
    """Planner/classifier LLM selection and prompt ids."""

    model_config = ConfigDict(extra="forbid")

    planner_llm_profile: LLMProfile | None = None
    planner_llm_profile_id: str | None = None
    planner_prompt_id: str = "nexus_task_planner"
    planner_parse_retries: int = Field(default=0, ge=0, le=8)
    denied_planner_model_ids: list[str] = Field(default_factory=list)
    tool_planner_prompt_id: str = "tools_agent_planner"
    engine_planner_prompt_id: str = "planner_default"
    classifier_prompt_id: str = "nexus_task_classifier"
