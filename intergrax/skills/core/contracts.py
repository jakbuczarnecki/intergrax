# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill manifest contracts (architecture §7.1.8, Phase R-Skill.1)."""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.skills.version_validation import validate_skill_version


class SkillRiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SkillManifest(BaseModel):
    """
    Versioned composable capability pack — tools + prompt refs + optional policy fragment.

    Skills are NOT LLM-invokable tools; the runtime resolves them into allow-lists and context.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: str
    version: str = "1.0.0"
    description: str
    tool_ids: tuple[str, ...] = ()
    prompt_instruction_ids: tuple[str, ...] = ()
    policy_fragment_id: str | None = None
    risk_tier: SkillRiskTier = SkillRiskTier.LOW
    tags: tuple[str, ...] = ()
    requires_skills: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Other skill_ids merged before this skill (transitive).",
    )

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        return validate_skill_version(value)

    @field_validator("skill_id")
    @classmethod
    def _validate_skill_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("skill_id must be non-empty")
        if " " in normalized:
            raise ValueError("skill_id must not contain spaces")
        return normalized

    @field_validator("tool_ids", "prompt_instruction_ids", "tags", mode="before")
    @classmethod
    def _coerce_tuple(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value.strip(),) if value.strip() else ()
        return tuple(str(item).strip() for item in value if str(item).strip())

    @field_validator("tool_ids")
    @classmethod
    def _unique_tool_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("tool_ids must be unique within a skill manifest")
        return value

    @property
    def qualified_id(self) -> str:
        return f"{self.skill_id}@{self.version}"

    def with_version(self, version: str) -> Self:
        return self.model_copy(update={"version": version})
