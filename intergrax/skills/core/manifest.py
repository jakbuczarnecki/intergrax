# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill bundle manifest for catalog registration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from intergrax.skills.registry.catalog import SkillBundleStatus


class SkillBundleManifest(BaseModel):
    """Declarative metadata for a skill bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    skill_ids: tuple[str, ...] = ()
    status: SkillBundleStatus = SkillBundleStatus.STABLE
    description: str = ""

    @field_validator("bundle_id")
    @classmethod
    def _normalize_bundle_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("bundle_id must be non-empty")
        return normalized

    @field_validator("skill_ids", mode="before")
    @classmethod
    def _coerce_skill_ids(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value.strip(),) if value.strip() else ()
        return tuple(str(item).strip() for item in value if str(item).strip())
