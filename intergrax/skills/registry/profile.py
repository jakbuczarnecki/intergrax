# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.skills.registry.catalog import get_bundle, iter_bundles


class SkillProfile(BaseModel):
    """Declarative Tier-3 skill enablement (Phase R-Skill.2)."""

    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(default_factory=list)
    enabled_bundles: list[str] = Field(default_factory=list)
    register_all_catalog_bundles: bool = False

    @field_validator("enabled", "enabled_bundles", mode="before")
    @classmethod
    def _coerce_str_list(cls, value: list[str] | None) -> list[str]:
        if not value:
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("enabled_bundles", mode="after")
    @classmethod
    def _normalize_bundle_ids(cls, value: list[str]) -> list[str]:
        return [item.lower() for item in value]

    def should_register_bundle(self, bundle_id: str, *, skill_ids: tuple[str, ...]) -> bool:
        if self.register_all_catalog_bundles:
            return True
        normalized = bundle_id.strip().lower()
        if normalized in self.enabled_bundles:
            return True
        if not self.enabled:
            return False
        enabled_set = set(self.enabled)
        return any(sid in enabled_set for sid in skill_ids)

    def is_skill_enabled(self, skill_id: str) -> bool:
        if self.register_all_catalog_bundles:
            return True
        if skill_id in self.enabled:
            return True
        if not self.enabled and not self.enabled_bundles:
            return False
        for bundle_id in self.enabled_bundles:
            try:
                entry = get_bundle(bundle_id)
            except KeyError:
                continue
            if skill_id in entry.skill_ids:
                return True
        return False
