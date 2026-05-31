# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Declarative Tier-3 tool selection (Phase O.2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolProfile(BaseModel):
    """
    Typed tool enablement for a Tier-3 application.

    ``enabled`` lists ``tool_id`` values (e.g. ``jira.search_tasks``).
    ``enabled_bundles`` lists catalog bundle ids (e.g. ``jira``) — all tools
    from matching bundles are registered when the bundle is invoked.

    When both are empty and ``register_all_catalog_bundles`` is False (default),
    the registry stays empty unless explicit ``ToolProvider`` modules add tools.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(default_factory=list)
    enabled_bundles: list[str] = Field(default_factory=list)
    register_all_catalog_bundles: bool = False
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)

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

    def options_for_tool(self, tool_id: str) -> dict[str, Any]:
        return dict(self.options.get(tool_id, {}))

    def is_tool_enabled(self, tool_id: str) -> bool:
        if self.register_all_catalog_bundles:
            return True
        if tool_id in self.enabled:
            return True
        if not self.enabled and not self.enabled_bundles:
            return False
        from intergrax.tools.registry.catalog import get_bundle

        for bundle_id in self.enabled_bundles:
            try:
                entry = get_bundle(bundle_id)
            except KeyError:
                continue
            if tool_id in entry.tool_ids:
                return True
        return False

    def should_register_bundle(self, bundle_id: str, *, tool_ids: tuple[str, ...]) -> bool:
        normalized = bundle_id.strip().lower()
        if self.register_all_catalog_bundles:
            return True
        if normalized in self.enabled_bundles:
            return True
        if self.enabled:
            return any(tool_id in self.enabled for tool_id in tool_ids)
        return False

    @classmethod
    def lab(cls) -> ToolProfile:
        """Laboratory default — no catalog tools until bundles ship (Phase O.3+)."""
        return cls()

    @classmethod
    def all_catalog(cls) -> ToolProfile:
        """Enable every bundle registered in the tool catalog."""
        return cls(register_all_catalog_bundles=True)


def default_lab_tool_profile() -> ToolProfile:
    return ToolProfile.lab()
