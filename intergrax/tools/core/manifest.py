# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool bundle manifest for catalog registration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.tools.registry.catalog import ToolBundleStatus


class ToolBundleManifest(BaseModel):
    """Declarative metadata for a tool bundle (catalog row identity)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    tool_ids: tuple[str, ...] = ()
    status: ToolBundleStatus = ToolBundleStatus.STABLE
    description: str = ""

    @field_validator("bundle_id")
    @classmethod
    def _normalize_bundle_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("bundle_id must be non-empty")
        return normalized

    @field_validator("tool_ids", mode="before")
    @classmethod
    def _coerce_tool_ids(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value.strip(),) if value.strip() else ()
        return tuple(str(item).strip() for item in value if str(item).strip())
