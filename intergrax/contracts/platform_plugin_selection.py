# © Artur Czarnecki. All rights reserved.

"""Metadata-only external plugin selection locator — public contract."""

from __future__ import annotations

from packaging.utils import InvalidName, canonicalize_name
from pydantic import BaseModel, ConfigDict, field_validator


def _require_non_empty_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def normalize_distribution_package_name(value: str) -> str:
    normalized = _require_non_empty_text(value, field_name="package name")
    try:
        return canonicalize_name(normalized, validate=True)
    except InvalidName as exc:
        raise ValueError(f"invalid package name: {normalized!r}") from exc


class PlatformPluginSelectionRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plugin_id: str
    entry_point_group: str
    entry_point_name: str
    distribution: str

    @field_validator("plugin_id")
    @classmethod
    def _validate_plugin_id(cls, value: str) -> str:
        return _require_non_empty_text(value, field_name="plugin_id")

    @field_validator("entry_point_group")
    @classmethod
    def _validate_entry_point_group(cls, value: str) -> str:
        return _require_non_empty_text(value, field_name="entry_point_group")

    @field_validator("entry_point_name")
    @classmethod
    def _validate_entry_point_name(cls, value: str) -> str:
        return _require_non_empty_text(value, field_name="entry_point_name")

    @field_validator("distribution")
    @classmethod
    def _validate_distribution(cls, value: str) -> str:
        return normalize_distribution_package_name(value)

    @property
    def location_key(self) -> tuple[str, str, str]:
        return (self.distribution, self.entry_point_group, self.entry_point_name)
