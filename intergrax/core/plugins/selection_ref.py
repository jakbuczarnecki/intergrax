# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Metadata-only external plugin selection locator (P0-A-R3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from intergrax.core.distribution.package_identity import normalize_distribution_package_name
from intergrax.core.plugins.discovery import EntryPointSpec


def _require_non_empty_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


class PlatformPluginSelectionRef(BaseModel):
    """Stable pre-manifest locator for one requested external platform plugin."""

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

    def unresolved_entry_point_spec(self) -> EntryPointSpec:
        """Synthetic spec for admission evidence when metadata discovery misses the locator."""
        return EntryPointSpec(
            name=self.entry_point_name,
            group=self.entry_point_group,
            value="<unresolved>",
            distribution=self.distribution,
        )


def entry_point_spec_matches_selection_ref(
    spec: EntryPointSpec,
    ref: PlatformPluginSelectionRef,
) -> bool:
    if spec.group != ref.entry_point_group or spec.name != ref.entry_point_name:
        return False
    if spec.distribution is None:
        return False
    try:
        normalized = normalize_distribution_package_name(spec.distribution)
    except ValueError:
        return False
    return normalized == ref.distribution
