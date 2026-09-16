# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill catalog provider SPI — pluginable exact package resolution."""

from __future__ import annotations

from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.skills.identity import SkillPackageCandidate

_NON_EMPTY = Field(min_length=1)

SCHEMA_SKILL_CATALOG_ENTRY_V1: Final = "skill_catalog_entry.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class SkillCatalogEntry(BaseModel):
    """Catalog metadata for one skill package line — not runtime binding truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_SKILL_CATALOG_ENTRY_V1
    catalog_entry_id: str = _NON_EMPTY
    catalog_source_id: str = _NON_EMPTY
    logical_skill_id: str = _NON_EMPTY
    package_reference: str = _NON_EMPTY
    display_name: str = _NON_EMPTY

    @field_validator(
        "catalog_entry_id",
        "catalog_source_id",
        "logical_skill_id",
        "package_reference",
        "display_name",
    )
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


class SkillPackageResolution(BaseModel):
    """Exact package resolution from a catalog provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entry: SkillCatalogEntry
    package_candidate: SkillPackageCandidate


class SkillCatalogProvider(Protocol):
    """Port for skill catalog discovery and exact version resolution."""

    @property
    def catalog_source_id(self) -> str:
        """Stable provider instance id."""

    def list_entries(self) -> list[SkillCatalogEntry]:
        """List catalog entries exposed by this provider."""

    def resolve_package(
        self,
        entry: SkillCatalogEntry,
        *,
        version_selector: str,
    ) -> SkillPackageResolution:
        """Resolve entry to exact version — MUST NOT fall back to latest."""


class SkillCatalogProviderRegistry:
    """Registry of ``SkillCatalogProvider`` contracts — not concrete implementations."""

    def __init__(self, providers: dict[str, SkillCatalogProvider]) -> None:
        self._providers = dict(providers)

    def require(self, catalog_source_id: str) -> SkillCatalogProvider:
        from intergrax.skills.errors import DynamicSkillAcquisitionResolutionError

        provider = self._providers.get(catalog_source_id)
        if provider is None:
            raise DynamicSkillAcquisitionResolutionError(
                f"skill catalog source {catalog_source_id} has no registered provider",
            )
        if provider.catalog_source_id != catalog_source_id:
            raise DynamicSkillAcquisitionResolutionError(
                "skill catalog provider instance id does not match registry key",
            )
        return provider


__all__ = [
    "SCHEMA_SKILL_CATALOG_ENTRY_V1",
    "SkillCatalogEntry",
    "SkillCatalogProvider",
    "SkillCatalogProviderRegistry",
    "SkillPackageResolution",
]
