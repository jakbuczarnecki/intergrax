# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool catalog provider SPI — pluginable exact package resolution."""

from __future__ import annotations

from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.tools.identity import ToolPackageCandidate

_NON_EMPTY = Field(min_length=1)

SCHEMA_TOOL_CATALOG_ENTRY_V1: Final = "tool_catalog_entry.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class ToolCatalogEntry(BaseModel):
    """Catalog metadata for one tool package line — not runtime activation truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TOOL_CATALOG_ENTRY_V1
    catalog_entry_id: str = _NON_EMPTY
    catalog_source_id: str = _NON_EMPTY
    logical_tool_id: str = _NON_EMPTY
    package_reference: str = _NON_EMPTY
    display_name: str = _NON_EMPTY

    @field_validator(
        "catalog_entry_id",
        "catalog_source_id",
        "logical_tool_id",
        "package_reference",
        "display_name",
    )
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


class ToolPackageResolution(BaseModel):
    """Exact package resolution from a catalog provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entry: ToolCatalogEntry
    package_candidate: ToolPackageCandidate


class ToolCatalogProvider(Protocol):
    """Port for tool catalog discovery and exact version resolution."""

    @property
    def catalog_source_id(self) -> str:
        """Stable provider instance id."""

    def list_entries(self) -> list[ToolCatalogEntry]:
        """List catalog entries exposed by this provider."""

    def resolve_package(
        self,
        entry: ToolCatalogEntry,
        *,
        version_selector: str,
    ) -> ToolPackageResolution:
        """Resolve entry to exact version — MUST NOT fall back to latest."""


class ToolCatalogProviderRegistry:
    """Registry of ``ToolCatalogProvider`` contracts — not concrete implementations."""

    def __init__(self, providers: dict[str, ToolCatalogProvider]) -> None:
        self._providers = dict(providers)

    @property
    def registered_source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def require(self, catalog_source_id: str) -> ToolCatalogProvider:
        from intergrax.tools.errors import DynamicToolAcquisitionResolutionError

        provider = self._providers.get(catalog_source_id)
        if provider is None:
            raise DynamicToolAcquisitionResolutionError(
                f"tool catalog source {catalog_source_id} has no registered provider",
            )
        if provider.catalog_source_id != catalog_source_id:
            raise DynamicToolAcquisitionResolutionError(
                "tool catalog provider instance id does not match registry key",
            )
        return provider


__all__ = [
    "SCHEMA_TOOL_CATALOG_ENTRY_V1",
    "ToolCatalogEntry",
    "ToolCatalogProvider",
    "ToolCatalogProviderRegistry",
    "ToolPackageResolution",
]
