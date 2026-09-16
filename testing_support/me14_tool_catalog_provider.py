# © Artur Czarnecki. All rights reserved.

"""ME-14 fixture catalog provider — plugin SPI proof only (not production authority)."""

from __future__ import annotations

from intergrax.tools.catalog import ToolCatalogEntry, ToolPackageResolution
from intergrax.tools.errors import DynamicToolAcquisitionResolutionError
from intergrax.tools.identity import ToolPackageCandidate
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_DIGEST_V2,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
    ME14_VERSION_V2,
)

ME14_CATALOG_SOURCE_ID = "official.intergrax.me14"
ME14_CATALOG_ENTRY_ID = "catalog-entry-me14-echo"


class Me14ToolCatalogProvider:
    """Deterministic multi-version tool catalog for marketplace ME-14 proofs."""

    @property
    def catalog_source_id(self) -> str:
        return ME14_CATALOG_SOURCE_ID

    def list_entries(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(
                catalog_entry_id=ME14_CATALOG_ENTRY_ID,
                catalog_source_id=self.catalog_source_id,
                logical_tool_id=ME14_TOOL_LOGICAL_ID,
                package_reference=ME14_PACKAGE_REFERENCE_V1,
                display_name="ME-14 Canonical Echo Tool",
            ),
        ]

    def resolve_package(
        self,
        entry: ToolCatalogEntry,
        *,
        version_selector: str,
    ) -> ToolPackageResolution:
        if entry.catalog_entry_id != ME14_CATALOG_ENTRY_ID:
            raise DynamicToolAcquisitionResolutionError("unknown catalog entry")
        if version_selector == ME14_VERSION_V1:
            digest = ME14_DIGEST_V1
        elif version_selector == ME14_VERSION_V2:
            digest = ME14_DIGEST_V2
        else:
            raise DynamicToolAcquisitionResolutionError(
                f"exact version {version_selector!r} is not available",
            )
        candidate = ToolPackageCandidate(
            logical_tool_id=ME14_TOOL_LOGICAL_ID,
            package_reference=entry.package_reference,
            package_version=version_selector,
            package_digest=digest,
        )
        return ToolPackageResolution(entry=entry, package_candidate=candidate)


class _CustomMe14ToolCatalogProvider:
    """Custom provider instance for pluginability gate — no subclass of default."""

    @property
    def catalog_source_id(self) -> str:
        return "custom.me14.provider"

    def list_entries(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(
                catalog_entry_id="custom-me14-entry",
                catalog_source_id=self.catalog_source_id,
                logical_tool_id=ME14_TOOL_LOGICAL_ID,
                package_reference=ME14_PACKAGE_REFERENCE_V1,
                display_name="Custom ME-14 Provider Entry",
            ),
        ]

    def resolve_package(
        self,
        entry: ToolCatalogEntry,
        *,
        version_selector: str,
    ) -> ToolPackageResolution:
        provider = Me14ToolCatalogProvider()
        return provider.resolve_package(entry, version_selector=version_selector)


__all__ = [
    "ME14_CATALOG_ENTRY_ID",
    "ME14_CATALOG_SOURCE_ID",
    "Me14ToolCatalogProvider",
    "_CustomMe14ToolCatalogProvider",
]
