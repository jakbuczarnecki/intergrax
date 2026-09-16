# © Artur Czarnecki. All rights reserved.

"""ME-15 fixture catalog provider — plugin SPI proof only (not production authority)."""

from __future__ import annotations

from intergrax.skills.catalog import SkillCatalogEntry, SkillPackageResolution
from intergrax.skills.errors import DynamicSkillAcquisitionResolutionError
from intergrax.skills.identity import SkillPackageCandidate
from testing_support.canonical_me15_reference_skill import (
    ME15_DIGEST_V1,
    ME15_DIGEST_V2,
    ME15_PACKAGE_REFERENCE_V1,
    ME15_SKILL_LOGICAL_ID,
    ME15_VERSION_V1,
    ME15_VERSION_V2,
)

ME15_CATALOG_SOURCE_ID = "official.intergrax.me15"
ME15_CATALOG_ENTRY_ID = "catalog-entry-me15-instruction"


class Me15SkillCatalogProvider:
    """Deterministic multi-version skill catalog for marketplace ME-15 proofs."""

    @property
    def catalog_source_id(self) -> str:
        return ME15_CATALOG_SOURCE_ID

    def list_entries(self) -> list[SkillCatalogEntry]:
        return [
            SkillCatalogEntry(
                catalog_entry_id=ME15_CATALOG_ENTRY_ID,
                catalog_source_id=self.catalog_source_id,
                logical_skill_id=ME15_SKILL_LOGICAL_ID,
                package_reference=ME15_PACKAGE_REFERENCE_V1,
                display_name="ME-15 Canonical Instruction Skill",
            ),
        ]

    def resolve_package(
        self,
        entry: SkillCatalogEntry,
        *,
        version_selector: str,
    ) -> SkillPackageResolution:
        if entry.catalog_entry_id != ME15_CATALOG_ENTRY_ID:
            raise DynamicSkillAcquisitionResolutionError("unknown catalog entry")
        if version_selector == ME15_VERSION_V1:
            digest = ME15_DIGEST_V1
        elif version_selector == ME15_VERSION_V2:
            digest = ME15_DIGEST_V2
        else:
            raise DynamicSkillAcquisitionResolutionError(
                f"exact version {version_selector!r} is not available",
            )
        candidate = SkillPackageCandidate(
            logical_skill_id=ME15_SKILL_LOGICAL_ID,
            package_reference=entry.package_reference,
            package_version=version_selector,
            package_digest=digest,
        )
        return SkillPackageResolution(entry=entry, package_candidate=candidate)


class _CustomMe15SkillCatalogProvider:
    """Custom provider instance for pluginability gate — no subclass of default."""

    @property
    def catalog_source_id(self) -> str:
        return "custom.me15.provider"

    def list_entries(self) -> list[SkillCatalogEntry]:
        return [
            SkillCatalogEntry(
                catalog_entry_id="custom-me15-entry",
                catalog_source_id=self.catalog_source_id,
                logical_skill_id=ME15_SKILL_LOGICAL_ID,
                package_reference=ME15_PACKAGE_REFERENCE_V1,
                display_name="Custom ME-15 Provider Entry",
            ),
        ]

    def resolve_package(
        self,
        entry: SkillCatalogEntry,
        *,
        version_selector: str,
    ) -> SkillPackageResolution:
        provider = Me15SkillCatalogProvider()
        return provider.resolve_package(entry, version_selector=version_selector)


__all__ = [
    "ME15_CATALOG_ENTRY_ID",
    "ME15_CATALOG_SOURCE_ID",
    "Me15SkillCatalogProvider",
    "_CustomMe15SkillCatalogProvider",
]
