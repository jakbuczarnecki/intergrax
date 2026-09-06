# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill bundle catalog → capability catalog entry adapter (Stage 2)."""

from __future__ import annotations

from typing import Final

from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.identity import (
    CapabilityDiscoveryIdentity,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance
from intergrax.contracts.capability_catalog.skill_version_binding import (
    SkillVersionBindingDisposition,
)
from intergrax.skills.registry.catalog import SkillBundleEntry, iter_bundles
from intergrax.skills.registry.catalog_manifests import catalog_manifest_for_skill_id

SKILL_BUILTIN_CATALOG_SOURCE_ID: Final = "skills.catalog.builtin"

_BUILTIN_SOURCE = CapabilitySourceIdentity(
    source_id=SKILL_BUILTIN_CATALOG_SOURCE_ID,
    source_kind=CapabilitySourceKind.BUILTIN,
)


def project_skill_bundle_entry(
    bundle: SkillBundleEntry,
    skill_id: str,
) -> CapabilityCatalogEntry:
    """Map one skill capability from a bundle row to a federated catalog entry."""
    catalog_manifest = catalog_manifest_for_skill_id(skill_id)
    version_label = catalog_manifest.version if catalog_manifest is not None else None
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=_BUILTIN_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id=skill_id,
            ),
        ),
        provenance=CapabilityProvenance(
            source=_BUILTIN_SOURCE,
            version_label=version_label,
            version_binding_disposition=SkillVersionBindingDisposition.MATERIALIZED,
            package_reference=bundle.bundle_id,
        ),
        display_label=skill_id,
    )


class SkillBundleCatalogSource:
    """Read-only adapter over the in-memory skill bundle catalog."""

    @property
    def source_id(self) -> str:
        return SKILL_BUILTIN_CATALOG_SOURCE_ID

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        entries: list[CapabilityCatalogEntry] = []
        for bundle in iter_bundles():
            for skill_id in bundle.skill_ids:
                entries.append(project_skill_bundle_entry(bundle, skill_id))
        return tuple(entries)
