# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Enterprise-private Skill catalog → capability catalog entry adapter (Stage 7)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.capability_catalog.adapters._private_validation import (
    validate_enterprise_private_source,
    validate_unique_source_logical_records,
)
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.identity import (
    CapabilityDiscoveryIdentity,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance
from intergrax.skills.core.contracts import SkillManifest


@dataclass(frozen=True, slots=True)
class PrivateSkillCatalogPackage:
    """Read-only Skill catalog package metadata supplied by an enterprise-private source."""

    manifest: SkillManifest
    package_reference: str | None = None
    content_digest: str | None = None
    publisher: str | None = None
    display_label: str | None = None


def project_private_skill_package(
    source: CapabilitySourceIdentity,
    package: PrivateSkillCatalogPackage,
) -> CapabilityCatalogEntry:
    """Map one private Skill catalog package to a federated catalog entry.

    Requires an ENTERPRISE_PRIVATE source identity.
    """
    validate_enterprise_private_source(source)
    skill_id = package.manifest.skill_id
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id=skill_id,
            ),
        ),
        provenance=CapabilityProvenance(
            source=source,
            version_label=package.manifest.version,
            package_reference=package.package_reference,
            content_digest=package.content_digest,
            publisher=package.publisher,
        ),
        display_label=package.display_label or skill_id,
    )


class PrivateSkillCapabilityCatalogSource:
    """Read-only adapter over in-memory enterprise-private Skill catalog metadata."""

    def __init__(
        self,
        *,
        source: CapabilitySourceIdentity,
        packages: tuple[PrivateSkillCatalogPackage, ...],
    ) -> None:
        validate_enterprise_private_source(source)
        validate_unique_source_logical_records(
            records=tuple(
                (package.manifest.skill_id, package.manifest.version)
                for package in packages
            ),
            record_label="skill",
        )
        self._source = source
        self._packages = packages

    @property
    def source_id(self) -> str:
        return self._source.source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        entries = [
            project_private_skill_package(self._source, package)
            for package in self._packages
        ]
        return tuple(sorted(entries, key=lambda entry: entry.identity.sort_key))
