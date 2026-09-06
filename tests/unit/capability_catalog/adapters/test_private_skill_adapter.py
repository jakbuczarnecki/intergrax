# © Artur Czarnecki. All rights reserved.

"""Enterprise-private Skill catalog adapter tests (Stage 7)."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog.adapters.private_skill import (
    PrivateSkillCapabilityCatalogSource,
    PrivateSkillCatalogPackage,
    project_private_skill_package,
)
from intergrax.capability_catalog.errors import CapabilityCatalogConfigurationError
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity import CapabilitySourceIdentity
from intergrax.skills.core.contracts import SkillManifest

pytestmark = pytest.mark.unit


def _enterprise_skill_source(
    source_id: str = "enterprise.acme.skills",
) -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id=source_id,
        source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
    )


def _manifest(*, skill_id: str = "skill.enterprise.research", version: str = "2.4.0") -> SkillManifest:
    return SkillManifest(
        skill_id=skill_id,
        version=version,
        description="Enterprise research skill",
    )


def test_private_skill_source_returns_source_qualified_entry() -> None:
    source = _enterprise_skill_source()
    package = PrivateSkillCatalogPackage(
        manifest=_manifest(version="4.2.0"),
        package_reference="private://skills/research/4.2.0",
        content_digest="sha256:def",
        publisher="acme-skills",
    )
    source_impl = PrivateSkillCapabilityCatalogSource(source=source, packages=(package,))
    entries = source_impl.read_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.identity.kind is CapabilityKind.SKILL
    assert entry.identity.source == source
    assert entry.identity.source.source_kind is CapabilitySourceKind.ENTERPRISE_PRIVATE
    assert entry.identity.logical.logical_id == "skill.enterprise.research"
    assert entry.provenance.source == source
    assert entry.provenance.version_label == "4.2.0"
    assert entry.provenance.version_binding_disposition is None
    assert entry.provenance.package_reference == "private://skills/research/4.2.0"
    assert entry.provenance.content_digest == "sha256:def"
    assert entry.provenance.publisher == "acme-skills"


def test_private_skill_source_projects_manifest_version_without_override() -> None:
    source = _enterprise_skill_source()
    manifest = _manifest(version="2.4.0")
    entry = project_private_skill_package(
        source,
        PrivateSkillCatalogPackage(manifest=manifest),
    )
    assert entry.provenance.version_label == manifest.version


def test_private_skill_source_rejects_non_private_source_kind() -> None:
    with pytest.raises(CapabilityCatalogConfigurationError, match="ENTERPRISE_PRIVATE"):
        PrivateSkillCapabilityCatalogSource(
            source=CapabilitySourceIdentity(
                source_id="official.skills",
                source_kind=CapabilitySourceKind.BUILTIN,
            ),
            packages=(PrivateSkillCatalogPackage(manifest=_manifest()),),
        )


def test_private_skill_source_rejects_conflicting_manifest_versions_for_same_skill() -> None:
    source = _enterprise_skill_source()
    with pytest.raises(CapabilityCatalogConfigurationError, match="conflicting version metadata"):
        PrivateSkillCapabilityCatalogSource(
            source=source,
            packages=(
                PrivateSkillCatalogPackage(manifest=_manifest(version="1.0.0")),
                PrivateSkillCatalogPackage(manifest=_manifest(version="2.0.0")),
            ),
        )


def test_private_skill_source_read_entries_is_idempotent() -> None:
    source = _enterprise_skill_source()
    packages = (
        PrivateSkillCatalogPackage(manifest=_manifest(skill_id="skill.z.last")),
        PrivateSkillCatalogPackage(manifest=_manifest(skill_id="skill.a.first")),
    )
    source_impl = PrivateSkillCapabilityCatalogSource(source=source, packages=packages)
    assert source_impl.read_entries() == source_impl.read_entries()


def test_project_private_skill_package_rejects_non_private_source_kind() -> None:
    source = CapabilitySourceIdentity(
        source_id="skills.catalog.builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
    )
    package = PrivateSkillCatalogPackage(manifest=_manifest())
    with pytest.raises(CapabilityCatalogConfigurationError, match="ENTERPRISE_PRIVATE"):
        project_private_skill_package(source, package)


def test_project_private_skill_package_projects_source_qualified_entry() -> None:
    source = _enterprise_skill_source()
    manifest = _manifest(version="4.2.0")
    package = PrivateSkillCatalogPackage(
        manifest=manifest,
        package_reference="private://skills/research/4.2.0",
    )
    entry = project_private_skill_package(source, package)
    assert entry.identity.source == source
    assert entry.provenance.source == source
    assert entry.identity.logical.logical_id == manifest.skill_id
    assert entry.provenance.version_label == manifest.version
    assert entry.provenance.package_reference == "private://skills/research/4.2.0"
