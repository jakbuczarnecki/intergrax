# © Artur Czarnecki. All rights reserved.

"""ME-7 release identity contract tests."""

from __future__ import annotations

from intergrax.contracts.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)

pytestmark = __import__("pytest").mark.unit


def _entry() -> CapabilityCatalogEntry:
    source = CapabilitySourceIdentity(
        source_id="official.release.test",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.release.sample",
            ),
        ),
        provenance=CapabilityProvenance(
            source=source,
            publisher="pub-1",
            version_label="4.5.6",
            content_digest="sha256:deadbeef",
            package_reference="pkg:tool/sample@4.5.6",
        ),
    )


def test_release_identity_from_catalog_entry_is_deterministic() -> None:
    entry = _entry()
    first = CapabilityReleaseIdentity.from_catalog_entry(entry)
    second = CapabilityReleaseIdentity.from_catalog_entry(entry)
    assert first == second
    assert first.release_sort_key == second.release_sort_key


def test_release_identity_separates_discovery_from_version_facts() -> None:
    release = CapabilityReleaseIdentity.from_catalog_entry(_entry())
    assert release.discovery.logical.logical_id == "tools.release.sample"
    assert release.publisher == "pub-1"
    assert release.version_label == "4.5.6"
    assert release.release_sort_key[-4:] == (
        "pub-1",
        "4.5.6",
        "sha256:deadbeef",
        "pkg:tool/sample@4.5.6",
    )
