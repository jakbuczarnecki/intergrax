# © Artur Czarnecki. All rights reserved.

"""Enterprise-private Tool catalog adapter tests (Stage 7)."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog.adapters.private_tool import (
    PrivateToolCapabilityCatalogSource,
    PrivateToolCatalogRecord,
    project_private_tool_record,
)
from intergrax.capability_catalog.errors import CapabilityCatalogConfigurationError
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity import CapabilitySourceIdentity

pytestmark = pytest.mark.unit


def _enterprise_tool_source(source_id: str = "enterprise.acme.tools") -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id=source_id,
        source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
    )


def test_private_tool_source_returns_source_qualified_entries() -> None:
    source = _enterprise_tool_source()
    record = PrivateToolCatalogRecord(
        logical_id="tool.enterprise.search",
        version_label="3.2.1",
        package_reference="private://tools/search/3.2.1",
        content_digest="sha256:abc",
        publisher="acme-tools",
    )
    source_impl = PrivateToolCapabilityCatalogSource(
        source=source,
        records=(record,),
    )
    entries = source_impl.read_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.identity.kind is CapabilityKind.TOOL
    assert entry.identity.source == source
    assert entry.identity.source.source_kind is CapabilitySourceKind.ENTERPRISE_PRIVATE
    assert entry.identity.logical.logical_id == "tool.enterprise.search"
    assert entry.provenance.source == source
    assert entry.provenance.version_label == "3.2.1"
    assert entry.provenance.package_reference == "private://tools/search/3.2.1"
    assert entry.provenance.content_digest == "sha256:abc"
    assert entry.provenance.publisher == "acme-tools"


def test_private_tool_source_id_matches_configured_identity() -> None:
    source = _enterprise_tool_source("private.airgap.factory.tools")
    source_impl = PrivateToolCapabilityCatalogSource(
        source=source,
        records=(
            PrivateToolCatalogRecord(logical_id="tool.enterprise.search"),
        ),
    )
    assert source_impl.source_id == "private.airgap.factory.tools"


def test_private_tool_source_rejects_non_private_source_kind() -> None:
    with pytest.raises(CapabilityCatalogConfigurationError, match="ENTERPRISE_PRIVATE"):
        PrivateToolCapabilityCatalogSource(
            source=CapabilitySourceIdentity(
                source_id="official.tools",
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            records=(PrivateToolCatalogRecord(logical_id="tool.enterprise.search"),),
        )


def test_private_tool_source_rejects_duplicate_logical_id() -> None:
    source = _enterprise_tool_source()
    with pytest.raises(CapabilityCatalogConfigurationError, match="duplicate tool logical_id"):
        PrivateToolCapabilityCatalogSource(
            source=source,
            records=(
                PrivateToolCatalogRecord(logical_id="tool.enterprise.search"),
                PrivateToolCatalogRecord(logical_id="tool.enterprise.search"),
            ),
        )


def test_private_tool_source_rejects_conflicting_versions_for_same_logical_id() -> None:
    source = _enterprise_tool_source()
    with pytest.raises(CapabilityCatalogConfigurationError, match="conflicting version metadata"):
        PrivateToolCapabilityCatalogSource(
            source=source,
            records=(
                PrivateToolCatalogRecord(
                    logical_id="tool.enterprise.search",
                    version_label="1.0.0",
                ),
                PrivateToolCatalogRecord(
                    logical_id="tool.enterprise.search",
                    version_label="2.0.0",
                ),
            ),
        )


def test_private_tool_source_read_entries_is_deterministic() -> None:
    source = _enterprise_tool_source()
    records = (
        PrivateToolCatalogRecord(logical_id="tool.z.last"),
        PrivateToolCatalogRecord(logical_id="tool.a.first"),
    )
    source_impl = PrivateToolCapabilityCatalogSource(source=source, records=records)
    first = source_impl.read_entries()
    second = source_impl.read_entries()
    assert first == second
    assert [entry.identity.logical.logical_id for entry in first] == [
        "tool.a.first",
        "tool.z.last",
    ]


def test_project_private_tool_record_preserves_optional_version_absence() -> None:
    source = _enterprise_tool_source()
    entry = project_private_tool_record(
        source,
        PrivateToolCatalogRecord(logical_id="tool.enterprise.search"),
    )
    assert entry.provenance.version_label is None


def test_project_private_tool_record_rejects_non_private_source_kind() -> None:
    source = CapabilitySourceIdentity(
        source_id="tools.catalog.builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
    )
    record = PrivateToolCatalogRecord(logical_id="tool.enterprise.search")
    with pytest.raises(CapabilityCatalogConfigurationError, match="ENTERPRISE_PRIVATE"):
        project_private_tool_record(source, record)


def test_project_private_tool_record_projects_source_qualified_entry() -> None:
    source = _enterprise_tool_source()
    record = PrivateToolCatalogRecord(
        logical_id="tool.enterprise.search",
        version_label="3.2.1",
        package_reference="private://tools/search/3.2.1",
    )
    entry = project_private_tool_record(source, record)
    assert entry.identity.source == source
    assert entry.provenance.source == source
    assert entry.identity.logical.logical_id == "tool.enterprise.search"
    assert entry.provenance.version_label == "3.2.1"
    assert entry.provenance.package_reference == "private://tools/search/3.2.1"
