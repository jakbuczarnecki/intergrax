# © Artur Czarnecki. All rights reserved.

"""Agent catalog adapter contract tests (Stage 2)."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.capability_catalog.adapters.agent import (
    AgentCatalogCapabilitySource,
    project_agent_catalog_entry,
)
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind

pytestmark = pytest.mark.unit


class _Provider:
    def __init__(self, entries: list[AgentCatalogEntry]) -> None:
        self._entries = entries

    @property
    def catalog_source_id(self) -> str:
        return "official-agents"

    def list_entries(self, filters: object | None = None) -> list[AgentCatalogEntry]:
        del filters
        return list(self._entries)


def _agent_entry() -> AgentCatalogEntry:
    return AgentCatalogEntry(
        catalog_entry_id="agent.dispute_analyst",
        catalog_source=CatalogSourceIdentity(
            catalog_source_id="official-agents",
            provider_kind=CatalogProviderKind.OFFICIAL_CATALOG,
        ),
        display_name="Dispute Analyst",
        publisher="intergrax",
        package_id_line="pkg.dispute_analyst",
        version_channel_refs=(),
    )


def test_project_agent_catalog_entry_preserves_identity_source_and_provenance() -> None:
    projected = project_agent_catalog_entry(_agent_entry())
    assert projected.identity.kind is CapabilityKind.AGENT
    assert projected.identity.logical.logical_id == "agent.dispute_analyst"
    assert projected.identity.source.source_id == "official-agents"
    assert projected.identity.source.source_kind is CapabilitySourceKind.OFFICIAL
    assert projected.provenance.package_reference == "pkg.dispute_analyst"
    assert projected.provenance.publisher == "intergrax"
    assert projected.display_label == "Dispute Analyst"


def test_agent_catalog_capability_source_maps_provider_entries() -> None:
    source = AgentCatalogCapabilitySource(_Provider([_agent_entry()]))
    entries = source.read_entries()
    assert len(entries) == 1
    assert entries[0].identity.logical.logical_id == "agent.dispute_analyst"
