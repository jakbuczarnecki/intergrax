# © Artur Czarnecki. All rights reserved.

"""VK-EXT-3 unit qualification for the reference external provider (G1, G2)."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    VENDOR_KNOWLEDGE_PROVIDER_ENTRY_POINT_GROUP,
    VendorKnowledgePluginConflict,
    VendorKnowledgePluginLoadError,
    build_default_vendor_knowledge_contribution_catalog,
    build_vendor_knowledge_adapter_registry,
    build_vendor_knowledge_source_plugin_registry,
    discover_vendor_knowledge_contributions,
)
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeMode
from intergrax.runtime.vendor_knowledge.slack_contribution import (
    build_slack_vendor_knowledge_contribution,
)

from acme_reference_vk_plugin.constants import (
    ACME_DOCUMENTS_SOURCE_KIND,
    ACME_REFERENCE_PROVIDER_ID,
)
from acme_reference_vk_plugin.contribution import build_acme_reference_contribution

pytestmark = [
    pytest.mark.unit,
    pytest.mark.usefixtures("acme_reference_vk_plugin_installed"),
]


class _EntryPoint:
    def __init__(self, name: str, target: object) -> None:
        self.name = name
        self.value = "fixture:target"
        self._target = target

    def load(self) -> object:
        return self._target


class _EntryPoints:
    def __init__(self, entries: tuple[_EntryPoint, ...]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> tuple[_EntryPoint, ...]:
        assert group == VENDOR_KNOWLEDGE_PROVIDER_ENTRY_POINT_GROUP
        return self._entries


def test_reference_contribution_identity_and_modes() -> None:
    contribution = build_acme_reference_contribution()

    assert contribution.provider_id == ACME_REFERENCE_PROVIDER_ID
    assert contribution.integration_category is IntegrationCategory.WIKI_KNOWLEDGE
    assert len(contribution.adapters) == 1
    assert contribution.adapters[0].provider_id == ACME_REFERENCE_PROVIDER_ID
    assert len(contribution.source_plugins) == 1
    plugin = contribution.source_plugins[0]
    assert plugin.identity.source_kind == ACME_DOCUMENTS_SOURCE_KIND
    assert plugin.supports(VendorKnowledgeMode.DURABLE)
    assert plugin.supports(VendorKnowledgeMode.INDEXED)
    assert not plugin.supports(VendorKnowledgeMode.LIVE)
    assert len(contribution.connection_factories) == 1
    assert len(contribution.discovery_contributions) == 1
    assert len(contribution.indexed_materializers) == 1
    assert not contribution.live_contributions


def test_reference_factory_creates_integration_from_credential_ref() -> None:
    factory = build_acme_reference_contribution().connection_factories[0].factory
    integration = factory.create_integration(
        tenant_id="tenant-a",
        connection_ref="conn.acme",
        provider_id=ACME_REFERENCE_PROVIDER_ID,
        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
        credential_ref="secrets/tenant-a/acme",
        credential='{"api_key":"qualification-key"}',
        secret_free_config={"collection_endpoint": "inmemory://collections"},
    )
    documents = integration.list_documents(collection_id="col-ref-qualification-1")
    assert len(documents) == 1


def test_entry_point_discovery_loads_reference_contribution() -> None:
    contributions = discover_vendor_knowledge_contributions()
    assert any(item.provider_id == ACME_REFERENCE_PROVIDER_ID for item in contributions)


def test_enabled_catalog_increments_builtin_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled = build_default_vendor_knowledge_contribution_catalog()
    enabled = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=True,
    )
    assert len(build_vendor_knowledge_adapter_registry(disabled).registered_keys()) == 12
    assert len(build_vendor_knowledge_adapter_registry(enabled).registered_keys()) == 13
    assert len(build_vendor_knowledge_source_plugin_registry(disabled).list_plugins()) == 12
    assert len(build_vendor_knowledge_source_plugin_registry(enabled).list_plugins()) == 13
    assert sum(len(item.connection_factories) for item in disabled.list_contributions()) == 6
    assert sum(len(item.connection_factories) for item in enabled.list_contributions()) == 7


def test_external_discovery_disabled_keeps_reference_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=False,
    )
    assert all(
        item.provider_id != ACME_REFERENCE_PROVIDER_ID
        for item in catalog.list_contributions()
    )


@pytest.mark.parametrize(
    "target",
    [object(), lambda: object()],
)
def test_malformed_external_plugin_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint("broken-acme", target),)),
    )
    with pytest.raises(VendorKnowledgePluginLoadError):
        build_default_vendor_knowledge_contribution_catalog(
            discover_entry_points=True,
            built_in_builders=(),
        )


def test_duplicate_external_provider_key_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints(
            (
                _EntryPoint("acme-a", build_acme_reference_contribution),
                _EntryPoint("acme-b", build_acme_reference_contribution),
            )
        ),
    )
    with pytest.raises(VendorKnowledgePluginConflict):
        build_default_vendor_knowledge_contribution_catalog(
            discover_entry_points=True,
            built_in_builders=(),
        )


def test_builtin_identity_conflict_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints(
            (_EntryPoint("fixture-slack", build_slack_vendor_knowledge_contribution()),)
        ),
    )
    with pytest.raises(VendorKnowledgePluginConflict, match="conflicting_provider"):
        build_default_vendor_knowledge_contribution_catalog(discover_entry_points=True)
