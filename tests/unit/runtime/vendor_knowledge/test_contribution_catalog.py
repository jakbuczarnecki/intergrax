from __future__ import annotations

import importlib.metadata
from dataclasses import replace

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    VENDOR_KNOWLEDGE_PROVIDER_ENTRY_POINT_GROUP,
    VendorKnowledgeContributionCatalog,
    VendorKnowledgePluginConflict,
    VendorKnowledgePluginLoadError,
    build_default_vendor_knowledge_contribution_catalog,
    build_vendor_knowledge_adapter_registry,
    build_vendor_knowledge_source_plugin_registry,
)
from intergrax.runtime.vendor_knowledge.slack_contribution import (
    build_slack_vendor_knowledge_contribution,
)


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


def _empty_contribution(
    provider_id: str = "fixture",
) -> VendorKnowledgeProviderContribution:
    return VendorKnowledgeProviderContribution(
        provider_id=provider_id,
        integration_category=IntegrationCategory.ISSUE_TRACKER,
    )


def test_builtin_catalog_is_deterministic_and_databricks_has_connection_only() -> None:
    first = build_default_vendor_knowledge_contribution_catalog()
    second = build_default_vendor_knowledge_contribution_catalog()

    assert [item.provider_key for item in first.list_contributions()] == [
        item.provider_key for item in second.list_contributions()
    ]
    databricks = next(
        item
        for item in first.list_contributions()
        if item.provider_id == "databricks"
    )
    assert not databricks.adapters
    assert not databricks.source_plugins
    assert not databricks.indexed_materializers
    assert not databricks.live_contributions


def test_catalog_identical_registration_is_idempotent_and_conflicts_fail_closed() -> None:
    contribution = _empty_contribution()
    catalog = VendorKnowledgeContributionCatalog((contribution,))
    catalog.register(contribution)
    assert catalog.list_contributions() == (contribution,)

    with pytest.raises(VendorKnowledgePluginConflict, match="conflicting_provider"):
        catalog.register(replace(contribution, contract_version="fixture.v2"))


def test_opt_in_entry_point_contribution_reaches_both_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contribution = build_slack_vendor_knowledge_contribution()
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint("fixture-slack", contribution),)),
    )

    catalog = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=True,
        built_in_builders=(),
    )

    assert catalog.list_contributions() == (contribution,)
    assert len(build_vendor_knowledge_adapter_registry(catalog).registered_keys()) == 1
    assert len(build_vendor_knowledge_source_plugin_registry(catalog).list_plugins()) == 1


def test_external_builtin_identity_conflict_is_rejected(
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


@pytest.mark.parametrize(
    "target",
    [object(), lambda: object(), lambda: (_ for _ in ()).throw(RuntimeError("secret"))],
)
def test_malformed_external_plugin_fails_without_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint("fixture-invalid", target),)),
    )

    with pytest.raises(VendorKnowledgePluginLoadError) as error:
        build_default_vendor_knowledge_contribution_catalog(
            discover_entry_points=True,
            built_in_builders=(),
        )
    assert "secret" not in str(error.value)
