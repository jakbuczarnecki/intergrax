# © Artur Czarnecki. All rights reserved.

"""Federated catalog provider tests."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.federated_catalog import FederatedCatalogSourceProvider

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _Provider:
    def __init__(self, source_id: str, entries: list[AgentCatalogEntry]) -> None:
        self._source_id = source_id
        self._entries = entries

    @property
    def catalog_source_id(self) -> str:
        return self._source_id

    def list_entries(self, filters: object | None = None) -> list[AgentCatalogEntry]:
        del filters
        return list(self._entries)

    def resolve_package(self, entry: AgentCatalogEntry, *, version_selector: str) -> object:
        del entry, version_selector
        raise NotImplementedError

    def health(self) -> None:
        return None


def _entry(entry_id: str, source_id: str) -> AgentCatalogEntry:
    return AgentCatalogEntry(
        catalog_entry_id=entry_id,
        catalog_source=CatalogSourceIdentity(
            catalog_source_id=source_id,
            provider_kind=CatalogProviderKind.BUILTIN,
        ),
        display_name=entry_id,
        package_id_line=f"pkg-{entry_id}",
    )


def test_federated_catalog_preserves_source_qualified_identity() -> None:
    builtin = _Provider("builtin-1", [_entry("a", "builtin-1")])
    private = _Provider(
        "enterprise-private-1",
        [
            AgentCatalogEntry(
                catalog_entry_id="private-a",
                catalog_source=CatalogSourceIdentity(
                    catalog_source_id="enterprise-private-1",
                    provider_kind=CatalogProviderKind.ENTERPRISE_PRIVATE,
                ),
                display_name="Private",
                package_id_line="pkg-a",
            )
        ],
    )
    federated = FederatedCatalogSourceProvider((builtin, private))
    entries = federated.list_entries()
    assert len(entries) == 2
    source_ids = {entry.catalog_source.catalog_source_id for entry in entries}
    assert source_ids == {"builtin-1", "enterprise-private-1"}


def test_federated_catalog_deterministic_ordering() -> None:
    z = _Provider("source-z", [_entry("z", "source-z")])
    a = _Provider("source-a", [_entry("a", "source-a")])
    federated = FederatedCatalogSourceProvider((z, a))
    first = [entry.catalog_entry_id for entry in federated.list_entries()]
    second = [entry.catalog_entry_id for entry in federated.list_entries()]
    assert first == second == ["a", "z"]


def test_federated_catalog_rejects_duplicate_child_source_ids() -> None:
    one = _Provider("dup", [_entry("one", "dup")])
    two = _Provider("dup", [_entry("two", "dup")])
    with pytest.raises(ValueError, match="duplicate catalog_source_id"):
        FederatedCatalogSourceProvider((one, two))
