# © Artur Czarnecki. All rights reserved.

"""P2-003-B1-R2 — H-INT-GRAPH explicit contract bootstrap regression tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.providers.graph_store.arangodb.contract_spec import (
    CONTRACT_SPEC as ARANGODB_CONTRACT_SPEC,
)
from intergrax.integrations.providers.graph_store.neptune.contract_spec import (
    CONTRACT_SPEC as NEPTUNE_CONTRACT_SPEC,
)
from intergrax.integrations.providers.graph_store.orientdb.contract_spec import (
    CONTRACT_SPEC as ORIENTDB_CONTRACT_SPEC,
)
from intergrax.integrations.registry.bootstrap_h_int_graph import register_h_int_graph_integrations
from intergrax.integrations.registry.catalog import clear_catalog, get_entry
from intergrax.runtime.integrations.registry_v2 import build_integration_registration

pytestmark = pytest.mark.unit

H_INT_GRAPH_SLUGS = ("neptune", "orientdb", "arangodb")


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    yield
    clear_catalog()


def test_h_int_graph_bootstrap_registers_all_three_providers() -> None:
    register_h_int_graph_integrations()
    for slug in H_INT_GRAPH_SLUGS:
        entry = get_entry(slug)
        assert entry is not None
        assert entry.contract_specs
        assert entry.contract_specs[0].category == "graph_store"
        assert entry.contract_specs[0].metadata.get("source") == "explicit_provider_declaration"


@pytest.mark.parametrize("slug", H_INT_GRAPH_SLUGS)
def test_h_int_graph_registration_uses_explicit_specs(slug: str) -> None:
    register_h_int_graph_integrations()
    entry = get_entry(slug)
    assert entry.contract_specs
    assert entry.contract_specs[0].metadata.get("source") == "explicit_provider_declaration"


@pytest.mark.parametrize("slug", H_INT_GRAPH_SLUGS)
def test_h_int_graph_registration_does_not_execute_catalog_factory(
    slug: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_module_name = f"intergrax.integrations.providers.graph_store.{slug}.register"
    import importlib

    register_module = importlib.import_module(register_module_name)
    register_fn_name = f"register_{slug}_integration"
    register_fn = getattr(register_module, register_fn_name)
    from intergrax.integrations.registry import plugin_register

    original_rfm = plugin_register.register_from_manifest

    def tracking_rfm(
        manifest: object,
        factory: Callable[..., Any],
        **kwargs: Any,
    ) -> object:
        factory_mock = MagicMock(wraps=factory)
        result = original_rfm(manifest, factory_mock, **kwargs)
        factory_mock.assert_not_called()
        return result

    monkeypatch.setattr(register_module, "register_from_manifest", tracking_rfm)
    register_fn()


def test_h_int_graph_providers_have_provider_owned_explicit_specs() -> None:
    assert NEPTUNE_CONTRACT_SPEC.provider_id == "neptune"
    assert ORIENTDB_CONTRACT_SPEC.provider_id == "orientdb"
    assert ARANGODB_CONTRACT_SPEC.provider_id == "arangodb"
    for spec in (NEPTUNE_CONTRACT_SPEC, ORIENTDB_CONTRACT_SPEC, ARANGODB_CONTRACT_SPEC):
        assert spec.category == "graph_store"
        assert spec.metadata.get("source") == "explicit_provider_declaration"


def test_h_int_graph_providers_use_category_gated_explicit_specs() -> None:
    from intergrax.integrations.registry.contract_spec import typed_contract_categories

    assert "graph_store" in typed_contract_categories()


def test_registry_v2_derives_h_int_graph_rows() -> None:
    register_h_int_graph_integrations()
    for slug in H_INT_GRAPH_SLUGS:
        registration = build_integration_registration(slug)
        assert registration.provider_id == slug
        assert registration.category == "graph_store"


def test_h_int_graph_raw_factories_remain_callable_with_injected_client() -> None:
    from typing import Any

    class _FakeCypherGraphClient:
        def run(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
            del statement, parameters
            return [{"n": {"id": "1"}}]

        def get_node(self, node_id: str) -> dict[str, Any]:
            return {"id": node_id, "labels": ["Entity"], "properties": {"name": "node"}}

    class _FakeArangoGraphClient:
        def run_aql(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
            del statement, parameters
            return [{"id": "1"}]

        def get_document(self, node_id: str) -> dict[str, Any]:
            return {"id": node_id, "labels": ["Entity"], "properties": {"name": "node"}}

    from intergrax.integrations.providers.graph_store.arangodb.bundle import create_arangodb_graph_store
    from intergrax.integrations.providers.graph_store.neptune.bundle import create_neptune_graph_store
    from intergrax.integrations.providers.graph_store.orientdb.bundle import create_orientdb_graph_store

    for factory in (create_neptune_graph_store, create_orientdb_graph_store):
        store = factory(client=_FakeCypherGraphClient())
        assert store.get_node("1") is not None

    arango_store = create_arangodb_graph_store(client=_FakeArangoGraphClient())
    assert arango_store.get_node("1") is not None
