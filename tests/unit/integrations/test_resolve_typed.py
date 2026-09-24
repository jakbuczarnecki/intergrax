# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.examples.custom_memory_kv import CustomMemoryKvPlugin
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.plugin_register import register_integration_plugin
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.integrations.registry.resolve_typed import (
    resolve_contract,
    resolve_key_value_cache,
)
from intergrax.runtime.integrations.contract_metadata import CategoryIntegrationInstance

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean() -> None:
    clear_catalog()
    yield
    clear_catalog()


def test_resolve_contract_without_expected_returns_category_instance() -> None:
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(key_value_cache=CustomMemoryKvPlugin)
    result: CategoryIntegrationInstance = resolve_contract(
        profile,
        IntegrationCategory.KEY_VALUE_CACHE,
    )
    assert isinstance(result, KeyValueCache)
    result.set("t1", "k", b"v")
    assert result.get("t1", "k") == b"v"


def test_resolve_contract_with_expected_returns_narrow_type() -> None:
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(key_value_cache=CustomMemoryKvPlugin)
    cache: KeyValueCache = resolve_contract(
        profile,
        IntegrationCategory.KEY_VALUE_CACHE,
        expected=KeyValueCache,
    )
    cache.set("t1", "k", b"v")
    assert cache.get("t1", "k") == b"v"


def test_resolve_key_value_cache_typed() -> None:
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(key_value_cache=CustomMemoryKvPlugin)
    cache = resolve_key_value_cache(profile)
    cache.set("t1", "k", b"v")
    assert cache.get("t1", "k") == b"v"


class _StubExternalWorkIntegration:
    def discover(self, request): ...

    def create_work(self, request): ...

    def get_work(self, correlation): ...

    def get_quote(self, correlation): ...

    def submit_quote_acceptance(
        self,
        correlation,
        acceptance,
        *,
        idempotency_key: str,
    ): ...

    def cancel_work(self, correlation, *, idempotency_key: str, reason: str = ""): ...

    def get_timeline(self, correlation, *, limit: int = 50): ...

    def get_deliverables(self, correlation): ...

    def get_evidence(self, correlation): ...


def test_resolve_contract_external_work_without_expected() -> None:
    stub = _StubExternalWorkIntegration()
    assert isinstance(stub, ExternalWorkIntegration)
    profile = IntegrationProfile(external_work=stub)
    result: CategoryIntegrationInstance = resolve_contract(
        profile,
        IntegrationCategory.EXTERNAL_WORK,
    )
    assert isinstance(result, ExternalWorkIntegration)
    assert result is stub


def test_resolve_contract_type_mismatch_raises() -> None:
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(key_value_cache=CustomMemoryKvPlugin)
    from intergrax.integrations.contracts.relational_store import RelationalStore

    with pytest.raises(TypeError, match="expected"):
        resolve_contract(
            profile,
            IntegrationCategory.KEY_VALUE_CACHE,
            expected=RelationalStore,
        )
