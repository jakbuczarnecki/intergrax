# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.catalog_bootstrap import bootstrap_catalogs, reset_tier0_catalog_bootstrap_for_tests
from intergrax.integrations._shared.conformance import assert_key_value_cache
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("catalog_fixture_installed")]


@pytest.fixture(autouse=True)
def _clean() -> None:
    clear_catalog()
    reset_default_integrations_state()
    reset_tier0_catalog_bootstrap_for_tests()
    yield
    clear_catalog()
    reset_default_integrations_state()
    reset_tier0_catalog_bootstrap_for_tests()


def test_fixture_integration_resolves_via_entry_point() -> None:
    bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    profile = IntegrationProfile(key_value_cache="fixture_ep_kv")
    cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
    assert_key_value_cache(cache)
