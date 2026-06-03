# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.catalog_bootstrap import (
    bootstrap_catalogs,
    reset_tier0_catalog_bootstrap_for_tests,
)
from intergrax.core.catalog_conflict import (
    catalog_registration_override,
    entry_point_conflict_policy,
)
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.examples.custom_memory_kv import CustomMemoryKvPlugin
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog, get_entry
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_catalogs() -> None:
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()
    yield
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()


class _SqliteOverridePlugin:
    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        return SQLITE

    @classmethod
    def create_integration(cls, **kwargs: object) -> object:
        _ = kwargs
        return object()


def test_entry_point_conflict_policy_maps_warn_override() -> None:
    assert entry_point_conflict_policy("warn_override") == "override"
    assert entry_point_conflict_policy("error") == "error"


def test_catalog_registration_override_raises_on_error() -> None:
    with pytest.raises(ValueError, match="already registered"):
        catalog_registration_override(
            slug="sqlite",
            slug_registered=True,
            on_conflict="error",
            catalog_kind="integration",
            plugin_type=_SqliteOverridePlugin,
        )


def test_bootstrap_warn_override_replaces_shipped_slug(caplog: pytest.LogCaptureFixture) -> None:
    bootstrap_catalogs(
        register_shipped=True,
        integration_plugins=(_SqliteOverridePlugin,),
        on_conflict="warn_override",
    )
    assert any("Overriding catalog slug" in record.message for record in caplog.records)
    assert get_entry("sqlite").slug == "sqlite"


def test_bootstrap_skip_does_not_register_duplicate_plugin() -> None:
    bootstrap_catalogs(register_shipped=False, integration_plugins=(CustomMemoryKvPlugin,))
    result = bootstrap_catalogs(
        register_shipped=False,
        integration_plugins=(CustomMemoryKvPlugin,),
        on_conflict="skip",
    )
    assert result.integration_plugins == 0
