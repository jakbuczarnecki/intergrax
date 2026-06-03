# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.catalog_bootstrap import (
    bootstrap_catalogs,
    reset_tier0_catalog_bootstrap_for_tests,
)
from intergrax.core.catalog_snapshot import snapshot_catalogs
from intergrax.integrations.examples.custom_memory_kv import MANIFEST, CustomMemoryKvPlugin
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
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


def test_bootstrap_catalogs_registers_shipped_bundles() -> None:
    bootstrap_catalogs(register_shipped=True)
    snap = snapshot_catalogs()
    assert "rag" in snap.tool_bundle_ids
    assert "harness" in snap.skill_bundle_ids
    assert snap.integration_slugs


def test_bootstrap_catalogs_idempotent_shipped() -> None:
    bootstrap_catalogs(register_shipped=True)
    first = snapshot_catalogs()
    bootstrap_catalogs(register_shipped=True)
    second = snapshot_catalogs()
    assert first == second


def test_bootstrap_registers_explicit_integration_plugin() -> None:
    result = bootstrap_catalogs(
        register_shipped=False,
        integration_plugins=(CustomMemoryKvPlugin,),
    )
    assert result.integration_plugins == 1
    snap = snapshot_catalogs()
    assert MANIFEST.slug in snap.integration_slugs
