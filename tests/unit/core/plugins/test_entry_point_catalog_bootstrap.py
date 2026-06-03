# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.catalog_bootstrap import bootstrap_catalogs, reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.catalog_snapshot import snapshot_catalogs
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("catalog_fixture_installed")]


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


def test_bootstrap_discovers_fixture_plugins_via_entry_points() -> None:
    result = bootstrap_catalogs(
        register_shipped=False,
        discover_entry_points=True,
    )
    assert result.integration_plugins >= 1
    assert result.tool_plugins >= 1
    assert result.skill_plugins >= 1
    snap = snapshot_catalogs()
    assert "fixture_ep_kv" in snap.integration_slugs
    assert "fixture_ep" in snap.tool_bundle_ids
    assert "fixture_ep" in snap.skill_bundle_ids
