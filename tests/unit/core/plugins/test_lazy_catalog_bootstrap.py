# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.catalog_bootstrap import (
    bootstrap_catalogs,
    reset_tier0_catalog_bootstrap_for_tests,
)
from intergrax.core.catalog_snapshot import snapshot_catalogs
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset() -> None:
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


def test_lazy_tool_and_skill_bundles() -> None:
    bootstrap_catalogs(
        register_shipped=True,
        tool_bundle_ids=("rag",),
        skill_bundle_ids=("harness",),
    )
    snap = snapshot_catalogs()
    assert snap.tool_bundle_ids == ("rag",)
    assert snap.skill_bundle_ids == ("harness",)
    assert "rag.retrieve" in snap.tool_ids


def test_integration_preset_core() -> None:
    bootstrap_catalogs(register_shipped=True, integration_preset="core")
    snap = snapshot_catalogs()
    assert len(snap.integration_slugs) < 50
    assert "sqlite" in snap.integration_slugs
    assert "redis" in snap.integration_slugs
